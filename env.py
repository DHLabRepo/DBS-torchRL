import os
import glob
import shutil
from datetime import datetime
import time
import warnings

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch.utils.data import Dataset, DataLoader

import torchvision

import torchOptics.optics as tt
import torchOptics.metrics as tm

IPS = 256  # 이미지 픽셀 사이즈
CH = 8     # 채널
RW = 800   # 보상

warnings.filterwarnings('ignore')


class BinaryHologramEnv:
    """
    Binary Hologram 환경 클래스.
    gymnasium 의존성 없이 순수 Python으로 구현.
    reset()과 step() 인터페이스는 기존과 동일하게 유지.

    GPU 최적화:
    - state를 GPU 텐서로 유지하여 CPU-GPU 전송 최소화
    - _calculate_pixel_importance를 배치 시뮬레이션으로 병렬 처리
    - step()에서 in-place 픽셀 플립으로 텐서 재생성 방지
    """
    def __init__(self, target_function, trainloader, max_steps=10000, T_PSNR=30, T_steps=1, T_PSNR_DIFF=1/4, num_samples=10000,
                 importance_batch_size=64):

        # 행동 공간 크기: 픽셀 하나를 선택하는 인덱스 (CH * IPS * IPS)
        self.num_pixels = CH * IPS * IPS

        # 타겟 함수와 데이터 로더 설정
        self.target_function = target_function
        self.trainloader = trainloader

        # 환경 설정
        self.max_steps = max_steps
        self.T_PSNR = T_PSNR
        self.T_steps = T_steps
        self.T_PSNR_DIFF_o = T_PSNR_DIFF
        self.T_PSNR_DIFF = None
        self.num_samples = num_samples
        self.target_step = self.T_PSNR_DIFF_o * self.num_samples

        # GPU 배치 시뮬레이션 배치 크기
        self.importance_batch_size = importance_batch_size

        # 학습 상태 초기화
        self.state = None           # GPU 텐서 (1, CH, IPS, IPS) float32
        self.state_np = None        # CPU numpy (관측용 캐시)
        self.state_record = None    # CPU numpy
        self.state_record_gpu = None  # GPU 텐서
        self.observation = None
        self.steps = None
        self.psnr_sustained_steps = None
        self.flip_count = None
        self.start_time = None
        self.next_print_thresholds = 0
        self.total_start_time = None
        self.target_image_np = None
        self.initial_psnr = None

        # 최고 PSNR_DIFF 추적 변수
        self.max_psnr_diff = float('-inf')

        # PSNR 저장 변수
        self.previous_psnr = None

        # 데이터 로더에서 첫 배치 설정
        self.data_iter = iter(self.trainloader)
        self.target_image = None

        # 에피소드 카운트
        self.episode_num_count = 0

        # 광학 시뮬레이션 메타
        self.meta = {'dx': (7.56e-6, 7.56e-6), 'wl': 515e-9}

    def _simulate_and_psnr(self, binary_state, z=2e-3):
        """
        광학 시뮬레이션 + PSNR 계산 헬퍼.
        binary_state: (B, CH, IPS, IPS) GPU 텐서
        Returns: result (B, 1, IPS, IPS), psnr (scalar or tensor)
        """
        binary_tt = tt.Tensor(binary_state, meta=self.meta)
        sim = tt.simulate(binary_tt, z).abs() ** 2
        result = torch.mean(sim, dim=1, keepdim=True)
        return result

    def _calculate_pixel_importance(self, z):
        """
        배치 시뮬레이션으로 픽셀 중요도 계산.
        기존: 10000개 픽셀을 하나씩 순차 시뮬레이션 (매우 느림)
        최적화: importance_batch_size개씩 묶어서 배치 시뮬레이션 (GPU 병렬 처리)
        """
        num_samples = self.num_samples
        batch_size = self.importance_batch_size

        # 랜덤 픽셀 인덱스 생성
        random_actions = np.random.randint(0, self.num_pixels, size=num_samples)
        channels = random_actions // (IPS * IPS)
        pixel_indices = random_actions % (IPS * IPS)
        rows = pixel_indices // IPS
        cols = pixel_indices % IPS

        psnr_changes = np.zeros(num_samples, dtype=np.float64)
        positive_psnr_sum = 0.0

        # 배치 단위로 처리
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            cur_batch = end - start

            # 현재 state를 배치만큼 복제 (cur_batch, CH, IPS, IPS)
            batch_states = self.state.expand(cur_batch, -1, -1, -1).clone()

            # 각 복제본에서 해당 픽셀 플립
            for i in range(cur_batch):
                idx = start + i
                c, r, co = int(channels[idx]), int(rows[idx]), int(cols[idx])
                batch_states[i, c, r, co] = 1.0 - batch_states[i, c, r, co]

            # 배치 시뮬레이션 수행
            batch_tt = tt.Tensor(batch_states, meta=self.meta)
            sim_batch = tt.simulate(batch_tt, z).abs() ** 2
            result_batch = torch.mean(sim_batch, dim=1, keepdim=True)  # (cur_batch, 1, IPS, IPS)

            # 배치 PSNR 계산 (각 배치 요소별)
            target_expanded = self.target_image.expand(cur_batch, -1, -1, -1)
            for i in range(cur_batch):
                psnr_val = tt.relativeLoss(result_batch[i:i+1], target_expanded[i:i+1], tm.get_PSNR)
                change = psnr_val - self.initial_psnr
                psnr_changes[start + i] = float(change)
                if change > 0:
                    positive_psnr_sum += float(change)

        # 다항식 보상 함수 계산
        step_poly = np.array([num_samples, num_samples*90/100, num_samples*80/100, num_samples*50/100, num_samples*25/100, 1])
        rewards_poly = np.array([-0.5, -0.48, -0.45, -0.35, 0, 1])
        degree_poly = len(step_poly) - 1
        coefficients_poly = np.polyfit(step_poly, rewards_poly, degree_poly)
        poly_reward = np.poly1d(coefficients_poly)

        print("Polynomial Reward Function Equation:")
        print(poly_reward)

        # PSNR 변화량을 기준으로 순위 매기기 (오름차순 정렬)
        sorted_indices = np.argsort(psnr_changes)
        importance_ranks = np.zeros(num_samples)

        for rank, idx in enumerate(sorted_indices):
            x_val = num_samples - (num_samples - 1) * (rank / (num_samples - 1))
            importance_ranks[idx] = poly_reward(x_val)

        return psnr_changes.tolist(), importance_ranks, positive_psnr_sum

    def reset(self, seed=None, options=None, z=2e-3):
        torch.cuda.empty_cache()

        self.episode_num_count += 1

        # 이터레이터에서 다음 데이터를 가져옴
        try:
            self.target_image, self.current_file = next(self.data_iter)
        except StopIteration:
            print(f"\033[40;93m[INFO] Reached the end of dataset. Restarting from the beginning.\033[0m")
            self.data_iter = iter(self.trainloader)
            self.target_image, self.current_file = next(self.data_iter)

        print(f"\033[40;93m[Episode Start] Currently using dataset file: {self.current_file}, Episode count: {self.episode_num_count}\033[0m")

        self.target_image = self.target_image.cuda()
        self.target_image_np = self.target_image.cpu().numpy()

        with torch.no_grad():
            model_output = self.target_function(self.target_image)
        self.observation = model_output.cpu().numpy()  # (1, CH, IPS, IPS)

        # 매 에피소드마다 초기화
        self.max_psnr_diff = float('-inf')
        self.steps = 0
        self.flip_count = 0
        self.psnr_sustained_steps = 0
        self.next_print_thresholds = 0

        # state를 GPU 텐서로 유지
        self.state = (model_output >= 0.5).float().cuda()  # (1, CH, IPS, IPS) GPU
        self.state_np = self.state.cpu().numpy().astype(np.int8)
        self.state_record = np.zeros_like(self.state_np)
        self.state_record_gpu = torch.zeros_like(self.state)  # GPU

        # 시뮬레이션 (GPU 텐서 직접 사용)
        result = self._simulate_and_psnr(self.state, z)

        # MSE 및 PSNR 계산
        mse = tt.relativeLoss(result, self.target_image, F.mse_loss).detach().cpu().numpy()
        self.initial_psnr = tt.relativeLoss(result, self.target_image, tm.get_PSNR)
        self.previous_psnr = self.initial_psnr

        # 배치 시뮬레이션으로 픽셀 중요도 계산 (GPU 병렬)
        rw_start_time = time.time()
        self.psnr_change_list, self.importance_ranks, positive_psnr_sum = self._calculate_pixel_importance(z)
        data_processing_time = time.time() - rw_start_time
        print(
            f"\nTime taken for psnr_change_list: {data_processing_time:.2f} seconds"
        )

        self.T_PSNR_DIFF = self.T_PSNR_DIFF_o * positive_psnr_sum
        print(f"\033[94m[Dynamic Threshold] T_PSNR_DIFF set to: {self.T_PSNR_DIFF:.6f}\033[0m")

        obs = {"state_record": self.state_record,
               "state": self.state_np,
               "pre_model": self.observation,
               "recon_image": result.cpu().numpy(),
               "target_image": self.target_image_np,
               }

        print(
            f"\033[92mInitial PSNR: {self.initial_psnr:.6f}\033[0m"
            f"\nInitial MSE: {mse:.6f}\033[0m"
        )

        # 다음 출력 기준 PSNR 값 리스트 설정 (0.01 단위로 증가)
        self.next_print_thresholds = [self.initial_psnr + i * 0.01 for i in range(1, 21)]

        self.total_start_time = time.time()

        return obs, {"state": self.state_np}

    def step(self, action, z=2e-3):
        self.steps += 1

        # 행동을 기반으로 픽셀 좌표 계산
        channel = action // (IPS * IPS)
        pixel_index = action % (IPS * IPS)
        row = pixel_index // IPS
        col = pixel_index % IPS

        # GPU 텐서에서 직접 in-place 픽셀 플립 (CPU-GPU 전송 없음)
        self.state[0, channel, row, col] = 1.0 - self.state[0, channel, row, col]
        self.state_record_gpu[0, channel, row, col] += 1.0

        self.flip_count += 1

        # GPU 텐서로 직접 시뮬레이션 (tensor 재생성 없음)
        result_after = self._simulate_and_psnr(self.state, z)
        psnr_after = tt.relativeLoss(result_after, self.target_image, tm.get_PSNR)

        # 관측값 생성 (numpy 변환은 관측 반환 시에만)
        self.state_np = self.state.cpu().numpy().astype(np.int8)
        self.state_record = self.state_record_gpu.cpu().numpy().astype(np.int8)

        obs = {"state_record": self.state_record,
               "state": self.state_np,
               "pre_model": self.observation,
               "recon_image": result_after.cpu().numpy(),
               "target_image": self.target_image_np,
               }

        # PSNR 변화량 계산
        psnr_change = psnr_after - self.previous_psnr
        psnr_diff = psnr_after - self.initial_psnr

        # 가장 유사한 PSNR 변화량의 순위 점수를 보상으로 사용
        closest_index = np.argmin(np.abs(np.array(self.psnr_change_list) - psnr_change))
        reward = self.importance_ranks[closest_index]

        # psnr_change가 음수인 경우 상태 롤백 수행 (GPU에서 직접)
        if psnr_change < 0:
            self.state[0, channel, row, col] = 1.0 - self.state[0, channel, row, col]
            self.state_record_gpu[0, channel, row, col] -= 1.0
            self.flip_count -= 1

            return obs, reward, False, False, {}

        self.max_psnr_diff = max(self.max_psnr_diff, psnr_diff)

        success_ratio = self.flip_count / self.steps if self.steps > 0 else 0

        # 출력 추가 (0.01 PSNR 상승마다 출력)
        while self.next_print_thresholds and psnr_after >= self.next_print_thresholds[0]:
            self.next_print_thresholds.pop(0)
            data_processing_time = time.time() - self.total_start_time
            print(
                f"Step: {self.steps:<6} | Initial PSNR: {self.initial_psnr:.6f}"
                f"\nPSNR After: {psnr_after:.6f} | Change: {psnr_change:.6f} | Diff: {psnr_diff:.6f}"
                f"\nReward: {reward:.2f} | Success Ratio: {success_ratio:.6f} | Flip Count: {self.flip_count}"
                f"\nFlip Pixel: Channel={channel}, Row={row}, Col={col}"
                f"\nTime taken for this data: {data_processing_time:.2f} seconds"
            )

        self.previous_psnr = psnr_after

        if psnr_diff >= self.T_PSNR_DIFF or (psnr_after >= self.T_PSNR and psnr_diff < 0.1):
            data_processing_time = time.time() - self.total_start_time
            print(
                f"Step: {self.steps:<6} | Initial PSNR: {self.initial_psnr:.6f}"
                f"\nPSNR After: {psnr_after:.6f} | Change: {psnr_change:.6f} | Diff: {psnr_diff:.6f}"
                f"\nReward: {reward:.2f} | Success Ratio: {success_ratio:.6f} | Flip Count: {self.flip_count}"
                f"\nFlip Pixel: Channel={channel}, Row={row}, Col={col}"
                f"\nTime taken for this data: {data_processing_time:.2f} seconds"
            )
            self.psnr_sustained_steps += 1

            if self.psnr_sustained_steps >= self.T_steps and psnr_diff >= self.T_PSNR_DIFF:
                m = -1000 / (3 * self.target_step)
                additional_reward = 100 + m * (self.steps - (2 / 5) * self.target_step)
                reward += additional_reward

        if self.steps >= self.max_steps:
            data_processing_time = time.time() - self.total_start_time
            print(
                f"Step: {self.steps:<6} | Initial PSNR: {self.initial_psnr:.6f}"
                f"\nPSNR After: {psnr_after:.6f} | Change: {psnr_change:.6f} | Diff: {psnr_diff:.6f}"
                f"\nReward: {reward:.2f} | Success Ratio: {success_ratio:.6f} | Flip Count: {self.flip_count}"
                f"\nFlip Pixel: Channel={channel}, Row={row}, Col={col}"
                f"\nTime taken for this data: {data_processing_time:.2f} seconds"
            )
            m = -1000 / (3 * self.target_step)
            additional_reward = 100 + m * (self.steps - (2 / 5) * self.target_step)
            reward += additional_reward

        # 성공 종료 조건
        terminated = self.steps >= self.max_steps or self.psnr_sustained_steps >= self.T_steps
        truncated = self.steps >= self.max_steps

        return obs, reward, terminated, truncated, {}
