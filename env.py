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
    Binary Hologram 환경 클래스 (Full GPU).
    모든 관측값, 상태, 보상 계산이 GPU에서 수행됩니다.
    CPU-GPU 전송 완전 제거.
    """
    def __init__(self, target_function, trainloader, max_steps=10000, T_PSNR=30, T_steps=1, T_PSNR_DIFF=1/4, num_samples=10000,
                 importance_batch_size=64):

        self.num_pixels = CH * IPS * IPS
        self.target_function = target_function
        self.trainloader = trainloader

        self.max_steps = max_steps
        self.T_PSNR = T_PSNR
        self.T_steps = T_steps
        self.T_PSNR_DIFF_o = T_PSNR_DIFF
        self.T_PSNR_DIFF = None
        self.num_samples = num_samples
        self.target_step = self.T_PSNR_DIFF_o * self.num_samples

        self.importance_batch_size = importance_batch_size

        # 모든 상태를 GPU 텐서로 관리
        self.state = None               # (1, CH, IPS, IPS) GPU float32
        self.state_record = None         # (1, CH, IPS, IPS) GPU float32
        self.observation = None          # (1, CH, IPS, IPS) GPU float32 (pre_model)
        self.target_image = None         # (1, 1, IPS, IPS) GPU
        self.recon_image = None          # (1, 1, IPS, IPS) GPU

        self.steps = None
        self.psnr_sustained_steps = None
        self.flip_count = None
        self.next_print_thresholds = 0
        self.total_start_time = None
        self.initial_psnr = None
        self.max_psnr_diff = float('-inf')
        self.previous_psnr = None

        # 보상 룩업 테이블 (GPU 텐서)
        self.psnr_change_tensor = None   # (num_samples,) GPU
        self.importance_tensor = None    # (num_samples,) GPU

        self.data_iter = iter(self.trainloader)
        self.episode_num_count = 0
        self.meta = {'dx': (7.56e-6, 7.56e-6), 'wl': 515e-9}

    def _simulate(self, binary_state, z=2e-3):
        """광학 시뮬레이션. Returns: result (B, 1, IPS, IPS) GPU"""
        binary_tt = tt.Tensor(binary_state, meta=self.meta)
        sim = tt.simulate(binary_tt, z).abs() ** 2
        return torch.mean(sim, dim=1, keepdim=True)

    def _calculate_pixel_importance(self, z):
        """배치 시뮬레이션으로 픽셀 중요도 계산. 결과를 GPU 텐서로 반환."""
        num_samples = self.num_samples
        batch_size = self.importance_batch_size

        # 랜덤 픽셀 인덱스 (CPU에서 생성 후 GPU로)
        random_actions = torch.randint(0, self.num_pixels, (num_samples,))
        channels = random_actions // (IPS * IPS)
        pixel_indices = random_actions % (IPS * IPS)
        rows = pixel_indices // IPS
        cols = pixel_indices % IPS

        psnr_changes = torch.zeros(num_samples, device='cuda', dtype=torch.float64)
        positive_psnr_sum = 0.0

        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            cur_batch = end - start

            batch_states = self.state.expand(cur_batch, -1, -1, -1).clone()

            # 배치 픽셀 플립 (GPU에서)
            batch_c = channels[start:end]
            batch_r = rows[start:end]
            batch_co = cols[start:end]
            for i in range(cur_batch):
                batch_states[i, batch_c[i], batch_r[i], batch_co[i]] = 1.0 - batch_states[i, batch_c[i], batch_r[i], batch_co[i]]

            # 배치 시뮬레이션
            result_batch = self._simulate(batch_states, z)
            del batch_states  # 중간 텐서 즉시 해제

            # 배치 PSNR 계산
            target_expanded = self.target_image.expand(cur_batch, -1, -1, -1)
            for i in range(cur_batch):
                psnr_val = tt.relativeLoss(result_batch[i:i+1], target_expanded[i:i+1], tm.get_PSNR)
                change = psnr_val - self.initial_psnr
                psnr_changes[start + i] = change
                if change > 0:
                    positive_psnr_sum += float(change)

            del result_batch, target_expanded  # 메모리 해제
            torch.cuda.empty_cache()

        # 다항식 보상 함수 (CPU numpy로 계산 후 GPU로)
        step_poly = np.array([num_samples, num_samples*90/100, num_samples*80/100, num_samples*50/100, num_samples*25/100, 1])
        rewards_poly = np.array([-0.5, -0.48, -0.45, -0.35, 0, 1])
        degree_poly = len(step_poly) - 1
        coefficients_poly = np.polyfit(step_poly, rewards_poly, degree_poly)
        poly_reward = np.poly1d(coefficients_poly)

        print("Polynomial Reward Function Equation:")
        print(poly_reward)

        # 순위 계산 (GPU에서)
        sorted_indices = torch.argsort(psnr_changes)
        importance_ranks = torch.zeros(num_samples, device='cuda', dtype=torch.float32)

        # poly_reward 값을 미리 GPU 텐서로 계산
        rank_values = torch.arange(num_samples, device='cuda', dtype=torch.float64)
        x_vals = num_samples - (num_samples - 1) * (rank_values / (num_samples - 1))
        # numpy poly1d를 GPU에서 수동 계산
        x_vals_np = x_vals.cpu().numpy()
        poly_results = poly_reward(x_vals_np)
        poly_results_gpu = torch.tensor(poly_results, device='cuda', dtype=torch.float32)

        importance_ranks[sorted_indices] = poly_results_gpu

        # GPU 텐서로 저장
        self.psnr_change_tensor = psnr_changes.float()
        self.importance_tensor = importance_ranks

        return positive_psnr_sum

    def _get_obs(self):
        """현재 상태에서 관측값 dict 생성 (모두 GPU 텐서, clone)"""
        return {
            "state_record": self.state_record.clone(),
            "state": self.state.clone(),
            "pre_model": self.observation,          # 에피소드 내 불변 → clone 불필요
            "recon_image": self.recon_image.clone(),
            "target_image": self.target_image,      # 에피소드 내 불변
        }

    def reset(self, seed=None, options=None, z=2e-3):
        torch.cuda.empty_cache()
        self.episode_num_count += 1

        try:
            self.target_image, self.current_file = next(self.data_iter)
        except StopIteration:
            print(f"\033[40;93m[INFO] Reached the end of dataset. Restarting from the beginning.\033[0m")
            self.data_iter = iter(self.trainloader)
            self.target_image, self.current_file = next(self.data_iter)

        print(f"\033[40;93m[Episode Start] Currently using dataset file: {self.current_file}, Episode count: {self.episode_num_count}\033[0m")

        self.target_image = self.target_image.cuda()

        with torch.no_grad():
            model_output = self.target_function(self.target_image)
        self.observation = model_output  # GPU 텐서 그대로

        # 상태 초기화 (전부 GPU)
        self.max_psnr_diff = float('-inf')
        self.steps = 0
        self.flip_count = 0
        self.psnr_sustained_steps = 0
        self.next_print_thresholds = 0

        self.state = (model_output >= 0.5).float().cuda()
        self.state_record = torch.zeros_like(self.state)

        # 시뮬레이션
        self.recon_image = self._simulate(self.state, z)

        # PSNR
        mse = tt.relativeLoss(self.recon_image, self.target_image, F.mse_loss).item()
        self.initial_psnr = tt.relativeLoss(self.recon_image, self.target_image, tm.get_PSNR)
        self.previous_psnr = self.initial_psnr

        # 배치 픽셀 중요도 (GPU)
        rw_start_time = time.time()
        positive_psnr_sum = self._calculate_pixel_importance(z)
        data_processing_time = time.time() - rw_start_time
        print(f"\nTime taken for psnr_change_list: {data_processing_time:.2f} seconds")

        self.T_PSNR_DIFF = self.T_PSNR_DIFF_o * positive_psnr_sum
        print(f"\033[94m[Dynamic Threshold] T_PSNR_DIFF set to: {self.T_PSNR_DIFF:.6f}\033[0m")

        obs = self._get_obs()

        print(
            f"\033[92mInitial PSNR: {self.initial_psnr:.6f}\033[0m"
            f"\nInitial MSE: {mse:.6f}\033[0m"
        )

        self.next_print_thresholds = [self.initial_psnr + i * 0.01 for i in range(1, 21)]
        self.total_start_time = time.time()

        return obs, {}

    def step(self, action, z=2e-3):
        self.steps += 1

        channel = action // (IPS * IPS)
        pixel_index = action % (IPS * IPS)
        row = pixel_index // IPS
        col = pixel_index % IPS

        # GPU에서 in-place 플립
        self.state[0, channel, row, col] = 1.0 - self.state[0, channel, row, col]
        self.state_record[0, channel, row, col] += 1.0
        self.flip_count += 1

        # GPU 시뮬레이션
        self.recon_image = self._simulate(self.state, z)
        psnr_after = tt.relativeLoss(self.recon_image, self.target_image, tm.get_PSNR)

        # PSNR 변화량 (GPU에서 보상 lookup)
        psnr_change = psnr_after - self.previous_psnr
        psnr_diff = psnr_after - self.initial_psnr

        # GPU argmin으로 가장 유사한 PSNR 변화량 찾기
        closest_index = torch.argmin(torch.abs(self.psnr_change_tensor - psnr_change))
        reward = self.importance_tensor[closest_index].item()

        # 음수 PSNR → state만 롤백 (원래 코드와 동일: state_record는 유지)
        if psnr_change < 0:
            self.state[0, channel, row, col] = 1.0 - self.state[0, channel, row, col]
            # state_record는 롤백하지 않음 (실패한 플립도 시도 기록으로 남김)
            self.flip_count -= 1

            obs = self._get_obs()
            return obs, reward, False, False, {}

        self.max_psnr_diff = max(self.max_psnr_diff, psnr_diff)
        success_ratio = self.flip_count / self.steps if self.steps > 0 else 0

        # 1000 스텝마다 진행 상황 출력
        if self.steps % 1000 == 0:
            data_processing_time = time.time() - self.total_start_time
            print(
                f"\033[93m[Progress] Step: {self.steps} | PSNR: {psnr_after:.6f} | "
                f"Diff: {psnr_diff:.6f} | Success: {success_ratio:.4f} | "
                f"Flips: {self.flip_count} | Time: {data_processing_time:.1f}s\033[0m"
            )

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

        terminated = self.steps >= self.max_steps or self.psnr_sustained_steps >= self.T_steps
        truncated = self.steps >= self.max_steps

        obs = self._get_obs()
        return obs, reward, terminated, truncated, {}
