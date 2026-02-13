import sys
import logging
from datetime import datetime
import os
from utils.logger import setup_logger

# 로거 설정
log_file = setup_logger()

print("이 메시지는 콘솔과 파일에 동시에 기록됩니다.")
logging.info("이 메시지도 로그에 기록됩니다.")

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
from torch.utils.data import DataLoader

import torchOptics.optics as tt
import torchOptics.metrics as tm

from env import BinaryHologramEnv
from models import BinaryNet, Dataset512, IPS, CH
from ppo import ActorCriticPolicy, PPO

warnings.filterwarnings('ignore')

current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# ============================================================================
# GPU 최적화 설정
# ============================================================================
# cuDNN 활성화 (기존 코드에서는 False로 비활성화되어 있었음)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True  # 입력 크기가 일정할 때 최적 알고리즘 자동 선택

# TF32 활성화 (Ampere GPU 이상, RTX 4090 포함)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

print(f"cuDNN enabled: {torch.backends.cudnn.enabled}")
print(f"cuDNN benchmark: {torch.backends.cudnn.benchmark}")
print(f"TF32 matmul: {torch.backends.cuda.matmul.allow_tf32}")
print(f"TF32 cuDNN: {torch.backends.cudnn.allow_tf32}")

# ============================================================================
# BinaryNet 모델 초기화 및 로드
# ============================================================================
model = BinaryNet(num_hologram=CH, in_planes=1, convReLU=False,
                  convBN=False, poolReLU=False, poolBN=False,
                  deconvReLU=False, deconvBN=False).cuda()
test = torch.randn(1, 1, IPS, IPS).cuda()
out = model(test)
print(out.shape)

# ============================================================================
# 데이터셋 준비
# ============================================================================
batch_size = 1
target_dir = '/nfs/dataset/DIV2K/DIV2K_train_HR/DIV2K_train_HR/'
valid_dir = '/nfs/dataset/DIV2K/DIV2K_valid_HR/DIV2K_valid_HR/'
meta = {'wl': (515e-9), 'dx': (7.56e-6, 7.56e-6)}
padding = 0

train_dataset = Dataset512(target_dir=target_dir, meta=meta, isTrain=True, padding=padding)
valid_dataset = Dataset512(target_dir=valid_dir, meta=meta, isTrain=False, padding=padding)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

# ============================================================================
# BinaryNet 사전학습 모델 로드
# ============================================================================
model = BinaryNet(num_hologram=CH, in_planes=1, convReLU=False, convBN=False,
                  poolReLU=False, poolBN=False, deconvReLU=False, deconvBN=False).cuda()
model.load_state_dict(torch.load('result_v/2024-12-19 20:37:52.499731_pre_reinforce_8_0.002/2024-12-19 20:37:52.499731_pre_reinforce_8_0.002'))
model.eval()

# ============================================================================
# 환경 생성 (GPU 배치 시뮬레이션 활성화)
# ============================================================================
# importance_batch_size: 픽셀 중요도 계산 시 한 번에 배치 처리할 픽셀 수
# GPU VRAM 여유에 따라 조정 (RTX 4090 24GB 기준 64~128 권장)
env = BinaryHologramEnv(
    target_function=model,
    trainloader=train_loader,
    max_steps=10000,
    T_PSNR=30,
    T_steps=1,
    T_PSNR_DIFF=1/4,
    num_samples=10000,
    importance_batch_size=64,  # GPU 배치 크기 (VRAM에 맞게 조정)
)

# ============================================================================
# PPO 설정 (확장된 네트워크 + AMP)
# ============================================================================
device = "cuda" if torch.cuda.is_available() else "cpu"

# 정책 네트워크 생성 (SB3 기본과 동일: MLP [64,64])
policy = ActorCriticPolicy()

# 파라미터 수 출력
total_params = sum(p.numel() for p in policy.parameters())
trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
print(f"\nPolicy Network Parameters: {total_params:,} total, {trainable_params:,} trainable")

# PPO 하이퍼파라미터
ppo_kwargs = dict(
    n_steps=512,
    batch_size=128,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.9,
    learning_rate=1e-4,
    clip_range=0.2,
    vf_coef=0.5,
    ent_coef=0.01,
    max_grad_norm=0.5,
    use_amp=True,  # Mixed Precision 활성화
)

# 저장 디렉토리
save_dir = "./ppo_pytorch_models/"
os.makedirs(save_dir, exist_ok=True)

# 모델 로드 또는 새로 생성
ppo_model_path = os.path.join(save_dir, "ppo_latest.pt")
resume_training = True

if resume_training and os.path.exists(ppo_model_path):
    print(f"Loading trained PPO model from {ppo_model_path}")
    ppo = PPO.load(ppo_model_path, policy=policy, device=device, **ppo_kwargs)
else:
    if resume_training:
        print(f"Warning: PPO model not found at {ppo_model_path}. Starting training from scratch.")
    print("Starting training from scratch.")
    ppo = PPO(policy=policy, device=device, **ppo_kwargs)

# torch.compile 적용 (PyTorch 2.0+, 선택적)
try:
    ppo.policy = torch.compile(ppo.policy, mode="reduce-overhead")
    print("torch.compile applied to policy network (reduce-overhead mode)")
except Exception as e:
    print(f"torch.compile not available or failed: {e}")
    print("Continuing without torch.compile")

# ============================================================================
# GPU 메모리 상태 출력
# ============================================================================
if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"\nGPU Memory: {allocated:.2f}GB allocated / {reserved:.2f}GB reserved / {total:.2f}GB total")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ============================================================================
# 학습 루프
# ============================================================================
max_episodes = 8000
total_timesteps = 1000000000

episode_count = 0
global_step = 0
episode_rewards = []

print(f"\n{'='*60}")
print(f"PPO Training Start (GPU Optimized)")
print(f"Max Episodes: {max_episodes}")
print(f"n_steps: {ppo_kwargs['n_steps']}, batch_size: {ppo_kwargs['batch_size']}")
print(f"Mixed Precision (AMP): {ppo_kwargs['use_amp']}")
print(f"{'='*60}\n")

try:
    while episode_count < max_episodes and global_step < total_timesteps:
        # 에피소드 시작
        obs, info = env.reset()
        done = False
        episode_reward = 0.0

        while not done:
            # 한 스텝 수집
            next_obs, done, reward, step_info = ppo.collect_step(obs, env)
            episode_reward += reward
            global_step += 1

            # 버퍼가 가득 차면 업데이트
            if ppo.buffer.is_full():
                metrics = ppo.update()
                print(
                    f"\033[94m[PPO Update #{ppo.n_updates}] "
                    f"Policy Loss: {metrics['policy_loss']:.4f} | "
                    f"Value Loss: {metrics['value_loss']:.4f} | "
                    f"Entropy Loss: {metrics['entropy_loss']:.4f} | "
                    f"Total Loss: {metrics['total_loss']:.4f}\033[0m"
                )

            obs = next_obs

        # 에피소드 종료
        episode_count += 1
        episode_rewards.append(episode_reward)

        print(f"\033[41mEpisode {episode_count}: Total Reward: {episode_reward:.2f}\033[0m")

        # GPU 메모리 상태 주기적 출력 (50 에피소드마다)
        if episode_count % 50 == 0 and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            print(f"\033[93m[GPU] Memory allocated: {allocated:.2f}GB\033[0m")

        # 주기적으로 모델 저장 (10 에피소드마다)
        if episode_count % 10 == 0:
            ppo.save(ppo_model_path)

        if episode_count >= max_episodes:
            print(f"Stopping training at episode {episode_count}")
            break

except KeyboardInterrupt:
    print("\n학습이 사용자에 의해 중단되었습니다.")

# ============================================================================
# 최종 모델 저장
# ============================================================================
current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
print(f"Start {current_date}_model saving at {save_dir}")
ppo_model_save_path = os.path.join(save_dir, f"ppo_{current_date}.pt")
ppo.save(ppo_model_save_path)
print(f"PPO Model saved at {save_dir}")

# 최신 모델 업데이트
print(f"Start latest_model updating at {save_dir}")
ppo_latest_path = os.path.join(save_dir, "ppo_latest.pt")
if os.path.exists(ppo_latest_path):
    os.remove(ppo_latest_path)
shutil.copyfile(ppo_model_save_path, ppo_latest_path)
print(f"Latest model updated at {ppo_latest_path}")
