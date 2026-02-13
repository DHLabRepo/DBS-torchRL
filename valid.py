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

# GPU 최적화 설정
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

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

valid_dataset = Dataset512(target_dir=valid_dir, meta=meta, isTrain=False, padding=padding)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

# ============================================================================
# BinaryNet 사전학습 모델 로드
# ============================================================================
model = BinaryNet(num_hologram=CH, in_planes=1, convReLU=False, convBN=False,
                  poolReLU=False, poolBN=False, deconvReLU=False, deconvBN=False).cuda()
model.load_state_dict(torch.load('result_v/2024-12-19 20:37:52.499731_pre_reinforce_8_0.002/2024-12-19 20:37:52.499731_pre_reinforce_8_0.002'))
model.eval()

# ============================================================================
# PPO 모델 로드
# ============================================================================
device = "cuda" if torch.cuda.is_available() else "cpu"

# 정책 네트워크 생성 (학습 때와 동일한 구조 - 확장된 네트워크)
policy = ActorCriticPolicy(features_dim_per_key=128)

ppo_model_path = "./ppo_pytorch_models/ppo_latest.pt"
ppo = PPO.load(ppo_model_path, policy=policy, device=device)

# ============================================================================
# 환경 생성 (검증용 설정)
# ============================================================================
env = BinaryHologramEnv(
    target_function=model,
    trainloader=valid_loader,
    max_steps=10000,
    T_PSNR=30,
    T_steps=1,
    T_PSNR_DIFF=1,
    num_samples=1000,
    importance_batch_size=64,
)

# ============================================================================
# 검증 루프
# ============================================================================
result_dir = "./results/"
os.makedirs(result_dir, exist_ok=True)

num_episodes = 200

print(f"\n{'='*60}")
print(f"PPO Validation Start")
print(f"Num Episodes: {num_episodes}")
print(f"{'='*60}\n")

for episode in range(num_episodes):
    obs, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        # numpy obs를 텐서 dict로 변환
        obs_tensor = {}
        for key, val in obs.items():
            if isinstance(val, np.ndarray):
                obs_tensor[key] = torch.tensor(val, dtype=torch.float32, device=device).unsqueeze(0)
            else:
                obs_tensor[key] = torch.tensor(np.array(val), dtype=torch.float32, device=device).unsqueeze(0)

        # deterministic 추론
        action = ppo.policy.predict(obs_tensor, deterministic=True)
        action_int = int(action[0]) if isinstance(action, np.ndarray) else int(action)

        obs, reward, terminated, truncated, info = env.step(action_int)
        total_reward += reward
        done = terminated or truncated

    print(f"Episode {episode + 1}: Total Reward: {total_reward}")

    # 결과 저장
    result_file = os.path.join(result_dir, f"episode_{episode + 1}_result.txt")
    with open(result_file, "w") as f:
        f.write(f"Episode {episode + 1}: Total Reward: {total_reward}\n")
        f.write(f"Info: {info}\n")

print(f"\nValidation completed. Results saved to {result_dir}")
