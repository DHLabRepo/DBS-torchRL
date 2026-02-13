# DBS-torchRL

Direct Binary Search (DBS) 강화학습의 **순수 PyTorch 구현**입니다.

기존 `Direct-Binary-Search-Reinforcement-Learning/` 프로젝트에서 `stable-baselines3`와 `gymnasium` 의존성을 완전히 제거하고, 동일한 로직을 PyTorch만으로 재구현하였습니다.

---

## 설치

```bash
git clone git@github.com:DHLabRepo/DBS-torchRL.git

cd DBS-torchRL

mkdir -p result_v

cp -r "/home/songyb111/20260116/Direct-Binary-Search-Reinforcement-Learning/result_v/2024-12-19 20:37:52.499731_pre_reinforce_8_0.002" result_v/

cp -r "/home/songyb111/20260116/Direct-Binary-Search-Reinforcement-Learning/dataset6" .

git clone git@github.com:DHLabRepo/torchOptics.git

cd torchOptics

git reset --hard 8e50d6a
```

---

## 프로젝트 개요

이진 홀로그램(Binary Hologram)의 PSNR을 향상시키기 위해 PPO(Proximal Policy Optimization) 강화학습으로 픽셀 플립 순서를 학습합니다.

1. **BinaryNet** (U-Net)으로 초기 홀로그램 생성
2. **RL 에이전트**가 어떤 픽셀을 플립할지 결정 (524,288개 이산 액션)
3. **torchOptics**로 광학 시뮬레이션 후 PSNR 변화량 기반 보상 계산
4. PPO 알고리즘으로 정책 최적화

---

## 폴더 구조

```
DBS-torchRL/
├── env.py          # 환경 (Full GPU, 배치 시뮬레이션)
├── ppo.py          # 순수 PyTorch PPO (CNN + AMP + Full GPU Buffer)
├── models.py       # BinaryNet (U-Net) + Dataset512 (DIV2K 데이터 로더)
├── train.py        # 학습 스크립트 (cuDNN + TF32 + torch.compile + TensorBoard)
├── valid.py        # 검증 스크립트
├── utils/
│   └── logger.py   # 콘솔+파일 동시 로깅
└── README.md
```

---

## 기존 대비 변경 사항

### 제거된 의존성

| 기존 (stable-baselines3 + gymnasium) | 현재 (순수 PyTorch) |
|--------------------------------------|---------------------|
| `gymnasium.Env` | plain Python class |
| `gymnasium.spaces.Dict`, `Box`, `Discrete` | 사용 안 함 |
| `stable_baselines3.PPO` | `ppo.py > PPO` 클래스 |
| `stable_baselines3.MultiInputPolicy` | `ppo.py > ActorCriticPolicy` |
| `stable_baselines3.BaseCallback` | `train.py` 내 직접 루프 제어 |

### 유지된 의존성

- **PyTorch** - 신경망, 학습
- **torchOptics** - 광학 시뮬레이션 (simulate, PSNR 계산)
- **numpy** - 다항식 보상 함수 계산 (polyfit)
- **torchvision** - 이미지 전처리 (crop, resize)

---

## 아키텍처

### Feature Extractor: NatureCNN

각 관측 키별로 **NatureCNN** (3-layer Conv)을 사용합니다.

> **참고**: 원래 SB3 코드의 관측 shape이 4D `(1, 8, 256, 256)`이라 SB3 내부에서는 `is_image_space=False`로 판단하여 **Flatten + MLP**를 사용합니다. 현재 코드는 CNN 기반이므로 원래 SB3 동작과 다릅니다.

```
관측 키 (5개):
  state_record  (1, 8, 256, 256) → NatureCNN(8ch) → 64 features
  state         (1, 8, 256, 256) → NatureCNN(8ch) → 64 features
  pre_model     (1, 8, 256, 256) → NatureCNN(8ch) → 64 features
  recon_image   (1, 1, 256, 256) → NatureCNN(1ch) → 64 features
  target_image  (1, 1, 256, 256) → NatureCNN(1ch) → 64 features
                                                     ──────────
                                    concat →          320 features

Actor:  320 → 256 → 256 → 524,288 (Categorical)
Critic: 320 → 256 → 256 → 1       (Scalar Value)
```

### 액션 공간

- **Discrete(524,288)**: 8채널 x 256 x 256 = 524,288 픽셀 중 하나를 선택하여 플립
- Categorical 분포에서 샘플링

---

## GPU 최적화

### 1. 환경 (`env.py`) - Full GPU

| 항목 | 기존 | 현재 |
|------|------|------|
| 관측값 반환 | numpy (CPU) | **GPU 텐서** |
| `state` 저장 | numpy (CPU) | **GPU 텐서** (in-place 플립) |
| `state_record` 저장 | numpy (CPU) | **GPU 텐서** |
| 보상 lookup | numpy `argmin` | **`torch.argmin` (GPU)** |
| `_calculate_pixel_importance` | 10,000회 순차 | **배치 병렬** (256개씩) |
| 시뮬레이션 | 매 스텝 텐서 재생성 | **GPU 텐서 직접 사용** |

**전체 데이터 흐름이 GPU에서 처리됩니다. CPU-GPU 전송 없음.**

- `_get_obs()`: 변경되는 state/state_record는 `.clone()`, 불변인 pre_model/target_image는 참조
- `psnr_change_tensor`, `importance_tensor`: GPU 텐서로 보상 계산
- `state_record`는 음수 PSNR 시 롤백하지 않음 (원래 코드와 동일: 실패 플립도 기록)

### 2. PPO (`ppo.py`) - AMP + Full GPU Buffer

| 항목 | 설명 |
|------|------|
| 연산 정밀도 | Mixed Precision (AMP, float16 forward + float32 loss) |
| Feature Extractor | NatureCNN × 5 (CNN 기반) |
| `features_dim_per_key` | 64 |
| Actor/Critic hidden | [256, 256] |
| RolloutBuffer | GPU 텐서 clone 저장 |
| GradScaler | `torch.cuda.amp.GradScaler` |

### 3. 학습 (`train.py`)

| 항목 | 기존 | 현재 |
|------|------|------|
| `cudnn.enabled` | **False** | **True** |
| `cudnn.benchmark` | 미설정 | **True** |
| TF32 | 미설정 | **활성화** (Ampere GPU) |
| `torch.compile` | 없음 | **reduce-overhead** 모드 |
| 학습 모니터링 | 없음 | **TensorBoard** |

---

## TensorBoard

학습 진행도를 실시간으로 확인할 수 있습니다.

### 실행 방법

```bash
tensorboard --logdir=./ppo_tensorboard
```

브라우저에서 `http://localhost:6006` 접속.

### 로깅 메트릭

**loss/** (PPO 업데이트마다, 512 스텝당 1회)
| 메트릭 | 설명 |
|--------|------|
| `policy_loss` | PPO 클립 정책 손실 |
| `value_loss` | Critic MSE 손실 |
| `entropy_loss` | 엔트로피 손실 (탐색 장려) |
| `total_loss` | 가중 합산 손실 |

**episode/** (에피소드 종료마다)
| 메트릭 | 설명 |
|--------|------|
| `reward` | 에피소드 총 보상 |
| `length` | 에피소드 스텝 수 |
| `psnr_diff` | 최대 PSNR 향상량 |
| `initial_psnr` | 초기 PSNR |
| `final_psnr` | 최종 PSNR |
| `flip_count` | 성공 플립 수 |
| `success_ratio` | 플립 성공률 |

**system/** (50 에피소드마다)
| 메트릭 | 설명 |
|--------|------|
| `gpu_memory_gb` | GPU 메모리 사용량 |

---

## PPO 하이퍼파라미터

| 파라미터 | 값 |
|----------|-----|
| `n_steps` | 512 |
| `batch_size` | 128 |
| `n_epochs` | 10 |
| `gamma` | 0.99 |
| `gae_lambda` | 0.9 |
| `learning_rate` | 1e-4 |
| `clip_range` | 0.2 |
| `vf_coef` | 0.5 |
| `ent_coef` | 0.01 |
| `max_grad_norm` | 0.5 |
| `use_amp` | True |

---

## 관측 공간

Dict 형태의 관측값 (5개 키, **모두 GPU 텐서**):

| 키 | Shape | 설명 |
|----|-------|------|
| `state_record` | `(1, 8, 256, 256)` | 각 픽셀의 플립 시도 횟수 (실패 포함) |
| `state` | `(1, 8, 256, 256)` | 현재 이진 홀로그램 상태 |
| `pre_model` | `(1, 8, 256, 256)` | BinaryNet 초기 출력 |
| `recon_image` | `(1, 1, 256, 256)` | 현재 재구성 이미지 |
| `target_image` | `(1, 1, 256, 256)` | 목표 이미지 |

---

## 사용법

### 학습

```bash
cd DBS-torchRL
python train.py
```

- BinaryNet 사전학습 모델 경로를 `train.py`에서 수정 필요
- 데이터셋 경로(`target_dir`, `valid_dir`)를 환경에 맞게 수정 필요
- 학습된 PPO 모델은 `./ppo_pytorch_models/`에 저장됨
- TensorBoard 로그는 `./ppo_tensorboard/`에 저장됨

### 텐서보드 모니터링

```bash
tensorboard --logdir=./ppo_tensorboard
```

### GPU 배치 크기 조정

`train.py`에서 `importance_batch_size`를 GPU VRAM에 맞게 조정:

```python
env = BinaryHologramEnv(
    ...
    importance_batch_size=256,  # RTX 4090 24GB 기준
)
```

### 검증

```bash
cd DBS-torchRL
python valid.py
```

### 모델 저장/로드

```python
# 저장
ppo.save("path/to/model.pt")

# 로드
policy = ActorCriticPolicy(features_dim_per_key=64)
ppo = PPO.load("path/to/model.pt", policy=policy, device="cuda")
```

---

## 광학 시뮬레이션 파라미터

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| 파장 (wavelength) | 515 nm | 녹색 광원 |
| 픽셀 크기 (pixel pitch) | 7.56 um | SLM 픽셀 크기 |
| 전파 거리 (propagation distance) | 2 mm | 홀로그램-카메라 거리 |
| 이미지 크기 | 256 x 256 | 입출력 해상도 |
| 채널 수 | 8 | 이진 홀로그램 채널 수 |
