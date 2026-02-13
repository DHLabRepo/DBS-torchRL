# DBS-torchRL

Direct Binary Search (DBS) 강화학습의 **순수 PyTorch 구현**입니다.

기존 `Direct-Binary-Search-Reinforcement-Learning/` 프로젝트에서 `stable-baselines3`와 `gymnasium` 의존성을 완전히 제거하고, 동일한 로직을 PyTorch만으로 재구현하였습니다.

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
├── env.py          # 환경 (gymnasium 제거, plain Python class)
├── ppo.py          # 순수 PyTorch PPO (Feature Extractor + Actor-Critic + Buffer + PPO)
├── models.py       # BinaryNet (U-Net) + Dataset512 (DIV2K 데이터 로더)
├── train.py        # 학습 스크립트
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
- **numpy** - 수치 연산
- **torchvision** - 이미지 전처리 (crop, resize)

### 파일별 변경 내용

**`env.py`** - 환경
- `gym.Env` 상속 제거 -> 일반 Python 클래스
- `spaces.Dict/Box/Discrete` 제거
- `reset()` / `step()` 인터페이스 동일 유지
- 나머지 로직 100% 동일 (픽셀 중요도 계산, 다항식 보상, PSNR 롤백 등)

**`ppo.py`** - PPO 알고리즘 (신규)
- `NatureCNN`: SB3의 NatureCNN과 동일한 CNN Feature Extractor
- `MultiInputFeatureExtractor`: 5개 관측 키 각각을 별도 CNN으로 처리 후 concat (SB3의 CombinedExtractor 대체)
- `ActorCriticPolicy`: Actor (Categorical, 524,288 이산 액션) + Critic (스칼라 가치)
- `RolloutBuffer`: n_steps 분량 트랜지션 저장 + GAE 계산
- `PPO`: PPO Clip Loss + Value Loss + Entropy Loss 기반 업데이트

**`models.py`** - 모델 (분리)
- `BinaryNet`: 기존 train.py에 있던 U-Net 모델을 별도 파일로 분리
- `Dataset512`: DIV2K 데이터셋 로더를 별도 파일로 분리

**`train.py`** - 학습
- SB3의 `ppo_model.learn(total_timesteps=...)` 한 줄 호출 대신 명시적 루프:
  - `env.reset()` -> `ppo.collect_step()` -> 버퍼 가득 차면 `ppo.update()`
- 콜백 시스템 대신 직접 에피소드 보상 추적
- 모델 저장/로드 (`torch.save` / `torch.load`)

**`valid.py`** - 검증
- SB3의 `ppo_model.predict(obs)` 대신 `policy.predict(obs_tensor)` 직접 호출

---

## PPO 하이퍼파라미터

기존 SB3 설정과 동일하게 유지:

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

---

## 관측 공간

Dict 형태의 관측값 (5개 키):

| 키 | Shape | 설명 |
|----|-------|------|
| `state_record` | `(1, 8, 256, 256)` | 각 픽셀의 플립 횟수 기록 |
| `state` | `(1, 8, 256, 256)` | 현재 이진 홀로그램 상태 |
| `pre_model` | `(1, 8, 256, 256)` | BinaryNet 초기 출력 |
| `recon_image` | `(1, 1, 256, 256)` | 현재 재구성 이미지 |
| `target_image` | `(1, 1, 256, 256)` | 목표 이미지 |

## 액션 공간

- **Discrete(524,288)**: 8채널 x 256 x 256 = 524,288 픽셀 중 하나를 선택하여 플립
- Categorical 분포에서 샘플링

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

### 검증

```bash
cd DBS-torchRL
python valid.py
```

- 학습된 모델(`ppo_latest.pt`)을 로드하여 200개 에피소드 평가
- 결과는 `./results/`에 저장됨

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
