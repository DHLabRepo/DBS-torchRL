"""
순수 PyTorch PPO 구현 (Full GPU).
SB3의 MultiInputPolicy (MLP 기반 CombinedExtractor) 재현.
모든 데이터가 GPU 텐서로 처리됩니다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.cuda.amp import autocast, GradScaler


IPS = 256
CH = 8


# ============================================================================
# Feature Extractor (SB3의 CombinedExtractor 재현 - MLP 기반)
# ============================================================================

# SB3의 CombinedExtractor는 4D shape에 대해 nn.Flatten()을 사용합니다.
# 각 관측 키를 flatten 후 concat하여 하나의 벡터로 만듭니다.
#
# 각 키의 flattened 크기:
#   state_record: 1*8*256*256 = 524,288
#   state:        1*8*256*256 = 524,288
#   pre_model:    1*8*256*256 = 524,288
#   recon_image:  1*1*256*256 = 65,536
#   target_image: 1*1*256*256 = 65,536
#   총 concat 크기: 1,703,936

OBS_KEYS = ["state_record", "state", "pre_model", "recon_image", "target_image"]

# 각 키의 채널 수
OBS_CHANNELS = {
    "state_record": CH,   # 8
    "state":        CH,   # 8
    "pre_model":    CH,   # 8
    "recon_image":  1,
    "target_image": 1,
}


class CombinedExtractor(nn.Module):
    """
    SB3의 CombinedExtractor와 동일.
    각 Dict 관측 키를 Flatten 후 concat합니다.
    4D shape (1, C, H, W)은 is_image_space=False이므로 MLP(Flatten) 사용.
    """
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()

        # 총 concat 크기 계산
        self.features_dim = 0
        for key in OBS_KEYS:
            c = OBS_CHANNELS[key]
            self.features_dim += 1 * c * IPS * IPS  # shape = (1, C, IPS, IPS)

    def forward(self, obs_dict):
        features = []
        for key in OBS_KEYS:
            x = obs_dict[key].float()
            # (B, 1, C, H, W) -> (B, 1*C*H*W)
            features.append(self.flatten(x))
        return torch.cat(features, dim=-1)


# ============================================================================
# Actor-Critic Policy (MLP 기반)
# ============================================================================

class ActorCriticPolicy(nn.Module):
    """
    SB3 MultiInputPolicy 재현 (MLP 기반).
    CombinedExtractor(Flatten+concat) → 공유 없는 별도 Actor/Critic MLP.

    SB3 기본 net_arch: dict(pi=[64, 64], vf=[64, 64])
    → 각각 별도의 2-layer MLP
    """
    def __init__(self, net_arch_pi=None, net_arch_vf=None):
        super().__init__()

        self.feature_extractor = CombinedExtractor()
        features_dim = self.feature_extractor.features_dim
        num_actions = CH * IPS * IPS  # 524288

        # Actor 네트워크 (SB3 기본: [64, 64])
        if net_arch_pi is None:
            net_arch_pi = [64, 64]

        pi_layers = []
        prev_dim = features_dim
        for hidden_dim in net_arch_pi:
            pi_layers.append(nn.Linear(prev_dim, hidden_dim))
            pi_layers.append(nn.ReLU())
            prev_dim = hidden_dim
        pi_layers.append(nn.Linear(prev_dim, num_actions))
        self.actor = nn.Sequential(*pi_layers)

        # Critic 네트워크 (SB3 기본: [64, 64])
        if net_arch_vf is None:
            net_arch_vf = [64, 64]

        vf_layers = []
        prev_dim = features_dim
        for hidden_dim in net_arch_vf:
            vf_layers.append(nn.Linear(prev_dim, hidden_dim))
            vf_layers.append(nn.ReLU())
            prev_dim = hidden_dim
        vf_layers.append(nn.Linear(prev_dim, 1))
        self.critic = nn.Sequential(*vf_layers)

    def forward(self, obs_dict):
        features = self.feature_extractor(obs_dict)
        return self.actor(features), self.critic(features)

    def get_action_and_value(self, obs_dict, action=None, deterministic=False):
        logits, value = self.forward(obs_dict)
        dist = Categorical(logits=logits)
        if action is None:
            action = logits.argmax(dim=-1) if deterministic else dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value

    def get_value(self, obs_dict):
        features = self.feature_extractor(obs_dict)
        return self.critic(features)

    def predict(self, obs_dict, deterministic=True):
        with torch.no_grad():
            logits, _ = self.forward(obs_dict)
            action = logits.argmax(dim=-1) if deterministic else Categorical(logits=logits).sample()
        return action


# ============================================================================
# Rollout Buffer (관측값은 CPU 저장, 학습 시 GPU 전송)
# ============================================================================

class RolloutBuffer:
    """
    PPO 롤아웃 버퍼.
    관측값(obs)은 CPU에 저장하여 GPU 메모리 절약.
    (512 스텝 × 5 obs × ~6.5MB = ~3.3GB GPU 메모리 절약)
    학습(get_samples) 시에만 미니배치를 GPU로 전송.
    스칼라 값(action, reward, done 등)은 GPU 유지.
    """
    def __init__(self, n_steps, device="cuda", gamma=0.99, gae_lambda=0.95):
        self.n_steps = n_steps
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.obs_list = []       # list of dict{str: CPU tensor}
        self.actions = torch.zeros(n_steps, dtype=torch.long, device=device)
        self.rewards = torch.zeros(n_steps, dtype=torch.float32, device=device)
        self.dones = torch.zeros(n_steps, dtype=torch.float32, device=device)
        self.log_probs = torch.zeros(n_steps, dtype=torch.float32, device=device)
        self.values = torch.zeros(n_steps, dtype=torch.float32, device=device)

        self.advantages = None
        self.returns = None
        self.pos = 0

    def add(self, obs, action, reward, done, log_prob, value):
        # GPU 텐서 → CPU로 이동하여 저장 (GPU 메모리 절약)
        cpu_obs = {k: v.detach().cpu() for k, v in obs.items()}
        self.obs_list.append(cpu_obs)
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = float(done)
        self.log_probs[self.pos] = log_prob
        self.values[self.pos] = value
        self.pos += 1

    def is_full(self):
        return self.pos >= self.n_steps

    def compute_returns_and_advantages(self, last_value, last_done):
        advantages = torch.zeros(self.pos, device=self.device)
        last_gae_lam = 0.0

        for step in reversed(range(self.pos)):
            if step == self.pos - 1:
                next_non_terminal = 1.0 - float(last_done)
                next_values = last_value
            else:
                next_non_terminal = 1.0 - self.dones[step + 1].item()
                next_values = self.values[step + 1].item()

            delta = self.rewards[step].item() + self.gamma * next_values * next_non_terminal - self.values[step].item()
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            advantages[step] = last_gae_lam

        self.advantages = advantages
        self.returns = advantages + self.values[:self.pos]

    def get_samples(self, batch_size):
        """미니배치 생성: CPU obs를 배치 단위로 GPU 전송"""
        indices = torch.randperm(self.pos, device='cpu')  # CPU에서 인덱싱

        for start in range(0, self.pos, batch_size):
            end = min(start + batch_size, self.pos)
            batch_idx = indices[start:end]

            # 미니배치만 GPU로 전송
            batch_obs = {}
            for key in OBS_KEYS:
                batch_obs[key] = torch.stack(
                    [self.obs_list[i][key] for i in batch_idx]
                ).to(self.device, non_blocking=True)

            yield (
                batch_obs,
                self.actions[batch_idx.to(self.device)],
                self.log_probs[batch_idx.to(self.device)],
                self.advantages[batch_idx.to(self.device)],
                self.returns[batch_idx.to(self.device)],
            )

    def reset(self):
        self.obs_list.clear()
        self.actions.zero_()
        self.rewards.zero_()
        self.dones.zero_()
        self.log_probs.zero_()
        self.values.zero_()
        self.advantages = None
        self.returns = None
        self.pos = 0


# ============================================================================
# PPO Algorithm (Full GPU + AMP)
# ============================================================================

class PPO:
    def __init__(
        self,
        policy: ActorCriticPolicy,
        device="cuda",
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
        use_amp=True,
    ):
        self.policy = policy.to(device)
        self.device = device

        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate, eps=1e-5)

        self.use_amp = use_amp and device == "cuda"
        self.scaler = GradScaler(enabled=self.use_amp)

        self.buffer = RolloutBuffer(n_steps, device, gamma, gae_lambda)
        self.n_updates = 0

    def collect_step(self, obs, env):
        obs_batched = {k: v.unsqueeze(0) if v.dim() == 4 else v for k, v in obs.items()}

        with torch.no_grad():
            if self.use_amp:
                with autocast():
                    action, log_prob, entropy, value = self.policy.get_action_and_value(obs_batched)
            else:
                action, log_prob, entropy, value = self.policy.get_action_and_value(obs_batched)

        action_int = action.item()
        log_prob_float = log_prob.float().item()
        value_float = value.float().item()

        next_obs, reward, terminated, truncated, info = env.step(action_int)
        done = terminated or truncated

        self.buffer.add(obs, action_int, float(reward), done, log_prob_float, value_float)

        return next_obs, done, reward, info

    def update(self):
        last_obs = self.buffer.obs_list[-1]  # CPU tensor dict
        last_done = self.buffer.dones[self.buffer.pos - 1].item() > 0.5

        if last_done:
            last_value = 0.0
        else:
            # CPU → GPU 전송 후 배치 차원 추가
            obs_batched = {k: v.to(self.device).unsqueeze(0) if v.dim() == 4 else v.to(self.device) for k, v in last_obs.items()}
            with torch.no_grad():
                if self.use_amp:
                    with autocast():
                        last_value = self.policy.get_value(obs_batched).float().item()
                else:
                    last_value = self.policy.get_value(obs_batched).item()

        self.buffer.compute_returns_and_advantages(last_value, last_done)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy_loss = 0.0
        total_loss_val = 0.0
        n_updates = 0

        for epoch in range(self.n_epochs):
            for batch_obs, batch_actions, batch_old_log_probs, batch_advantages, batch_returns in \
                    self.buffer.get_samples(self.batch_size):

                with autocast(enabled=self.use_amp):
                    _, new_log_probs, entropy, new_values = self.policy.get_action_and_value(
                        batch_obs, action=batch_actions
                    )
                    new_values = new_values.squeeze(-1).float()
                    new_log_probs = new_log_probs.float()
                    entropy = entropy.float()

                    adv = batch_advantages
                    if len(adv) > 1:
                        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                    ratio = torch.exp(new_log_probs - batch_old_log_probs)
                    surr1 = ratio * adv
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * adv
                    policy_loss = -torch.min(surr1, surr2).mean()

                    value_loss = F.mse_loss(new_values, batch_returns)
                    entropy_loss = -entropy.mean()
                    loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss

                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy_loss.item()
                total_loss_val += loss.item()
                n_updates += 1

        self.n_updates += 1
        self.buffer.reset()

        if n_updates > 0:
            return {
                "policy_loss": total_policy_loss / n_updates,
                "value_loss": total_value_loss / n_updates,
                "entropy_loss": total_entropy_loss / n_updates,
                "total_loss": total_loss_val / n_updates,
            }
        return {}

    def save(self, path):
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'n_updates': self.n_updates,
        }, path)
        print(f"PPO model saved to {path}")

    @classmethod
    def load(cls, path, policy, device="cuda", **kwargs):
        ppo = cls(policy=policy, device=device, **kwargs)
        checkpoint = torch.load(path, map_location=device)
        ppo.policy.load_state_dict(checkpoint['policy_state_dict'])
        ppo.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scaler_state_dict' in checkpoint:
            ppo.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        ppo.n_updates = checkpoint.get('n_updates', 0)
        print(f"PPO model loaded from {path}")
        return ppo
