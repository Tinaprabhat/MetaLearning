# meta_learners/ppo_agent.py
"""
Proximal Policy Optimisation (PPO) agent for continuous lr-scale control.

Architecture
------------
  Actor  : MLP  obs → mean of Gaussian action distribution
  Critic : MLP  obs → scalar value estimate V(s)

Both share the same hidden layers (separate heads on top).

PPO clip objective (Schulman et al., 2017):
    L_CLIP = E[ min( r_t A_t,  clip(r_t, 1−ε, 1+ε) A_t ) ]
where r_t = π_θ(a|s) / π_θ_old(a|s)

The agent uses Generalised Advantage Estimation (GAE) for lower-variance
advantage estimates.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Normal


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

class ActorCritic(nn.Module):
    """
    Shared-trunk actor-critic network.

    Input  : obs_dim-dimensional state vector
    Actor  : outputs (mean, log_std) for a Normal distribution over actions
    Critic : outputs scalar V(s)
    """

    def __init__(self, obs_dim: int = 3, hidden_dim: int = 64):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor_mean = nn.Linear(hidden_dim, 1)
        self.actor_log_std = nn.Parameter(torch.zeros(1))  # learnable std
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor):
        h = self.trunk(obs)
        mean = torch.sigmoid(self.actor_mean(h))          # keep in (0, 1)
        std = self.actor_log_std.exp().expand_as(mean)
        value = self.critic(h).squeeze(-1)
        return mean, std, value

    def act(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample an action and return (action, log_prob)."""
        mean, std, _ = self.forward(obs)
        dist = Normal(mean, std)
        action = dist.sample()
        action = action.clamp(1e-3, 1.0)
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob

    def evaluate(
        self, obs: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return log_prob, entropy, value for a batch."""
        mean, std, value = self.forward(obs)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy, value


# ---------------------------------------------------------------------------
# Rollout Buffer
# ---------------------------------------------------------------------------

class RolloutBuffer:
    """Stores one episode's worth of transitions for PPO updates."""

    def __init__(self):
        self.obs: list[np.ndarray] = []
        self.actions: list[float] = []
        self.log_probs: list[float] = []
        self.rewards: list[float] = []
        self.values: list[float] = []
        self.dones: list[bool] = []

    def add(self, obs, action, log_prob, reward, value, done):
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.rewards)


# ---------------------------------------------------------------------------
# PPO Agent
# ---------------------------------------------------------------------------

class PPOAgent:
    """
    PPO agent that learns a continuous lr-scaling policy.

    Args:
        obs_dim      : Dimensionality of the state observation (default 3).
        hidden_dim   : Width of hidden layers in the actor-critic network.
        lr           : Learning rate for the Adam optimiser.
        clip_eps     : PPO clipping epsilon ε.
        gamma        : Discount factor for returns.
        gae_lambda   : GAE λ parameter.
        ppo_epochs   : Number of optimisation epochs per batch of data.
        entropy_coef : Coefficient for the entropy bonus term.
        vf_coef      : Coefficient for the value-function loss.
    """

    def __init__(
        self,
        obs_dim: int = 3,
        hidden_dim: int = 64,
        lr: float = 3e-4,
        clip_eps: float = 0.2,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        ppo_epochs: int = 4,
        entropy_coef: float = 0.01,
        vf_coef: float = 0.5,
    ):
        self.clip_eps = clip_eps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ppo_epochs = ppo_epochs
        self.entropy_coef = entropy_coef
        self.vf_coef = vf_coef

        self.ac = ActorCritic(obs_dim, hidden_dim)
        self.optimizer = optim.Adam(self.ac.parameters(), lr=lr)
        self.buffer = RolloutBuffer()

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, obs: np.ndarray) -> tuple[float, float, float]:
        """
        Given a numpy obs, return (action_scalar, log_prob, value).
        Called at each environment step.
        """
        obs_t = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            action, log_prob = self.ac.act(obs_t)
            _, _, value = self.ac.forward(obs_t)

        return action.item(), log_prob.item(), value.item()

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def compute_gae(self, last_value: float = 0.0) -> torch.Tensor:
        """Compute GAE advantages and discounted returns."""
        rewards = self.buffer.rewards
        values = self.buffer.values + [last_value]
        dones = self.buffer.dones

        advantages = []
        gae = 0.0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        advantages = torch.FloatTensor(advantages)
        returns = advantages + torch.FloatTensor(self.buffer.values)
        return advantages, returns

    def update(self) -> dict:
        """
        Run PPO_epochs of gradient updates on the collected rollout.
        Returns a dict of loss scalars for logging.
        """
        advantages, returns = self.compute_gae()

        # Normalise advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        obs_t = torch.FloatTensor(np.array(self.buffer.obs))
        actions_t = torch.FloatTensor(self.buffer.actions).unsqueeze(-1)
        old_log_probs_t = torch.FloatTensor(self.buffer.log_probs)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0

        for _ in range(self.ppo_epochs):
            log_probs, entropy, values = self.ac.evaluate(obs_t, actions_t)

            ratio = (log_probs - old_log_probs_t).exp()
            surr1 = ratio * advantages
            surr2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * advantages

            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = ((values - returns) ** 2).mean()
            entropy_loss = -entropy.mean()

            loss = (
                policy_loss
                + self.vf_coef * value_loss
                + self.entropy_coef * entropy_loss
            )

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.ac.parameters(), 0.5)
            self.optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.mean().item()

        self.buffer.clear()

        return {
            "policy_loss": total_policy_loss / self.ppo_epochs,
            "value_loss": total_value_loss / self.ppo_epochs,
            "entropy": total_entropy / self.ppo_epochs,
        }

    def save(self, path: str):
        torch.save(self.ac.state_dict(), path)

    def load(self, path: str):
        self.ac.load_state_dict(torch.load(path, weights_only=True))