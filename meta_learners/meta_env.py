# meta_learners/meta_env.py
"""
Reinforcement-Learning environment for adaptive learning-rate control.

The RL agent (PPO) observes training diagnostics and selects a learning-rate
scale at each step.  The environment wraps one epoch of base-learner training.

State  s_t = [ loss_t,  grad_norm_t,  weight_norm_t ]    (shape: 3,)
Action a_t ∈ [0, 1]  →  effective_lr = base_lr × a_t
Reward r_t = prev_loss − curr_loss  (positive when loss decreases)

Episode: one full training run of `max_steps` epochs on a single task.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from torch.func import functional_call


class MetaLearningEnv:
    """
    Gym-style environment (no gym dependency) for lr-controller RL.

    Args:
        model        : BaseLearner instance (weights treated as initial state).
        X, y         : Task data (torch tensors).
        base_lr      : Nominal learning rate — agent scales this.
        max_steps    : Number of training epochs per episode.
    """

    OBS_DIM = 3   # [loss, grad_norm, weight_norm]
    ACT_DIM = 1   # continuous scalar in (0, 1)

    def __init__(
        self,
        model: nn.Module,
        X: torch.Tensor,
        y: torch.Tensor,
        base_lr: float = 0.01,
        max_steps: int = 50,
    ):
        self.model = model
        self.X = X
        self.y = y
        self.base_lr = base_lr
        self.max_steps = max_steps
        self.criterion = nn.CrossEntropyLoss()

        self._step = 0
        self._prev_loss: float | None = None

    # ------------------------------------------------------------------
    # Environment interface
    # ------------------------------------------------------------------

    def reset(self, init_state_dict: dict | None = None) -> np.ndarray:
        """
        Reset the environment for a new episode.

        Args:
            init_state_dict : If provided, load these weights into the model
                              (e.g. MAML meta-parameters) instead of randomly
                              re-initialising.  This is the key fix: without it,
                              the MAML initialisation is discarded every episode.
        """
        if init_state_dict is not None:
            self.model.load_state_dict(init_state_dict)
        else:
            for p in self.model.parameters():
                nn.init.xavier_uniform_(p) if p.dim() >= 2 else nn.init.zeros_(p)

        self._optimizer = optim.SGD(self.model.parameters(), lr=self.base_lr)
        self._step = 0
        self._prev_loss = None
        return self._get_obs()

    def step(self, action: float) -> tuple[np.ndarray, float, bool]:
        """
        Apply the agent's lr-scale action for one training epoch.

        Returns:
            obs     : Next state observation (np.ndarray, shape [3]).
            reward  : r_t = prev_loss − curr_loss.
            done    : True when episode ends.
        """
        # Clip action to a safe range to avoid exploding updates
        lr_scale = float(np.clip(action, 1e-3, 1.0))
        effective_lr = self.base_lr * lr_scale

        # Apply learning rate
        for g in self._optimizer.param_groups:
            g["lr"] = effective_lr

        # Forward + backward
        self._optimizer.zero_grad()
        logits = self.model(self.X)
        loss = self.criterion(logits, self.y)
        loss.backward()
        self._optimizer.step()

        curr_loss = loss.item()

        # Reward: improvement over previous step
        if self._prev_loss is None:
            reward = 0.0
        else:
            reward = self._prev_loss - curr_loss

        self._prev_loss = curr_loss
        self._step += 1
        done = self._step >= self.max_steps

        return self._get_obs(), reward, done

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        """Compute [loss, grad_norm, weight_norm] without running a new step."""
        with torch.no_grad():
            logits = self.model(self.X)
            loss = self.criterion(logits, self.y).item()

        grad_norm = self._compute_grad_norm()
        weight_norm = self._compute_weight_norm()
        return np.array([loss, grad_norm, weight_norm], dtype=np.float32)

    def _compute_grad_norm(self) -> float:
        total = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total += p.grad.data.norm(2).item() ** 2
        return total ** 0.5

    def _compute_weight_norm(self) -> float:
        total = 0.0
        for p in self.model.parameters():
            total += p.data.norm(2).item() ** 2
        return total ** 0.5