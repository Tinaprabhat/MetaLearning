# meta_learners/maml_trainer.py
"""
Model-Agnostic Meta-Learning (MAML) implementation.

The outer loop maintains a set of meta-parameters θ that are trained such that
a small number of gradient steps on a new task produces a good model.

  θ* = argmin_θ  Σ_T  L_T( θ − α ∇_θ L_T(θ) )
                         └─ inner update ───┘

Reference: Finn et al., 2017 — "Model-Agnostic Meta-Learning for Fast Adaptation
           of Deep Neural Networks" (https://arxiv.org/abs/1703.03400)
"""

import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.func import functional_call, grad


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clone_params(model: nn.Module) -> dict:
    """Return a {name: tensor} copy of model parameters (no grad history)."""
    return {k: v.clone() for k, v in model.named_parameters()}


def _inner_step(
    model: nn.Module,
    params: dict,
    X: torch.Tensor,
    y: torch.Tensor,
    criterion: nn.Module,
    inner_lr: float,
) -> dict:
    """
    One gradient step on the support set using the supplied parameter dict.
    Uses torch.func.functional_call so we never mutate model.state_dict().
    """
    # Forward pass with the current (possibly adapted) params
    logits = functional_call(model, params, (X,))
    loss = criterion(logits, y)

    # Compute gradients w.r.t. params (create_graph=True for meta-gradient)
    grads = torch.autograd.grad(loss, params.values(), create_graph=True)

    adapted = {
        k: p - inner_lr * g
        for (k, p), g in zip(params.items(), grads)
    }
    return adapted


# ---------------------------------------------------------------------------
# MAML Trainer
# ---------------------------------------------------------------------------

class MAMLTrainer:
    """
    Trains a model using MAML.

    Args:
        model        : The base learner (nn.Module). Weights serve as θ.
        inner_lr     : Learning rate for the inner (task-specific) update.
        outer_lr     : Learning rate for the meta (outer loop) optimiser.
        inner_steps  : Number of gradient steps in the inner loop (k).
        meta_batch   : Number of tasks sampled per outer update.
    """

    def __init__(
        self,
        model: nn.Module,
        inner_lr: float = 0.05,
        outer_lr: float = 1e-3,
        inner_steps: int = 5,
        meta_batch: int = 4,
    ):
        self.model = model
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.meta_batch = meta_batch

        self.criterion = nn.CrossEntropyLoss()
        self.outer_optim = optim.Adam(model.parameters(), lr=outer_lr)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def meta_train_step(self, tasks: list) -> float:
        """
        One outer-loop update over a batch of tasks.

        Args:
            tasks : list of (X_support, y_support, X_query, y_query) tuples.
                    Each element is a torch.Tensor already cast to the right dtype.

        Returns:
            Scalar meta-loss (float) for logging.
        """
        self.outer_optim.zero_grad()
        meta_loss = torch.tensor(0.0)

        for X_s, y_s, X_q, y_q in tasks[: self.meta_batch]:
            # ---- Inner loop: adapt θ on support set ----
            adapted_params = _clone_params(self.model)
            for _ in range(self.inner_steps):
                adapted_params = _inner_step(
                    self.model, adapted_params, X_s, y_s, self.criterion, self.inner_lr
                )

            # ---- Outer loss: evaluate on query set with adapted params ----
            query_logits = functional_call(self.model, adapted_params, (X_q,))
            task_loss = self.criterion(query_logits, y_q)
            meta_loss = meta_loss + task_loss

        meta_loss = meta_loss / len(tasks[: self.meta_batch])
        meta_loss.backward()
        self.outer_optim.step()

        return meta_loss.item()

    def adapt(self, X_support: torch.Tensor, y_support: torch.Tensor) -> dict:
        """
        Adapt meta-parameters to a new task without updating θ.
        Returns the adapted parameter dict (for evaluation or RL env).
        """
        adapted = _clone_params(self.model)
        for _ in range(self.inner_steps):
            adapted = _inner_step(
                self.model, adapted, X_support, y_support, self.criterion, self.inner_lr
            )
        return adapted

    def evaluate(
        self,
        X_support: torch.Tensor,
        y_support: torch.Tensor,
        X_query: torch.Tensor,
        y_query: torch.Tensor,
    ) -> float:
        """Adapt then measure query loss.  Used for reporting / RL reward."""
        adapted = self.adapt(X_support, y_support)
        with torch.no_grad():
            logits = functional_call(self.model, adapted, (X_query,))
            loss = self.criterion(logits, y_query)
        return loss.item()