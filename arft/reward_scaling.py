from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from verl.protocol import DataProto


@dataclass
class RunningMeanStd:
    """
    Numerically-stable running mean/std, compatible with OpenAI Baselines' RunningMeanStd.

    Tracks mean/var in float64 for stability.
    """

    mean: torch.Tensor
    var: torch.Tensor
    count: float

    @classmethod
    def create(cls, shape: tuple[int, ...] = (), epsilon: float = 1e-4) -> RunningMeanStd:
        mean = torch.zeros(shape, dtype=torch.float64)
        var = torch.ones(shape, dtype=torch.float64)
        return cls(mean=mean, var=var, count=float(epsilon))

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        if x.numel() == 0:
            return
        x64 = x.detach().to(dtype=torch.float64)
        batch_count = x64.shape[0]
        batch_mean = x64.mean(dim=0)
        batch_var = x64.var(dim=0, unbiased=False)
        self._update_from_moments(batch_mean=batch_mean, batch_var=batch_var, batch_count=batch_count)

    @torch.no_grad()
    def _update_from_moments(self, batch_mean: torch.Tensor, batch_var: torch.Tensor, batch_count: int) -> None:
        if batch_count <= 0:
            return

        delta = batch_mean - self.mean
        tot_count = self.count + float(batch_count)

        new_mean = self.mean + delta * (float(batch_count) / tot_count)

        m_a = self.var * self.count
        m_b = batch_var * float(batch_count)
        m2 = m_a + m_b + (delta**2) * (self.count * float(batch_count) / tot_count)
        new_var = m2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count


def _compute_discounted_returns_per_step(
    step_rewards: torch.Tensor,
    trajectory_uids: np.ndarray,
    step_indices: np.ndarray,
    gamma: float,
) -> torch.Tensor:
    """
    Compute discounted return R_t = gamma * R_{t-1} + r_t for each (trajectory_uid, step_index).

    Args:
        step_rewards: (N,) immediate reward per step (one row per step).
        trajectory_uids: (N,) numpy array grouping steps into trajectories.
        step_indices: (N,) numpy array ordering steps within each trajectory (0..T-1).
        gamma: discount factor.

    Returns:
        discounted_returns: (N,) tensor aligned with input rows.
    """
    if step_rewards.numel() == 0:
        return step_rewards

    device = step_rewards.device
    dtype = step_rewards.dtype

    # Map trajectory_uids (may be dtype=object) to contiguous integer ids via numpy.
    unique_uids, inv = np.unique(trajectory_uids, return_inverse=True)
    num_traj = int(len(unique_uids))
    inv_t = torch.as_tensor(inv, device=device, dtype=torch.long)
    step_t = torch.as_tensor(step_indices, device=device, dtype=torch.long)

    max_step = int(step_t.max().item()) + 1
    rewards_map = torch.zeros((num_traj, max_step), device=device, dtype=dtype)
    rewards_map[inv_t, step_t] = step_rewards

    returns_map = torch.zeros_like(rewards_map)
    running = torch.zeros((num_traj,), device=device, dtype=dtype)
    g = torch.as_tensor(gamma, device=device, dtype=dtype)
    for t in range(max_step):
        running = running * g + rewards_map[:, t]
        returns_map[:, t] = running

    discounted_returns = returns_map[inv_t, step_t]
    return discounted_returns


class RewardScalingByReturnStd:
    """
    VecNormalize-style reward scaling using running std of discounted returns.

    Maintain running statistics of R_t, where:
        R_t = gamma * R_{t-1} + r_t
    And scale current immediate reward r_t by:
        r_t / sqrt(Var(R) + eps)

    Notes:
    - We DO NOT subtract mean (scale-only).
    - In agent_flow, "environment step" corresponds to one AgentFlowStep (one generated sequence row).
      We define r_t as step-level sum of token rewards under response_mask, and apply the scale factor
      to the whole token-level reward row.
    """

    def __init__(self, gamma: float, eps: float = 1e-8) -> None:
        self.gamma = float(gamma)
        self.eps = float(eps)
        self.ret_rms = RunningMeanStd.create(shape=(), epsilon=1e-4)

    @torch.no_grad()
    def scale_batch(self, batch: DataProto) -> tuple[DataProto, dict[str, Any]]:
        if "token_level_rewards" not in batch.batch:
            return batch, {}

        # Valid (non-pad) rows only
        is_pad = batch.non_tensor_batch.get("is_pad", None)
        if is_pad is not None:
            valid_mask = torch.from_numpy(~is_pad).to(batch.batch.device)
        else:
            valid_mask = torch.ones(len(batch), dtype=torch.bool, device=batch.batch.device)

        if not torch.any(valid_mask):
            return batch, {"count": 0}

        rewards = batch.batch["token_level_rewards"]
        response_mask = batch.batch.get("response_mask", None)
        if response_mask is None:
            # Fall back to attention_mask's response segment is not robust here; require response_mask in agent_flow.
            raise KeyError("reward scaling requires batch.batch['response_mask'] to compute step-level rewards")

        # Step-level immediate reward r_t
        step_rewards = (rewards * response_mask).sum(dim=1)
        step_rewards_valid = step_rewards[valid_mask]

        # Discounted return R_t per step inside each trajectory.
        trajectory_uids = batch.non_tensor_batch["trajectory_uids"]
        step_indices = batch.non_tensor_batch["step_indices"]
        discounted_returns = _compute_discounted_returns_per_step(
            step_rewards=step_rewards_valid,
            trajectory_uids=trajectory_uids[valid_mask.cpu().numpy()],
            step_indices=step_indices[valid_mask.cpu().numpy()],
            gamma=self.gamma,
        )

        # Update RMS with current discounted returns (VecNormalize behavior).
        self.ret_rms.update(discounted_returns.detach().cpu())

        # Scale factor from running std of discounted returns.
        std = torch.sqrt(self.ret_rms.var + self.eps).to(dtype=torch.float32)
        scale = float((1.0 / std).item())

        # Apply scale-only to token-level rewards, per-row constant factor.
        rewards = rewards.clone()
        rewards[valid_mask] = rewards[valid_mask] * scale
        batch.batch["token_level_rewards"] = rewards

        metrics = {
            "scale": scale,
            "return_std": float(std.item()),
            "return_var": float(self.ret_rms.var.item()),
            "return_mean": float(self.ret_rms.mean.item()),
            "count": int(valid_mask.sum().item()),
            "step_reward_mean": float(step_rewards_valid.mean().item()),
            "step_reward_std": float(step_rewards_valid.std(unbiased=False).item()),
        }
        return batch, metrics
