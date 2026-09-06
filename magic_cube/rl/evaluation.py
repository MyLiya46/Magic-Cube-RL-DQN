"""Deterministic evaluation helpers shared by training and CLI tools."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from magic_cube.rl.environment import CubeEnv


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    episodes: int
    successes: int
    solved_steps: tuple[int, ...]
    total_actions: int
    redundant_actions: int

    @property
    def success_rate(self) -> float:
        return self.successes / self.episodes

    @property
    def average_steps(self) -> float:
        return (
            sum(self.solved_steps) / len(self.solved_steps)
            if self.solved_steps
            else 0.0
        )

    @property
    def redundancy_rate(self) -> float:
        return (
            self.redundant_actions / self.total_actions
            if self.total_actions
            else 0.0
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["success_rate"] = self.success_rate
        result["average_steps_on_success"] = self.average_steps
        result["redundancy_rate"] = self.redundancy_rate
        return result


def evaluate_model(
    model: Any,
    *,
    scramble_depth: int,
    episodes: int,
    max_steps: int,
    seed: int = 10_000,
) -> EvaluationResult:
    """Evaluate exact-depth random states without exploration."""

    if episodes <= 0:
        raise ValueError("episodes must be greater than zero")
    env = CubeEnv(scramble_depth=scramble_depth, max_steps=max_steps)
    successes = 0
    solved_steps: list[int] = []
    total_actions = 0
    redundant_actions = 0
    try:
        for episode in range(episodes):
            observation, _ = env.reset(seed=seed + episode)
            for step in range(1, max_steps + 1):
                action, _ = model.predict(observation, deterministic=True)
                action_index = int(np.asarray(action).reshape(-1)[0])
                observation, _, terminated, truncated, info = env.step(action_index)
                total_actions += 1
                if info["redundancy_kind"] is not None:
                    redundant_actions += 1
                if terminated:
                    successes += 1
                    solved_steps.append(step)
                    break
                if truncated:
                    break
    finally:
        env.close()
    return EvaluationResult(
        episodes=episodes,
        successes=successes,
        solved_steps=tuple(solved_steps),
        total_actions=total_actions,
        redundant_actions=redundant_actions,
    )
