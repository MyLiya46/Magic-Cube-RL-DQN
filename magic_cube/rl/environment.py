"""Gymnasium environment for a standard 12-action cube policy."""

from __future__ import annotations

import random
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from magic_cube.core.moves import MOVE_ORDER, coerce_move
from magic_cube.core.state import CubeState
from magic_cube.rl.action_rules import (
    DEFAULT_REWARD_CONFIG,
    RewardConfig,
    classify_redundancy,
)
from magic_cube.rl.observation import (
    MODEL_OBSERVATION_SIZE,
    ActionHistory,
    encode_model_observation,
)


def recommended_max_steps(scramble_depth: int) -> int:
    """Return a compact horizon that still allows a few corrective moves."""

    if scramble_depth < 0:
        raise ValueError("scramble_depth must be non-negative")
    return max(4, scramble_depth * 2 + 4)


class CubeEnv(gym.Env[np.ndarray, int]):
    """A local environment with clockwise and inverse quarter turns."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        scramble_depth: int = 5,
        min_scramble_depth: int | None = None,
        max_steps: int | None = None,
        render_mode: str | None = None,
        reward_config: RewardConfig = DEFAULT_REWARD_CONFIG,
        focus_depth: int | None = None,
        focus_depth_probability: float = 0.0,
    ) -> None:
        super().__init__()
        if scramble_depth < 0:
            raise ValueError("scramble_depth must be non-negative")
        if min_scramble_depth is not None and not 0 <= min_scramble_depth <= scramble_depth:
            raise ValueError("min_scramble_depth must be between 0 and scramble_depth")
        if max_steps is not None and max_steps <= 0:
            raise ValueError("max_steps must be greater than zero")
        if not 0.0 <= focus_depth_probability <= 1.0:
            raise ValueError("focus_depth_probability must be between 0 and 1")
        self.scramble_depth = scramble_depth
        self.min_scramble_depth = (
            scramble_depth if min_scramble_depth is None else min_scramble_depth
        )
        self.max_steps = (
            recommended_max_steps(scramble_depth)
            if max_steps is None
            else max_steps
        )
        self.render_mode = render_mode
        self.reward_config = reward_config
        if focus_depth is not None and not (
            self.min_scramble_depth <= focus_depth <= self.scramble_depth
        ):
            raise ValueError("focus_depth must be within the scramble depth range")
        if focus_depth is None and focus_depth_probability:
            raise ValueError(
                "focus_depth is required when focus_depth_probability is non-zero"
            )
        self.focus_depth = focus_depth
        self.focus_depth_probability = focus_depth_probability
        self._non_focus_depths = tuple(
            depth
            for depth in range(self.min_scramble_depth, self.scramble_depth + 1)
            if depth != focus_depth
        )
        self.action_space = spaces.Discrete(len(MOVE_ORDER))
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(MODEL_OBSERVATION_SIZE,),
            dtype=np.float32,
        )
        self.state = CubeState.solved()
        self.history = ActionHistory()
        self.steps = 0
        self.redundant_actions = 0
        self.scramble = tuple()

    def _sample_scramble_depth(self, random_source: random.Random) -> int:
        if self.focus_depth is None or self.focus_depth_probability == 0.0:
            return random_source.randint(self.min_scramble_depth, self.scramble_depth)
        if (
            self.focus_depth_probability == 1.0
            or not self._non_focus_depths
            or random_source.random() < self.focus_depth_probability
        ):
            return self.focus_depth
        return random_source.choice(self._non_focus_depths)

    def _observation(self) -> np.ndarray:
        return encode_model_observation(self.state, self.history)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        random_source = random.Random(seed) if seed is not None else random.Random()
        selected_depth = self._sample_scramble_depth(random_source)
        self.state, scramble = CubeState.scrambled(selected_depth, rng=random_source)
        self.scramble = scramble.moves
        self.history.clear()
        self.steps = 0
        self.redundant_actions = 0
        return self._observation(), {
            "scramble": tuple(move.value for move in self.scramble),
            "scramble_depth": selected_depth,
        }

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action!r}")
        move = coerce_move(int(action))
        redundancy_kind = classify_redundancy(self.history, move)
        redundancy_penalty = self.reward_config.redundancy_penalty(
            redundancy_kind
        )
        correct_before = self.state.correct_facelet_count()
        self.state.apply_move(move)
        correct_after = self.state.correct_facelet_count()
        self.history.append(move)
        self.steps += 1
        if redundancy_kind is not None:
            self.redundant_actions += 1
        terminated = self.state.is_solved()
        truncated = self.steps >= self.max_steps and not terminated
        progress_reward = self.reward_config.progress_scale * (
            correct_after - correct_before
        )
        reward = (
            self.reward_config.solved_reward
            if terminated
            else self.reward_config.step_penalty + progress_reward
        )
        reward += redundancy_penalty
        return self._observation(), reward, terminated, truncated, {
            "move": move.value,
            "steps": self.steps,
            "is_success": terminated,
            "redundancy_kind": (
                redundancy_kind.value if redundancy_kind is not None else None
            ),
            "redundancy_penalty": redundancy_penalty,
            "redundant_actions": self.redundant_actions,
            "reward_terms": {
                "progress": progress_reward,
                "redundancy": redundancy_penalty,
            },
        }
