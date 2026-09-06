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
    ) -> None:
        super().__init__()
        if scramble_depth < 0:
            raise ValueError("scramble_depth must be non-negative")
        if min_scramble_depth is not None and not 0 <= min_scramble_depth <= scramble_depth:
            raise ValueError("min_scramble_depth must be between 0 and scramble_depth")
        self.scramble_depth = scramble_depth
        self.min_scramble_depth = (
            scramble_depth if min_scramble_depth is None else min_scramble_depth
        )
        self.max_steps = max_steps or max(100, scramble_depth * 2 + 10)
        self.render_mode = render_mode
        self.reward_config = reward_config
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
        selected_depth = random_source.randint(
            self.min_scramble_depth, self.scramble_depth
        )
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
