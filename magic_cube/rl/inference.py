"""Safe loading and inference for a local Stable-Baselines3 checkpoint."""

from __future__ import annotations

from pathlib import Path
from collections.abc import Iterable
from typing import Any

from magic_cube.core.moves import INDEX_TO_MOVE, Move
from magic_cube.core.state import CubeState
from magic_cube.rl.observation import MODEL_OBSERVATION_SIZE, encode_model_observation


class ModelLoadError(RuntimeError):
    """Raised when a local RL model cannot be used by the application."""


class RLModelAdapter:
    """Load a 12-action SB3 policy lazily and validate its contract."""

    EXPECTED_ACTION_COUNT = 12
    EXPECTED_OBSERVATION_SIZE = MODEL_OBSERVATION_SIZE

    def __init__(self, checkpoint: str | Path = "models/cube_solver.zip") -> None:
        self.checkpoint = Path(checkpoint)
        self._model: Any | None = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return
        if not self.checkpoint.exists():
            raise ModelLoadError(
                f"找不到强化学习模型：{self.checkpoint}. "
                "请先运行 uv run python scripts/train_rl.py，或将预训练模型放入该路径。"
            )
        try:
            from stable_baselines3 import DQN

            model = DQN.load(str(self.checkpoint), device="auto")
        except Exception as exc:  # SB3 may raise several backend-specific errors.
            raise ModelLoadError(f"强化学习模型加载失败：{exc}") from exc

        action_space = getattr(model, "action_space", None)
        if action_space is None or getattr(action_space, "n", None) != self.EXPECTED_ACTION_COUNT:
            raise ModelLoadError("模型动作空间不是 Discrete(12)，无法安全执行。")
        observation_space = getattr(model, "observation_space", None)
        shape = getattr(observation_space, "shape", None)
        if shape != (self.EXPECTED_OBSERVATION_SIZE,):
            raise ModelLoadError(
                f"模型观测维度为 {shape}，期望 {(self.EXPECTED_OBSERVATION_SIZE,)}。"
            )
        self._model = model

    def predict(
        self,
        state: CubeState,
        history: Iterable[Move] = (),
    ) -> Move:
        if self._model is None:
            self.load()
        assert self._model is not None
        try:
            observation = encode_model_observation(state, history)
            action, _ = self._model.predict(observation, deterministic=True)
            index = int(action.item() if hasattr(action, "item") else action)
            return INDEX_TO_MOVE[index]
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            raise ModelLoadError(f"模型输出了非法原子动作：{exc}") from exc
