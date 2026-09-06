"""Versioned model observation encoding.

The cube geometry and recent action context are encoded separately and then
concatenated.  Keeping this contract in one module prevents training and GUI
inference from silently disagreeing about feature order.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Iterator

import numpy as np

from magic_cube.core.moves import MOVE_ORDER, MOVE_TO_INDEX, Move, coerce_move
from magic_cube.core.state import CubeState

MODEL_SCHEMA_VERSION = 2
HISTORY_LENGTH = 3
HISTORY_TOKEN_COUNT = len(MOVE_ORDER) + 1
NO_MOVE_INDEX = len(MOVE_ORDER)
CUBE_OBSERVATION_SIZE = CubeState.OBSERVATION_SIZE
HISTORY_OBSERVATION_SIZE = HISTORY_LENGTH * HISTORY_TOKEN_COUNT
MODEL_OBSERVATION_SIZE = CUBE_OBSERVATION_SIZE + HISTORY_OBSERVATION_SIZE


class ActionHistory:
    """Bounded recent-action context shared by environments and inference."""

    def __init__(self, moves: Iterable[Move | str | int] = ()) -> None:
        self._moves: deque[Move] = deque(maxlen=HISTORY_LENGTH)
        for move in moves:
            self.append(move)

    def append(self, move: Move | str | int) -> None:
        self._moves.append(coerce_move(move))

    def clear(self) -> None:
        self._moves.clear()

    @property
    def moves(self) -> tuple[Move, ...]:
        return tuple(self._moves)

    def __len__(self) -> int:
        return len(self._moves)

    def __iter__(self) -> Iterator[Move]:
        return iter(self._moves)


def encode_action_history(history: Iterable[Move | str | int]) -> np.ndarray:
    """Encode the latest three actions, left-padded with a no-action token."""

    recent = tuple(coerce_move(move) for move in history)[-HISTORY_LENGTH:]
    token_indices = [NO_MOVE_INDEX] * (HISTORY_LENGTH - len(recent))
    token_indices.extend(MOVE_TO_INDEX[move] for move in recent)
    encoded = np.zeros((HISTORY_LENGTH, HISTORY_TOKEN_COUNT), dtype=np.float32)
    encoded[np.arange(HISTORY_LENGTH), token_indices] = 1.0
    return encoded.reshape(HISTORY_OBSERVATION_SIZE)


def encode_model_observation(
    state: CubeState,
    history: Iterable[Move | str | int] = (),
) -> np.ndarray:
    """Build the complete, Markov-compatible policy observation."""

    return np.concatenate(
        (state.observation(), encode_action_history(history)),
        dtype=np.float32,
    )
