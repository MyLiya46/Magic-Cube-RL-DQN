"""Canonical move-sequence rules and their reward consequences."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from enum import Enum

from magic_cube.core.moves import Move, coerce_move


class RedundancyKind(str, Enum):
    IMMEDIATE_INVERSE = "immediate_inverse"
    THIRD_SAME_FACE = "third_same_face"
    FULL_FOUR_TURN_CYCLE = "full_four_turn_cycle"


@dataclass(frozen=True, slots=True)
class RewardConfig:
    """Central reward policy used by every training and evaluation environment."""

    solved_reward: float = 1.0
    step_penalty: float = -0.01
    progress_scale: float = 0.01
    immediate_inverse_penalty: float = -0.10
    third_same_face_penalty: float = -0.08
    full_four_turn_cycle_penalty: float = -0.15

    def redundancy_penalty(self, kind: RedundancyKind | None) -> float:
        if kind is RedundancyKind.IMMEDIATE_INVERSE:
            return self.immediate_inverse_penalty
        if kind is RedundancyKind.THIRD_SAME_FACE:
            return self.third_same_face_penalty
        if kind is RedundancyKind.FULL_FOUR_TURN_CYCLE:
            return self.full_four_turn_cycle_penalty
        return 0.0

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


DEFAULT_REWARD_CONFIG = RewardConfig()


def classify_redundancy(
    history: Iterable[Move | str | int],
    candidate: Move | str | int,
) -> RedundancyKind | None:
    """Classify mathematically reducible local move patterns.

    Two equal turns are deliberately allowed because they represent a 180°
    face turn, which is not a separate action in the 12-action policy.
    """

    recent = tuple(coerce_move(move) for move in history)
    move = coerce_move(candidate)
    if len(recent) >= 3 and all(previous == move for previous in recent[-3:]):
        return RedundancyKind.FULL_FOUR_TURN_CYCLE
    if recent and recent[-1].inverse == move:
        return RedundancyKind.IMMEDIATE_INVERSE
    if len(recent) >= 2 and all(
        previous.face == move.face for previous in recent[-2:]
    ):
        return RedundancyKind.THIRD_SAME_FACE
    return None
