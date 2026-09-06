"""The standard 12-action quarter-turn move set used by RL policies.

Each face has a clockwise and counter-clockwise action.  Half turns remain a
two-action sequence, keeping the branching factor below the 18-action face
turn metric while making every move reversible in a single decision.
"""

from __future__ import annotations

from enum import Enum


class Move(str, Enum):
    """A single 90-degree turn, viewed from the turned face."""

    U = "U"
    U_PRIME = "U'"
    F = "F"
    F_PRIME = "F'"
    R = "R"
    R_PRIME = "R'"
    L = "L"
    L_PRIME = "L'"
    B = "B"
    B_PRIME = "B'"
    D = "D"
    D_PRIME = "D'"

    @property
    def face(self) -> str:
        return self.value[0]

    @property
    def is_prime(self) -> bool:
        return self.value.endswith("'")

    @property
    def inverse(self) -> "Move":
        return Move(self.face if self.is_prime else f"{self.face}'")


MOVE_ORDER: tuple[Move, ...] = (
    Move.U,
    Move.U_PRIME,
    Move.F,
    Move.F_PRIME,
    Move.R,
    Move.R_PRIME,
    Move.L,
    Move.L_PRIME,
    Move.B,
    Move.B_PRIME,
    Move.D,
    Move.D_PRIME,
)

MOVE_TO_INDEX: dict[Move, int] = {move: index for index, move in enumerate(MOVE_ORDER)}
INDEX_TO_MOVE: dict[int, Move] = {index: move for index, move in enumerate(MOVE_ORDER)}


def coerce_move(value: Move | str | int) -> Move:
    """Convert a public action representation to :class:`Move`.

    Invalid values raise ``ValueError`` rather than being silently coerced.
    """

    if isinstance(value, Move):
        return value
    if isinstance(value, int):
        try:
            return INDEX_TO_MOVE[value]
        except KeyError as exc:
            raise ValueError(f"Invalid action index: {value}") from exc
    try:
        return Move(str(value).upper())
    except ValueError as exc:
        raise ValueError(f"Invalid primitive move: {value!r}") from exc
