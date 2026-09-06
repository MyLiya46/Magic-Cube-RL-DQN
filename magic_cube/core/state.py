"""A deterministic, sticker-level 3x3x3 cube state."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .geometry import (
    FACE_ORDER,
    GEOMETRY_TO_INDEX,
    STICKER_GEOMETRY,
    move_layer_contains,
    rotate_for_move,
)
from .moves import MOVE_ORDER, Move, coerce_move

COLOR_INDEX: dict[str, int] = {face: index for index, face in enumerate(FACE_ORDER)}
COLOR_NAMES: tuple[str, ...] = FACE_ORDER
ONE_HOT_COLORS = np.eye(len(FACE_ORDER), dtype=np.float32)


@dataclass(frozen=True, slots=True)
class Scramble:
    """A generated scramble and its primitive moves."""

    moves: tuple[Move, ...]


class CubeState:
    """The 54-sticker state of a standard 3x3x3 cube.

    Facelets are stored in row-major order for ``FACE_ORDER`` (U, F, R, L, B,
    D).  Values are integer colour ids in the same order.  The geometric move
    implementation preserves sticker counts and orientation by moving each
    sticker's position and normal together.
    """

    STICKER_COUNT = 54
    OBSERVATION_SIZE = STICKER_COUNT * len(FACE_ORDER)

    def __init__(self, facelets: Iterable[int] | None = None) -> None:
        values = (
            tuple(COLOR_INDEX[face] for face in FACE_ORDER for _ in range(9))
            if facelets is None
            else tuple(facelets)
        )
        if len(values) != self.STICKER_COUNT:
            raise ValueError(f"A cube state requires 54 facelets, got {len(values)}")
        if any(value not in range(len(FACE_ORDER)) for value in values):
            raise ValueError("Facelet colour ids must be integers from 0 through 5")
        self._facelets = list(values)

    @classmethod
    def solved(cls) -> "CubeState":
        return cls()

    @classmethod
    def from_facelets(cls, facelets: Iterable[int]) -> "CubeState":
        return cls(facelets)

    def copy(self) -> "CubeState":
        return type(self)(self._facelets)

    @property
    def facelets(self) -> tuple[int, ...]:
        """Return an immutable snapshot of the state."""

        return tuple(self._facelets)

    def apply_move(self, move: Move | str | int) -> None:
        """Apply exactly one primitive quarter-turn in place."""

        primitive = coerce_move(move)
        new_facelets: list[int | None] = [None] * self.STICKER_COUNT
        for source_index, sticker in enumerate(STICKER_GEOMETRY):
            if move_layer_contains(sticker.position, primitive.value):
                new_position = rotate_for_move(sticker.position, primitive.value)
                new_normal = rotate_for_move(sticker.normal, primitive.value)
                destination_index = GEOMETRY_TO_INDEX[(new_position, new_normal)]
            else:
                destination_index = source_index
            new_facelets[destination_index] = self._facelets[source_index]

        if any(value is None for value in new_facelets):
            raise RuntimeError(f"Move {primitive.value} produced an incomplete sticker mapping")
        self._facelets = [int(value) for value in new_facelets]

    def apply_sequence(self, moves: Iterable[Move | str | int]) -> None:
        for move in moves:
            self.apply_move(move)

    def is_solved(self) -> bool:
        return all(
            self._facelets[offset + index] == face_index
            for face_index in range(6)
            for offset in (face_index * 9,)
            for index in range(9)
        )

    def observation(self) -> np.ndarray:
        """Return a 324-value one-hot observation used by the RL environment."""

        indices = np.asarray(self._facelets, dtype=np.intp)
        return ONE_HOT_COLORS[indices].reshape(self.OBSERVATION_SIZE)

    def correct_facelet_count(self) -> int:
        """Count stickers currently occupying a solved-position colour slot."""

        return sum(
            self._facelets[face_index * 9 + offset] == face_index
            for face_index in range(6)
            for offset in range(9)
        )

    def face_grid(self, face: str) -> tuple[tuple[int, ...], ...]:
        """Return one face as three rows of colour ids."""

        try:
            face_index = FACE_ORDER.index(face.upper())
        except ValueError as exc:
            raise ValueError(f"Unknown face: {face!r}") from exc
        start = face_index * 9
        return tuple(
            tuple(self._facelets[start + row * 3 : start + row * 3 + 3])
            for row in range(3)
        )

    @staticmethod
    def generate_scramble(
        length: int = 20,
        rng: random.Random | None = None,
        avoid_same_face: bool = True,
    ) -> Scramble:
        """Generate a sequence containing only the 12 primitive actions."""

        if length < 0:
            raise ValueError("Scramble length must be non-negative")
        random_source = rng or random.Random()
        moves: list[Move] = []
        for _ in range(length):
            candidates = list(MOVE_ORDER)
            if avoid_same_face and moves:
                candidates = [move for move in candidates if move.face != moves[-1].face]
            moves.append(random_source.choice(candidates))
        return Scramble(tuple(moves))

    @classmethod
    def scrambled(
        cls,
        length: int = 20,
        rng: random.Random | None = None,
    ) -> tuple["CubeState", Scramble]:
        """Create a state and the primitive scramble that produced it."""

        while True:
            scramble = cls.generate_scramble(length, rng=rng)
            state = cls.solved()
            state.apply_sequence(scramble.moves)
            if length == 0 or not state.is_solved():
                return state, scramble
