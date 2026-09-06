"""Sticker geometry shared by the state engine and the 3D renderer.

Coordinates use a right-handed system:

* x: L (-) to R (+)
* y: D (-) to U (+)
* z: B (-) to F (+)

Every sticker is identified by its cubie position and outward normal.  This
means a move can be implemented as a geometric quarter-turn instead of a
large, error-prone hand-written adjacency table.
"""

from __future__ import annotations

from dataclasses import dataclass

Vector = tuple[int, int, int]

FACE_ORDER: tuple[str, ...] = ("U", "F", "R", "L", "B", "D")
FACE_NORMALS: dict[str, Vector] = {
    "U": (0, 1, 0),
    "F": (0, 0, 1),
    "R": (1, 0, 0),
    "L": (-1, 0, 0),
    "B": (0, 0, -1),
    "D": (0, -1, 0),
}

# ``right`` and ``down`` describe how a face is viewed from outside.  They
# are used both to establish sticker indexing and to draw the sticker quad.
FACE_BASIS: dict[str, tuple[Vector, Vector]] = {
    "U": ((1, 0, 0), (0, 0, 1)),
    "F": ((1, 0, 0), (0, -1, 0)),
    "R": ((0, 0, -1), (0, -1, 0)),
    "L": ((0, 0, 1), (0, -1, 0)),
    "B": ((-1, 0, 0), (0, -1, 0)),
    "D": ((1, 0, 0), (0, 0, -1)),
}


@dataclass(frozen=True, slots=True)
class StickerGeometry:
    """The immutable location and local basis of one sticker."""

    face: str
    position: Vector
    normal: Vector
    right: Vector
    down: Vector


def _add(a: Vector, b: Vector) -> Vector:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _scale(a: Vector, scalar: int) -> Vector:
    return (a[0] * scalar, a[1] * scalar, a[2] * scalar)


def build_sticker_geometry() -> tuple[StickerGeometry, ...]:
    """Return the 54 stickers in row-major order for ``FACE_ORDER``."""

    result: list[StickerGeometry] = []
    for face in FACE_ORDER:
        normal = FACE_NORMALS[face]
        right, down = FACE_BASIS[face]
        face_origin = normal
        for row in range(3):
            for column in range(3):
                horizontal = column - 1
                vertical = row - 1
                position = _add(
                    face_origin,
                    _add(_scale(right, horizontal), _scale(down, vertical)),
                )
                result.append(
                    StickerGeometry(
                        face=face,
                        position=position,
                        normal=normal,
                        right=right,
                        down=down,
                    )
                )
    return tuple(result)


STICKER_GEOMETRY: tuple[StickerGeometry, ...] = build_sticker_geometry()
GEOMETRY_TO_INDEX: dict[tuple[Vector, Vector], int] = {
    (sticker.position, sticker.normal): index
    for index, sticker in enumerate(STICKER_GEOMETRY)
}


def rotate_quarter(vector: Vector, axis: int, turns: int) -> Vector:
    """Rotate an integer vector by +90 or -90 degrees around +x/+y/+z.

    ``turns`` may be any integer; only its parity modulo four matters.
    """

    turns %= 4
    result = vector
    for _ in range(turns):
        x, y, z = result
        if axis == 0:  # +90 around x: y -> z
            result = (x, -z, y)
        elif axis == 1:  # +90 around y: z -> x
            result = (z, y, -x)
        elif axis == 2:  # +90 around z: x -> y
            result = (-y, x, z)
        else:
            raise ValueError(f"Invalid rotation axis: {axis}")
    return result


def rotate_for_move(vector: Vector, move: str) -> Vector:
    """Rotate a vector as part of one clockwise or counter-clockwise turn."""

    face = move[0]
    normal = FACE_NORMALS[face]
    axis = next(index for index, component in enumerate(normal) if component)
    # Clockwise when viewed from outside is -90 around the outward normal.
    # A signed normal converts that into a turn around the positive axis.
    turns = -normal[axis]
    if move.endswith("'"):
        turns = -turns
    return rotate_quarter(vector, axis, turns)


def move_layer_contains(position: Vector, move: str) -> bool:
    """Whether a sticker position lies on the face layer moved by ``move``."""

    normal = FACE_NORMALS[move[0]]
    axis = next(index for index, component in enumerate(normal) if component)
    return position[axis] == normal[axis]
