"""A small fixed-function OpenGL view for the cube stickers."""

from __future__ import annotations

import math
import time
from collections.abc import Callable

from PySide6.QtCore import QTimer, Qt
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import QSizePolicy
from OpenGL.GL import (
    GL_COLOR_BUFFER_BIT,
    GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_TEST,
    GL_MODELVIEW,
    GL_PROJECTION,
    GL_QUADS,
    glBegin,
    glClear,
    glClearColor,
    glColor3f,
    glEnable,
    glEnd,
    glLoadIdentity,
    glMatrixMode,
    glRotatef,
    glTranslatef,
    glVertex3f,
    glViewport,
)
from OpenGL.GLU import gluPerspective

from magic_cube.core.geometry import FACE_NORMALS, STICKER_GEOMETRY
from magic_cube.core.moves import Move
from magic_cube.core.state import CubeState

COLOR_RGB: dict[int, tuple[float, float, float]] = {
    0: (0.95, 0.95, 0.95),  # U white
    1: (0.10, 0.65, 0.16),  # F green
    2: (0.80, 0.08, 0.08),  # R red
    3: (1.00, 0.48, 0.03),  # L orange
    4: (0.08, 0.25, 0.75),  # B blue
    5: (0.95, 0.82, 0.05),  # D yellow
}


def _add(a: tuple[float, float, float], b: tuple[float, float, float]):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _scale(a: tuple[float, float, float], amount: float):
    return (a[0] * amount, a[1] * amount, a[2] * amount)


def _rotate(vector: tuple[float, float, float], normal: tuple[int, int, int], degrees: float):
    """Rotate around a signed coordinate normal using Rodrigues' formula."""

    axis = tuple(float(component) for component in normal)
    length = math.sqrt(sum(component * component for component in axis))
    axis = tuple(component / length for component in axis)
    radians = math.radians(degrees)
    cosine = math.cos(radians)
    sine = math.sin(radians)
    dot = sum(vector[index] * axis[index] for index in range(3))
    cross = (
        axis[1] * vector[2] - axis[2] * vector[1],
        axis[2] * vector[0] - axis[0] * vector[2],
        axis[0] * vector[1] - axis[1] * vector[0],
    )
    return tuple(
        vector[index] * cosine
        + cross[index] * sine
        + axis[index] * dot * (1.0 - cosine)
        for index in range(3)
    )


class CubeView(QOpenGLWidget):
    """Render cube stickers and animate one layer at a time."""

    def __init__(self, state: CubeState, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumSize(520, 520)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._state = state.copy()
        self._animation_move: Move | None = None
        self._animation_started = 0.0
        self._animation_duration = 0.5
        self._animation_callback: Callable[[], None] | None = None
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick_animation)
        self._yaw = -32.0
        self._pitch = 24.0
        self._last_mouse_position = None

    @property
    def is_animating(self) -> bool:
        return self._animation_move is not None

    def set_state(self, state: CubeState) -> None:
        self._state = state.copy()
        self.update()

    def stop_animation(self) -> None:
        self._timer.stop()
        self._animation_move = None
        self._animation_callback = None
        self.update()

    def animate_move(
        self,
        move: Move,
        on_complete: Callable[[], None],
        duration: float = 0.5,
    ) -> bool:
        if self.is_animating:
            return False
        self._animation_move = move
        self._animation_started = time.perf_counter()
        self._animation_duration = duration
        self._animation_callback = on_complete
        self._timer.start(16)
        self.update()
        return True

    def initializeGL(self) -> None:
        glClearColor(0.055, 0.075, 0.11, 1.0)
        glEnable(GL_DEPTH_TEST)

    def resizeGL(self, width: int, height: int) -> None:
        glViewport(0, 0, width, max(1, height))
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(42.0, width / max(1, height), 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)

    def paintGL(self) -> None:
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glTranslatef(0.0, 0.0, -8.2)
        glRotatef(self._pitch, 1.0, 0.0, 0.0)
        glRotatef(self._yaw, 0.0, 1.0, 0.0)
        self._draw_dark_cube()
        progress = self._animation_progress()
        self._draw_stickers(progress)

    def _animation_progress(self) -> float:
        if self._animation_move is None:
            return 0.0
        return min(1.0, (time.perf_counter() - self._animation_started) / self._animation_duration)

    def _tick_animation(self) -> None:
        if self._animation_move is None:
            self._timer.stop()
            return
        progress = self._animation_progress()
        self.update()
        if progress >= 1.0:
            callback = self._animation_callback
            self._animation_move = None
            self._animation_callback = None
            self._timer.stop()
            self.update()
            if callback is not None:
                callback()

    def _draw_dark_cube(self) -> None:
        """Draw a dark body behind the stickers, leaving visible sticker gaps."""

        glColor3f(0.015, 0.018, 0.025)
        faces = (
            ((0.0, 0.0, 1.52), ((-1.5, -1.5), (1.5, -1.5), (1.5, 1.5), (-1.5, 1.5))),
            ((0.0, 0.0, -1.52), ((1.5, -1.5), (-1.5, -1.5), (-1.5, 1.5), (1.5, 1.5))),
            ((1.52, 0.0, 0.0), ((-1.5, -1.5), (-1.5, 1.5), (1.5, 1.5), (1.5, -1.5))),
            ((-1.52, 0.0, 0.0), ((-1.5, 1.5), (-1.5, -1.5), (1.5, -1.5), (1.5, 1.5))),
            ((0.0, 1.52, 0.0), ((-1.5, -1.5), (1.5, -1.5), (1.5, 1.5), (-1.5, 1.5))),
            ((0.0, -1.52, 0.0), ((-1.5, 1.5), (1.5, 1.5), (1.5, -1.5), (-1.5, -1.5))),
        )
        # A subtle body is enough; sticker quads provide the actual face layout.
        for center, points in faces:
            glBegin(GL_QUADS)
            for first, second in points:
                if center[0]:
                    glVertex3f(center[0], first, second)
                elif center[1]:
                    glVertex3f(first, center[1], second)
                else:
                    glVertex3f(first, second, center[2])
            glEnd()

    def _draw_stickers(self, progress: float) -> None:
        moving = self._animation_move
        move_normal = FACE_NORMALS[moving.face] if moving is not None else None
        direction = 1.0 if moving is not None and moving.is_prime else -1.0
        angle = direction * 90.0 * progress if moving is not None else 0.0
        for index, sticker in enumerate(STICKER_GEOMETRY):
            position = tuple(float(component) for component in sticker.position)
            normal = tuple(float(component) for component in sticker.normal)
            right = tuple(float(component) for component in sticker.right)
            down = tuple(float(component) for component in sticker.down)
            if moving is not None and self._sticker_is_on_layer(sticker.position, move_normal):
                position = _rotate(position, move_normal, angle)
                normal = _rotate(normal, move_normal, angle)
                right = _rotate(right, move_normal, angle)
                down = _rotate(down, move_normal, angle)
            center = _add(position, _scale(normal, 0.535))
            half = 0.425
            corners = (
                _add(_add(center, _scale(right, -half)), _scale(down, -half)),
                _add(_add(center, _scale(right, half)), _scale(down, -half)),
                _add(_add(center, _scale(right, half)), _scale(down, half)),
                _add(_add(center, _scale(right, -half)), _scale(down, half)),
            )
            red, green, blue = COLOR_RGB[self._state.facelets[index]]
            glColor3f(red, green, blue)
            glBegin(GL_QUADS)
            for corner in corners:
                glVertex3f(*corner)
            glEnd()

    @staticmethod
    def _sticker_is_on_layer(position, move_normal) -> bool:
        if move_normal is None:
            return False
        axis = next(index for index, component in enumerate(move_normal) if component)
        return position[axis] == move_normal[axis]

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._last_mouse_position = event.position()

    def mouseMoveEvent(self, event) -> None:
        if self._last_mouse_position is None:
            return
        current = event.position()
        delta = current - self._last_mouse_position
        self._yaw += delta.x() * 0.6
        self._pitch = max(-80.0, min(80.0, self._pitch + delta.y() * 0.6))
        self._last_mouse_position = current
        self.update()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._last_mouse_position = None
