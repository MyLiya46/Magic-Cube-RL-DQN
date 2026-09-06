"""The main PySide6 window and the action/animation coordinator."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QFont, QKeyEvent, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QHBoxLayout,
    QGridLayout,
    QComboBox,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from magic_cube.core.moves import MOVE_ORDER, Move
from magic_cube.core.state import CubeState
from magic_cube.rl.inference import ModelLoadError, RLModelAdapter
from magic_cube.rl.observation import ActionHistory
from magic_cube.ui.cube_view import CubeView


class MainWindow(QMainWindow):
    """Coordinate one authoritative state with the OpenGL view and AI loop."""

    MOVE_ANIMATION_SECONDS = 0.5
    MAX_AI_STEPS = 1000
    SCRAMBLE_DEPTHS = (1, 3, 5, 10, 20)
    DEFAULT_SCRAMBLE_DEPTH = 20

    def __init__(self, project_root: Path | None = None) -> None:
        super().__init__()
        self.setWindowTitle("Magic Cube · 十二动作强化学习演示")
        self.resize(1180, 760)
        self.setMinimumSize(980, 650)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._project_root = project_root or Path(__file__).resolve().parents[2]
        self._state, _initial_scramble = CubeState.scrambled(
            self.DEFAULT_SCRAMBLE_DEPTH
        )
        self._model = RLModelAdapter(self._project_root / "models" / "cube_solver.zip")
        # Reward rules apply only inside one policy episode.  User moves are
        # already represented by the cube state and must not leak into this
        # context: every training episode also starts with an empty history.
        self._policy_history = ActionHistory()
        self._ai_running = False
        self._ai_stop_requested = False
        self._move_busy = False
        self._ai_steps = 0
        self._move_buttons: dict[Move, QPushButton] = {}
        self._shortcuts: list[QShortcut] = []

        self._view = CubeView(self._state)
        self._chat = QListWidget()
        self._chat.setObjectName("chatList")
        self._chat.setAlternatingRowColors(True)
        self._chat.setMinimumWidth(250)

        self._status = QLabel(
            f"状态：已随机打乱 {self.DEFAULT_SCRAMBLE_DEPTH} 步 · 等待操作"
        )
        self._status.setObjectName("statusLabel")
        self._status.setWordWrap(True)

        self._ai_button = QPushButton("AI推导")
        self._ai_button.setObjectName("aiButton")
        self._ai_button.clicked.connect(self.toggle_ai)
        self._scramble_depth_combo = QComboBox()
        self._scramble_depth_combo.setObjectName("scrambleDepthCombo")
        self._scramble_depth_combo.setToolTip("选择随机打乱使用的原子动作数量")
        for depth in self.SCRAMBLE_DEPTHS:
            self._scramble_depth_combo.addItem(f"{depth} 步", depth)
        default_index = self._scramble_depth_combo.findData(
            self.DEFAULT_SCRAMBLE_DEPTH
        )
        self._scramble_depth_combo.setCurrentIndex(default_index)
        self._scramble_button = QPushButton("随机打乱")
        self._scramble_button.setObjectName("scrambleButton")
        self._scramble_button.clicked.connect(self.scramble)

        self._build_ui()
        self._apply_style()
        self._update_controls()

    def _build_ui(self) -> None:
        root = QWidget(self)
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(14, 14, 14, 14)
        root_layout.setSpacing(14)

        left = QVBoxLayout()
        left.setSpacing(10)
        left.addWidget(self._view, stretch=1)

        controls = QGridLayout()
        controls.setSpacing(6)
        for index, move in enumerate(MOVE_ORDER):
            button = QPushButton(move.value)
            direction = "逆时针" if move.is_prime else "顺时针"
            shortcut_text = f"Shift+{move.face}" if move.is_prime else move.face
            button.setToolTip(
                f"{move.value}：对应面{direction} 90°（快捷键 {shortcut_text}）"
            )
            button.clicked.connect(lambda checked=False, selected=move: self.user_move(selected))
            self._move_buttons[move] = button
            shortcut = QShortcut(QKeySequence(shortcut_text), self)
            shortcut.activated.connect(lambda selected=move: self.user_move(selected))
            self._shortcuts.append(shortcut)
            controls.addWidget(button, index % 2, index // 2)
        for column in range(6):
            controls.setColumnStretch(column, 1)
        left.addLayout(controls)

        action_row = QHBoxLayout()
        action_row.setSpacing(8)
        action_row.addWidget(self._ai_button, stretch=1)
        depth_label = QLabel("打乱步数")
        action_row.addWidget(depth_label)
        action_row.addWidget(self._scramble_depth_combo)
        action_row.addWidget(self._scramble_button, stretch=1)
        left.addLayout(action_row)
        left.addWidget(self._status)

        right = QVBoxLayout()
        title = QLabel("动作记录")
        title.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        right.addWidget(title)
        right.addWidget(self._chat, stretch=1)
        hint = QLabel("每条记录只在动作实际完成后写入。随机打乱不会写入记录。")
        hint.setWordWrap(True)
        right.addWidget(hint)

        root_layout.addLayout(left, stretch=4)
        root_layout.addLayout(right, stretch=1)
        self.setCentralWidget(root)

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QWidget { background: #101522; color: #edf2f7; }
            QPushButton {
                background: #263449; border: 1px solid #43526b;
                border-radius: 6px; padding: 9px; font-size: 14px;
            }
            QPushButton:hover:enabled { background: #345277; }
            QPushButton:disabled { color: #718096; background: #182131; }
            QPushButton#aiButton { background: #196f55; font-weight: bold; }
            QPushButton#aiButton[stopMode="true"] {
                background: #b42318; border-color: #ef4444;
            }
            QPushButton#aiButton[stopMode="true"]:hover:enabled { background: #dc2626; }
            QPushButton#scrambleButton { background: #7c4d1e; font-weight: bold; }
            QComboBox#scrambleDepthCombo {
                background: #182131; border: 1px solid #43526b;
                border-radius: 5px; padding: 7px 22px 7px 9px;
                min-width: 72px;
            }
            QListWidget#chatList { background: #0b101a; border: 1px solid #344156; }
            QLabel#statusLabel { color: #b9c7db; padding: 4px; }
            QListWidget::item { padding: 5px; }
            """
        )

    def user_move(self, move: Move) -> None:
        if self._ai_running or self._move_busy:
            return
        self._move_busy = True
        self._set_status(f"状态：正在执行用户动作 {move.value}…")
        started = self._view.animate_move(
            move,
            lambda: self._commit_move(move, "用户"),
            duration=self.MOVE_ANIMATION_SECONDS,
        )
        if not started:
            self._move_busy = False
            self._set_status("状态：动画忙碌，动作未执行")
        self._update_controls()

    def _commit_move(self, move: Move, source: str) -> None:
        self._state.apply_move(move)
        self._view.set_state(self._state)
        self._chat.addItem(QListWidgetItem(f"{source}：{move.value}"))
        self._chat.scrollToBottom()
        self._move_busy = False
        if source == "用户":
            self._set_status("状态：用户动作已完成")
        self._update_controls()

    def scramble(self) -> None:
        if self._ai_running or self._move_busy:
            return
        scramble_depth = int(self._scramble_depth_combo.currentData())
        self._chat.clear()
        self._state, scramble = CubeState.scrambled(scramble_depth)
        self._policy_history.clear()
        self._view.set_state(self._state)
        notation = " ".join(move.value for move in scramble.moves)
        self._set_status(f"状态：已随机打乱 {scramble_depth} 步（聊天栏已清空）")
        self._chat.setToolTip(f"本次随机序列（不写入聊天栏）：{notation}")

    def toggle_ai(self) -> None:
        if self._ai_running:
            self.stop_ai()
        else:
            self.start_ai()

    def start_ai(self) -> None:
        if self._ai_running or self._move_busy:
            return
        if self._state.is_solved():
            self._set_status("状态：魔方已经还原，无需执行 AI")
            return
        try:
            self._model.load()
        except ModelLoadError as exc:
            self._set_status(f"状态：{exc}")
            QMessageBox.warning(self, "AI 模型不可用", str(exc))
            return
        self._policy_history.clear()
        self._ai_running = True
        self._ai_stop_requested = False
        self._ai_steps = 0
        self._set_status("状态：AI 开始逐步推导…")
        self._update_controls()
        self._run_ai_step()

    def stop_ai(self) -> None:
        if not self._ai_running or self._ai_stop_requested:
            return
        self._ai_stop_requested = True
        if self._move_busy:
            self._set_status("状态：正在停止 AI，当前原子动作完成后终止…")
            self._update_controls()
        else:
            self._finish_ai(False, f"AI 已停止，共执行 {self._ai_steps} 步")

    def _run_ai_step(self) -> None:
        if not self._ai_running:
            return
        if self._ai_stop_requested:
            self._finish_ai(False, f"AI 已停止，共执行 {self._ai_steps} 步")
            return
        if self._state.is_solved():
            self._finish_ai(True, "AI 已完成复原")
            return
        if self._ai_steps >= self.MAX_AI_STEPS:
            self._finish_ai(False, f"AI 达到最大步数 {self.MAX_AI_STEPS}，仍未还原")
            return
        try:
            move = self._model.predict(self._state, self._policy_history)
        except ModelLoadError as exc:
            self._finish_ai(False, str(exc))
            return
        self._move_busy = True
        self._set_status(f"状态：AI 第 {self._ai_steps + 1} 步，执行 {move.value}…")
        started = self._view.animate_move(
            move,
            lambda: self._commit_ai_move(move),
            duration=self.MOVE_ANIMATION_SECONDS,
        )
        if not started:
            self._finish_ai(False, "动画启动失败，AI 已停止")

    def _commit_ai_move(self, move: Move) -> None:
        self._state.apply_move(move)
        self._policy_history.append(move)
        self._view.set_state(self._state)
        self._chat.addItem(QListWidgetItem(f"AI：{move.value}"))
        self._chat.scrollToBottom()
        self._move_busy = False
        self._ai_steps += 1
        if self._state.is_solved():
            self._finish_ai(True, f"AI 已完成复原，共执行 {self._ai_steps} 步")
        elif self._ai_stop_requested:
            self._finish_ai(False, f"AI 已停止，共执行 {self._ai_steps} 步")
        else:
            # The previous animation occupied exactly 0.5 seconds.  The next
            # prediction starts only after its completion callback.
            QTimer.singleShot(0, self._run_ai_step)

    def _finish_ai(self, success: bool, message: str) -> None:
        self._ai_running = False
        self._ai_stop_requested = False
        self._move_busy = False
        self._set_status(f"状态：{message}")
        self._update_controls()

    def _set_status(self, text: str) -> None:
        self._status.setText(text)

    def _update_controls(self) -> None:
        enabled = not self._ai_running and not self._move_busy
        for button in self._move_buttons.values():
            button.setEnabled(enabled)
        self._scramble_button.setEnabled(enabled)
        self._scramble_depth_combo.setEnabled(enabled)
        self._ai_button.setText("停止" if self._ai_running else "AI推导")
        stop_mode = self._ai_running
        if self._ai_button.property("stopMode") != stop_mode:
            self._ai_button.setProperty("stopMode", stop_mode)
            self._ai_button.style().unpolish(self._ai_button)
            self._ai_button.style().polish(self._ai_button)
        self._ai_button.setEnabled(
            enabled or (self._ai_running and not self._ai_stop_requested)
        )

    def keyPressEvent(self, event: QKeyEvent) -> None:
        key = event.text().upper()
        if key in {"U", "F", "R", "L", "B", "D"}:
            is_prime = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
            self.user_move(Move(f"{key}'" if is_prime else key))
            event.accept()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event) -> None:
        self._ai_running = False
        self._view.stop_animation()
        event.accept()
