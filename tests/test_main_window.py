import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QWidget

from magic_cube.core.moves import Move
from magic_cube.core.state import CubeState
from magic_cube.rl.observation import MODEL_OBSERVATION_SIZE, encode_model_observation
from magic_cube.ui import main_window as main_window_module


class ImmediateCubeView(QWidget):
    def __init__(self, state: CubeState) -> None:
        super().__init__()
        self.state = state.copy()

    def set_state(self, state: CubeState) -> None:
        self.state = state.copy()

    def animate_move(self, move: Move, callback, *, duration: float) -> bool:
        assert duration == 0.5
        callback()
        return True

    def stop_animation(self) -> None:
        pass


class DeferredCubeView(ImmediateCubeView):
    def __init__(self, state: CubeState) -> None:
        super().__init__(state)
        self.pending_callback = None

    def animate_move(self, move: Move, callback, *, duration: float) -> bool:
        assert duration == 0.5
        self.pending_callback = callback
        return True

    def finish_animation(self) -> None:
        assert self.pending_callback is not None
        callback, self.pending_callback = self.pending_callback, None
        callback()


class RecordingModel:
    def __init__(self, move: Move) -> None:
        self.move = move
        self.observation_shapes: list[tuple[int, ...]] = []
        self.histories: list[tuple[Move, ...]] = []

    def load(self) -> None:
        pass

    def predict(self, state: CubeState, history=()) -> Move:
        history_snapshot = tuple(history)
        self.histories.append(history_snapshot)
        self.observation_shapes.append(
            encode_model_observation(state, history_snapshot).shape
        )
        return self.move


def make_window(monkeypatch, tmp_path, view_type=ImmediateCubeView):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(main_window_module, "CubeView", view_type)
    window = main_window_module.MainWindow(project_root=tmp_path)
    return app, window


def test_ai_history_starts_empty_after_user_moves(monkeypatch, tmp_path):
    app, window = make_window(monkeypatch, tmp_path)
    model = RecordingModel(Move.U_PRIME)
    window._model = model
    window._state = CubeState.solved()
    window._view.set_state(window._state)

    window.user_move(Move.U)
    assert window._chat.count() == 1
    assert tuple(window._policy_history) == ()

    window.start_ai()

    assert model.histories == [()]
    assert model.observation_shapes == [(MODEL_OBSERVATION_SIZE,)]
    assert window._state.is_solved()
    assert window._chat.count() == 2
    assert window._ai_button.text() == "AI推导"
    window.close()
    app.processEvents()


def test_scramble_clears_chat_and_policy_history(monkeypatch, tmp_path):
    app, window = make_window(monkeypatch, tmp_path)
    window._policy_history.append(Move.F)
    window._chat.addItem("AI：F")
    window._scramble_depth_combo.setCurrentIndex(
        window._scramble_depth_combo.findData(1)
    )

    window.scramble()

    assert window._chat.count() == 0
    assert tuple(window._policy_history) == ()
    assert not window._state.is_solved()
    window.close()
    app.processEvents()


def test_ai_button_enters_red_stop_mode_until_current_move_finishes(
    monkeypatch, tmp_path
):
    app, window = make_window(monkeypatch, tmp_path, DeferredCubeView)
    model = RecordingModel(Move.R_PRIME)
    window._model = model
    window._state = CubeState.solved()
    window._state.apply_move(Move.R)
    window._view.set_state(window._state)

    window.start_ai()

    assert window._ai_button.text() == "停止"
    assert window._ai_button.property("stopMode") is True
    assert window._ai_button.isEnabled()

    window.stop_ai()
    window._view.finish_animation()

    assert window._ai_button.text() == "AI推导"
    assert window._ai_button.property("stopMode") is False
    assert window._state.is_solved()
    window.close()
    app.processEvents()
