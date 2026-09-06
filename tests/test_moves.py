from magic_cube.core.moves import MOVE_ORDER, Move
from magic_cube.core.state import CubeState


def test_solved_state_has_nine_stickers_per_colour():
    state = CubeState.solved()
    assert state.is_solved()
    assert all(state.facelets.count(colour) == 9 for colour in range(6))
    observation = state.observation().reshape(54, 6)
    assert observation.shape == (54, 6)
    assert all(row.sum() == 1.0 for row in observation)


def test_every_primitive_move_four_times_returns_to_solved():
    for move in MOVE_ORDER:
        state = CubeState.solved()
        for _ in range(4):
            state.apply_move(move)
        assert state.is_solved(), move


def test_each_move_changes_a_solved_cube():
    for move in MOVE_ORDER:
        state = CubeState.solved()
        state.apply_move(move)
        assert not state.is_solved(), move
        assert all(state.facelets.count(colour) == 9 for colour in range(6))


def test_every_move_is_undone_by_its_inverse():
    for move in MOVE_ORDER:
        state = CubeState.solved()
        state.apply_move(move)
        state.apply_move(move.inverse)
        assert state.is_solved(), move


def test_string_and_integer_actions_are_supported_but_invalid_actions_fail():
    state = CubeState.solved()
    state.apply_move("u")
    for _ in range(3):
        state.apply_move(0)
    assert state.is_solved()
    try:
        state.apply_move("U2")
    except ValueError:
        pass
    else:
        raise AssertionError("half turns must not be accepted")


def test_scramble_uses_only_the_twelve_primitive_actions():
    state, scramble = CubeState.scrambled(20)
    assert len(scramble.moves) == 20
    assert all(move in MOVE_ORDER for move in scramble.moves)
    assert all(
        current.face != following.face
        for current, following in zip(scramble.moves, scramble.moves[1:])
    )
    assert not state.is_solved()


def test_face_grid_is_three_by_three():
    state = CubeState.solved()
    for face in ("U", "F", "R", "L", "B", "D"):
        grid = state.face_grid(face)
        assert len(grid) == 3
        assert all(len(row) == 3 for row in grid)
        assert {value for row in grid for value in row} == {list(("U", "F", "R", "L", "B", "D")).index(face)}
