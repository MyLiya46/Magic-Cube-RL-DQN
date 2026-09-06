import numpy as np

from magic_cube.core.moves import Move
from magic_cube.core.state import CubeState
from magic_cube.rl.action_rules import RedundancyKind, classify_redundancy
from magic_cube.rl.observation import (
    HISTORY_LENGTH,
    HISTORY_TOKEN_COUNT,
    MODEL_OBSERVATION_SIZE,
    NO_MOVE_INDEX,
    ActionHistory,
    encode_model_observation,
)


def test_redundancy_rules_keep_half_turns_but_reject_reducible_sequences():
    assert classify_redundancy([Move.U], Move.U) is None
    assert (
        classify_redundancy([Move.U], Move.U_PRIME)
        is RedundancyKind.IMMEDIATE_INVERSE
    )
    assert (
        classify_redundancy([Move.U, Move.U], Move.U)
        is RedundancyKind.THIRD_SAME_FACE
    )
    assert (
        classify_redundancy([Move.U, Move.U, Move.U], Move.U)
        is RedundancyKind.FULL_FOUR_TURN_CYCLE
    )


def test_model_observation_contains_left_padded_action_history():
    history = ActionHistory([Move.U, Move.F_PRIME])
    observation = encode_model_observation(CubeState.solved(), history)
    assert observation.shape == (MODEL_OBSERVATION_SIZE,)
    encoded_history = observation[-HISTORY_LENGTH * HISTORY_TOKEN_COUNT :].reshape(
        HISTORY_LENGTH, HISTORY_TOKEN_COUNT
    )
    assert np.argmax(encoded_history, axis=1).tolist() == [
        NO_MOVE_INDEX,
        0,
        3,
    ]
    assert np.allclose(encoded_history.sum(axis=1), 1.0)


def test_action_history_keeps_only_the_latest_three_moves():
    history = ActionHistory([Move.U, Move.F, Move.R, Move.L])
    assert history.moves == (Move.F, Move.R, Move.L)
