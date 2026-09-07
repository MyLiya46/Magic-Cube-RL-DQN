import numpy as np

from magic_cube.rl.environment import CubeEnv, recommended_max_steps
from magic_cube.rl.observation import (
    CUBE_OBSERVATION_SIZE,
    HISTORY_LENGTH,
    HISTORY_TOKEN_COUNT,
    MODEL_OBSERVATION_SIZE,
    NO_MOVE_INDEX,
)


def test_environment_contract_and_step():
    env = CubeEnv(scramble_depth=2, max_steps=20)
    observation, info = env.reset(seed=7)
    assert observation.shape == (MODEL_OBSERVATION_SIZE,)
    assert observation.dtype == np.float32
    cube_observation = observation[:CUBE_OBSERVATION_SIZE].reshape(54, 6)
    history_observation = observation[CUBE_OBSERVATION_SIZE:].reshape(
        HISTORY_LENGTH, HISTORY_TOKEN_COUNT
    )
    assert np.allclose(cube_observation.sum(axis=1), 1.0)
    assert np.argmax(history_observation, axis=1).tolist() == [
        NO_MOVE_INDEX,
        NO_MOVE_INDEX,
        NO_MOVE_INDEX,
    ]
    assert env.action_space.n == 12
    assert "scramble" in info
    assert info["scramble_depth"] == 2

    next_observation, reward, terminated, truncated, step_info = env.step(0)
    assert next_observation.shape == (MODEL_OBSERVATION_SIZE,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert step_info["move"] == "U"
    env.close()


def test_environment_rejects_invalid_actions():
    env = CubeEnv()
    env.reset(seed=1)
    try:
        env.step(12)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid action should raise ValueError")


def test_curriculum_environment_samples_within_depth_range():
    env = CubeEnv(scramble_depth=5, min_scramble_depth=1)
    observed_depths = {
        env.reset(seed=seed)[1]["scramble_depth"] for seed in range(30)
    }
    assert observed_depths <= {1, 2, 3, 4, 5}
    assert len(observed_depths) > 1


def test_curriculum_environment_can_focus_the_current_depth():
    env = CubeEnv(
        scramble_depth=5,
        min_scramble_depth=1,
        focus_depth=5,
        focus_depth_probability=1.0,
    )
    observed_depths = {env.reset(seed=seed)[1]["scramble_depth"] for seed in range(10)}
    assert observed_depths == {5}


def test_default_episode_horizon_scales_with_scramble_depth():
    assert recommended_max_steps(0) == 4
    assert recommended_max_steps(6) == 16


def test_environment_reports_and_penalizes_redundant_actions():
    env = CubeEnv(scramble_depth=5, max_steps=20)
    env.reset(seed=11)
    env.step(0)
    _, _, _, _, inverse_info = env.step(1)
    assert inverse_info["redundancy_kind"] == "immediate_inverse"
    assert inverse_info["redundancy_penalty"] == -0.10

    env.reset(seed=11)
    env.step(0)
    env.step(0)
    _, _, _, _, third_info = env.step(0)
    assert third_info["redundancy_kind"] == "third_same_face"
    env.close()
