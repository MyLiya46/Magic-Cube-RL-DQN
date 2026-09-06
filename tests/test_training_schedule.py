from magic_cube.rl.evaluation import EvaluationResult
from scripts.train_rl import next_round_steps


def evaluation(successes: int, episodes: int = 100) -> EvaluationResult:
    return EvaluationResult(
        episodes=episodes,
        successes=successes,
        solved_steps=(),
        total_actions=100,
        redundant_actions=0,
    )


def test_validation_interval_grows_when_far_from_target():
    result = evaluation(20)

    assert next_round_steps(
        10_000,
        result,
        initial_steps=10_000,
        maximum_steps=100_000,
        target_success_rate=0.80,
        near_target_ratio=0.75,
        growth_factor=2.0,
    ) == 20_000


def test_validation_interval_is_capped():
    result = evaluation(20)

    assert next_round_steps(
        80_000,
        result,
        initial_steps=10_000,
        maximum_steps=100_000,
        target_success_rate=0.80,
        near_target_ratio=0.75,
        growth_factor=2.0,
    ) == 100_000


def test_validation_interval_returns_to_minimum_near_target():
    result = evaluation(70)

    assert next_round_steps(
        80_000,
        result,
        initial_steps=10_000,
        maximum_steps=100_000,
        target_success_rate=0.80,
        near_target_ratio=0.75,
        growth_factor=2.0,
    ) == 10_000
