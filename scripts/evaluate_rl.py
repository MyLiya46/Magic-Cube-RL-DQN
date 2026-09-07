"""Evaluate a local 12-action DQN checkpoint on exact-depth states."""

from __future__ import annotations

import argparse
from pathlib import Path

from magic_cube.rl.environment import recommended_max_steps
from magic_cube.rl.evaluation import evaluate_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("models/cube_solver.zip"))
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--scramble-depth", type=int, default=5)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="每回合最大步数；默认按深度使用 2*N+4",
    )
    parser.add_argument("--seed", type=int, default=10_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from stable_baselines3 import DQN
    except ImportError as exc:
        raise SystemExit("Stable-Baselines3 未安装，请先执行 uv sync。") from exc

    if args.episodes <= 0:
        raise SystemExit("--episodes 必须大于 0")
    if args.max_steps is not None and args.max_steps <= 0:
        raise SystemExit("--max-steps 必须大于 0")
    model = DQN.load(str(args.checkpoint), device="auto")
    max_steps = (
        recommended_max_steps(args.scramble_depth)
        if args.max_steps is None
        else args.max_steps
    )
    result = evaluate_model(
        model,
        scramble_depth=args.scramble_depth,
        episodes=args.episodes,
        max_steps=max_steps,
        seed=args.seed,
    )
    print(f"episodes={result.episodes}")
    print(f"successes={result.successes}")
    print(f"success_rate={result.success_rate:.2%}")
    print(f"average_steps_on_success={result.average_steps:.2f}")
    print(f"redundant_actions={result.redundant_actions}/{result.total_actions}")
    print(f"redundancy_rate={result.redundancy_rate:.2%}")


if __name__ == "__main__":
    main()
