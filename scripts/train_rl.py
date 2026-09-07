"""Train a 12-action DQN with progressive scramble-depth curriculum.

Each stage trains on a mixture of depths 1..N and is validated on exact depth
N. A qualified checkpoint is saved only after reaching the requested success
rate; otherwise training stops before advancing to the next stage.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from magic_cube.core.moves import MOVE_ORDER, MOVE_TO_INDEX, coerce_move
from magic_cube.rl.action_rules import DEFAULT_REWARD_CONFIG
from magic_cube.rl.environment import CubeEnv, recommended_max_steps
from magic_cube.rl.evaluation import EvaluationResult, evaluate_model
from magic_cube.rl.observation import MODEL_OBSERVATION_SIZE, MODEL_SCHEMA_VERSION


def parse_depths(value: str) -> tuple[int, ...]:
    try:
        depths = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("深度必须是逗号分隔的正整数") from exc
    if not depths or any(depth <= 0 for depth in depths):
        raise argparse.ArgumentTypeError("深度必须是逗号分隔的正整数")
    if tuple(sorted(set(depths))) != depths:
        raise argparse.ArgumentTypeError("深度必须严格递增且不能重复")
    return depths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--depths",
        type=parse_depths,
        default=parse_depths("1,2,3,4,5"),
        help="课程深度，例如 1,2,3,4,5 或 1,3,5,10,20",
    )
    parser.add_argument("--target-success-rate", type=float, default=0.80)
    parser.add_argument("--max-redundancy-rate", type=float, default=0.05)
    parser.add_argument("--eval-episodes", type=int, default=200)
    parser.add_argument(
        "--initial-train-steps-per-round",
        type=int,
        default=10_000,
        help="首次验证前的训练步数，也是接近晋级线时的验证间隔",
    )
    parser.add_argument(
        "--train-steps-per-round",
        type=int,
        default=100_000,
        help="动态验证间隔的上限",
    )
    parser.add_argument(
        "--eval-interval-growth",
        type=float,
        default=2.0,
        help="远离晋级线时验证间隔的增长倍数",
    )
    parser.add_argument(
        "--near-target-ratio",
        type=float,
        default=0.75,
        help="达到目标成功率的此比例后，恢复最短验证间隔",
    )
    parser.add_argument("--max-timesteps-per-depth", type=int, default=5_000_000)
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=None,
        help="每回合最大步数；默认按深度使用 2*N+4",
    )
    parser.add_argument(
        "--current-depth-probability",
        type=float,
        default=0.75,
        help="训练时采样当前课程深度的概率",
    )
    parser.add_argument(
        "--expert-episodes",
        type=int,
        default=2_000,
        help="经验池为空时，用已知逆序解预填充的专家回合数",
    )
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="不跳过已经达到目标成功率的阶段模型",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not 0.0 < args.target_success_rate <= 1.0:
        raise SystemExit("--target-success-rate 必须在 (0, 1] 范围内")
    if not 0.0 <= args.max_redundancy_rate <= 1.0:
        raise SystemExit("--max-redundancy-rate 必须在 [0, 1] 范围内")
    for name in (
        "eval_episodes",
        "initial_train_steps_per_round",
        "train_steps_per_round",
        "max_timesteps_per_depth",
    ):
        if getattr(args, name) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} 必须大于 0")
    if args.max_episode_steps is not None and args.max_episode_steps <= 0:
        raise SystemExit("--max-episode-steps 必须大于 0")
    if not 0.0 <= args.current_depth_probability <= 1.0:
        raise SystemExit("--current-depth-probability 必须在 [0, 1] 范围内")
    if args.expert_episodes < 0:
        raise SystemExit("--expert-episodes 不能小于 0")
    if args.eval_interval_growth <= 1.0:
        raise SystemExit("--eval-interval-growth 必须大于 1")
    if not 0.0 < args.near_target_ratio <= 1.0:
        raise SystemExit("--near-target-ratio 必须在 (0, 1] 范围内")


def create_model(env: CubeEnv, seed: int):
    from stable_baselines3 import DQN

    return DQN(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=1e-4,
        buffer_size=100_000,
        learning_starts=10_000,
        batch_size=256,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=2_000,
        exploration_fraction=0.25,
        exploration_final_eps=0.05,
        gamma=0.99,
        policy_kwargs={"net_arch": [512, 512, 256]},
        seed=seed,
    )


def replay_buffer_path(checkpoint: Path) -> Path:
    """Return the ignored sidecar path used for an SB3 replay buffer."""

    return checkpoint.with_suffix(".replay.pkl")


def load_checkpoint(model_class: Any, checkpoint: Path, *, env: CubeEnv | None = None):
    """Load a model and restore its replay buffer when a sidecar exists."""

    model = model_class.load(str(checkpoint), env=env, device="auto")
    buffer_path = replay_buffer_path(checkpoint)
    if buffer_path.exists():
        model.load_replay_buffer(str(buffer_path))
        print(f"恢复经验池：{buffer_path}")
    return model


def save_training_checkpoint(model: Any, checkpoint: Path) -> None:
    """Save model weights and the replay buffer needed for true continuation."""

    model.save(str(checkpoint.with_suffix("")))
    model.save_replay_buffer(str(replay_buffer_path(checkpoint)))


def prefill_expert_replay(model: Any, env: CubeEnv, episodes: int, seed: int) -> int:
    """Add known inverse-scramble trajectories to an empty replay buffer."""

    replay_buffer = getattr(model, "replay_buffer", None)
    if replay_buffer is None or episodes <= 0 or replay_buffer.size() != 0:
        return 0

    added = 0
    for episode in range(episodes):
        observation, info = env.reset(seed=seed + episode)
        scramble = tuple(coerce_move(move) for move in info["scramble"])
        solution = tuple(move.inverse for move in reversed(scramble))
        for move in solution:
            action = MOVE_TO_INDEX[move]
            next_observation, reward, terminated, truncated, _step_info = env.step(
                action
            )
            replay_buffer.add(
                np.asarray(observation, dtype=np.float32)[None, :],
                np.asarray(next_observation, dtype=np.float32)[None, :],
                np.asarray([action], dtype=np.int64),
                np.asarray([reward], dtype=np.float32),
                np.asarray([terminated or truncated], dtype=bool),
                [{"TimeLimit.truncated": truncated}],
            )
            added += 1
            observation = next_observation
            if terminated or truncated:
                break
    return added


def find_previous_checkpoint(models_dir: Path, before_depth: int) -> Path | None:
    pattern = re.compile(r"cube_dqn_depth_(\d+)\.zip$")
    candidates: list[tuple[int, Path]] = []
    for path in models_dir.glob("cube_dqn_depth_*.zip"):
        match = pattern.match(path.name)
        if match and int(match.group(1)) < before_depth:
            candidates.append((int(match.group(1)), path))
    return max(candidates, default=(0, None), key=lambda item: item[0])[1]


def load_progress(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"stages": {}}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"stages": {}}


def save_progress(
    path: Path,
    progress: dict[str, Any],
    *,
    depth: int,
    stage_timesteps: int,
    result: EvaluationResult,
    qualified: bool,
) -> None:
    progress.update(
        {
            "strategy": "curriculum_dqn",
            "model_schema_version": MODEL_SCHEMA_VERSION,
            "action_order": [move.value for move in MOVE_ORDER],
            "observation_size": MODEL_OBSERVATION_SIZE,
            "reward_config": DEFAULT_REWARD_CONFIG.to_dict(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    progress.setdefault("stages", {})[str(depth)] = {
        "stage_timesteps": stage_timesteps,
        "qualified": qualified,
        **result.to_dict(),
    }
    path.write_text(
        json.dumps(progress, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def print_evaluation(depth: int, stage_timesteps: int, result: EvaluationResult) -> None:
    print(
        f"深度 {depth} | 本阶段步数 {stage_timesteps} | "
        f"成功率 {result.success_rate:.2%} ({result.successes}/{result.episodes}) | "
        f"成功样本平均动作数 {result.average_steps:.2f} | "
        f"冗余率 {result.redundancy_rate:.2%} "
        f"({result.redundant_actions}/{result.total_actions})"
    )


def is_qualified(result: EvaluationResult, args: argparse.Namespace) -> bool:
    return (
        result.success_rate >= args.target_success_rate
        and result.redundancy_rate <= args.max_redundancy_rate
    )


def next_round_steps(
    current_steps: int,
    result: EvaluationResult,
    *,
    initial_steps: int,
    maximum_steps: int,
    target_success_rate: float,
    near_target_ratio: float,
    growth_factor: float,
) -> int:
    """Choose the next training interval from the latest validation result."""

    near_target = result.success_rate >= target_success_rate * near_target_ratio
    if near_target:
        return min(initial_steps, maximum_steps)
    grown_steps = math.ceil(current_steps * growth_factor)
    return min(maximum_steps, max(initial_steps, grown_steps))


def main() -> None:
    args = parse_args()
    validate_args(args)
    try:
        from stable_baselines3 import DQN
    except ImportError as exc:
        raise SystemExit("Stable-Baselines3 未安装，请先执行 uv sync。") from exc

    args.models_dir.mkdir(parents=True, exist_ok=True)
    progress_path = args.models_dir / "curriculum_progress.json"
    progress = load_progress(progress_path)
    progress["qualification"] = {
        "minimum_success_rate": args.target_success_rate,
        "maximum_redundancy_rate": args.max_redundancy_rate,
        "evaluation_episodes": args.eval_episodes,
    }
    progress["training_config"] = {
        "current_depth_probability": args.current_depth_probability,
        "expert_episodes": args.expert_episodes,
        "max_episode_steps": args.max_episode_steps,
        "horizon_strategy": "2*N+4" if args.max_episode_steps is None else "fixed",
    }
    model = None

    if args.resume_from is not None:
        if not args.resume_from.exists():
            raise SystemExit(f"续训模型不存在：{args.resume_from}")
        model = load_checkpoint(DQN, args.resume_from)
        print(f"从模型继续训练：{args.resume_from}")

    for depth in args.depths:
        max_episode_steps = (
            recommended_max_steps(depth)
            if args.max_episode_steps is None
            else args.max_episode_steps
        )
        train_env = CubeEnv(
            scramble_depth=depth,
            min_scramble_depth=1,
            max_steps=max_episode_steps,
            focus_depth=depth,
            focus_depth_probability=args.current_depth_probability,
        )
        qualified_path = args.models_dir / f"cube_dqn_depth_{depth}.zip"
        latest_path = args.models_dir / f"cube_dqn_depth_{depth}_latest.zip"
        previous_stage_steps = int(
            progress.get("stages", {}).get(str(depth), {}).get("stage_timesteps", 0)
        )
        initial_result: EvaluationResult | None = None

        if qualified_path.exists() and not args.force_retrain:
            candidate = load_checkpoint(DQN, qualified_path, env=train_env)
            existing_result = evaluate_model(
                candidate,
                scramble_depth=depth,
                episodes=args.eval_episodes,
                max_steps=max_episode_steps,
                seed=args.seed + depth * 10_000,
            )
            print_evaluation(depth, 0, existing_result)
            if is_qualified(existing_result, args):
                model = candidate
                model.save(str(args.models_dir / "cube_solver"))
                save_progress(
                    progress_path,
                    progress,
                    depth=depth,
                    stage_timesteps=previous_stage_steps,
                    result=existing_result,
                    qualified=True,
                )
                train_env.close()
                print(f"深度 {depth} 已达标，跳过训练。")
                continue
            model = candidate
            initial_result = existing_result

        if latest_path.exists() and not args.force_retrain:
            model = load_checkpoint(DQN, latest_path, env=train_env)
            initial_result = None
            print(f"自动继续未达标阶段：{latest_path}")

        if model is None:
            previous = find_previous_checkpoint(args.models_dir, depth)
            if previous is not None:
                model = load_checkpoint(DQN, previous, env=train_env)
                print(f"自动继承上一阶段模型：{previous}")
            else:
                model = create_model(train_env, args.seed)
                print("创建新的 DQN 模型。")
        else:
            model.set_env(train_env)

        stage_timesteps = 0
        if initial_result is None:
            initial_result = evaluate_model(
                model,
                scramble_depth=depth,
                episodes=args.eval_episodes,
                max_steps=max_episode_steps,
                seed=args.seed + depth * 10_000,
            )
            print_evaluation(depth, stage_timesteps, initial_result)

        qualified = is_qualified(initial_result, args)
        save_progress(
            progress_path,
            progress,
            depth=depth,
            stage_timesteps=previous_stage_steps,
            result=initial_result,
            qualified=qualified,
        )
        if qualified:
            save_training_checkpoint(model, qualified_path)
            model.save(str(args.models_dir / "cube_solver"))
            train_env.close()
            print(f"深度 {depth} 无需追加训练，已保存：{qualified_path}")
            continue

        expert_transitions = prefill_expert_replay(
            model,
            train_env,
            args.expert_episodes,
            seed=args.seed + depth * 100_000,
        )
        if expert_transitions:
            print(f"专家轨迹预填充：{expert_transitions} 条经验。")

        scheduled_round_steps = min(
            args.initial_train_steps_per_round,
            args.train_steps_per_round,
        )
        while stage_timesteps < args.max_timesteps_per_depth:
            round_steps = min(
                scheduled_round_steps,
                args.max_timesteps_per_depth - stage_timesteps,
            )
            model.learn(total_timesteps=round_steps, reset_num_timesteps=False)
            stage_timesteps += round_steps
            result = evaluate_model(
                model,
                scramble_depth=depth,
                episodes=args.eval_episodes,
                max_steps=max_episode_steps,
                seed=args.seed + depth * 10_000,
            )
            print_evaluation(depth, stage_timesteps, result)
            save_training_checkpoint(model, latest_path)
            qualified = is_qualified(result, args)
            total_stage_timesteps = previous_stage_steps + stage_timesteps
            save_progress(
                progress_path,
                progress,
                depth=depth,
                stage_timesteps=total_stage_timesteps,
                result=result,
                qualified=qualified,
            )
            if qualified:
                save_training_checkpoint(model, qualified_path)
                model.save(str(args.models_dir / "cube_solver"))
                print(f"深度 {depth} 达到目标，已保存：{qualified_path}")
                break
            scheduled_round_steps = next_round_steps(
                scheduled_round_steps,
                result,
                initial_steps=args.initial_train_steps_per_round,
                maximum_steps=args.train_steps_per_round,
                target_success_rate=args.target_success_rate,
                near_target_ratio=args.near_target_ratio,
                growth_factor=args.eval_interval_growth,
            )
            print(f"下一次验证将在追加训练 {scheduled_round_steps} 步后进行。")

        train_env.close()
        if not qualified:
            raise SystemExit(
                f"深度 {depth} 在 {stage_timesteps} 步内未达到 "
                f"成功率≥{args.target_success_rate:.0%} 且冗余率≤"
                f"{args.max_redundancy_rate:.0%}；已保存 latest 检查点，"
                "课程未继续升级。"
            )

    print(f"课程完成，当前最高合格模型：{args.models_dir / 'cube_solver.zip'}")


if __name__ == "__main__":
    main()
