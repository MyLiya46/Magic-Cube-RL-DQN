# 本地强化学习模型

GUI 默认加载：

```text
models/cube_solver.zip
```

当前目录不包含伪造的模型权重。准备好 Python 环境后，可使用课程学习训练一个与 GUI 动作/观测接口兼容的 DQN 模型：

```bash
uv run python scripts/train_rl.py --depths 1,2,3,4,5 --target-success-rate 0.80 --max-redundancy-rate 0.05
```

训练脚本使用 363 维观测：324 维魔方 one-hot 状态，加最近 3 个动作的 39 维 one-hot 历史。动作空间为 `[U, U', F, F', R, R', L, L', B, B', D, D']`。深度 `N` 阶段混合训练 `1..N` 步状态，并在独立的精确 `N` 步状态上验证。

验证采用自适应间隔：阶段开始时先验证一次，随后从每 1 万训练步开始；远离晋级线时逐步扩大到最多 10 万步，接近晋级线时恢复为每 1 万步验证。因此简单难度不必固定等待 10 万步。

只有验证成功率达到 80% 且冗余动作率不超过 5% 才会生成合格检查点：

```text
models/cube_dqn_depth_1.zip
models/cube_dqn_depth_2.zip
models/cube_dqn_depth_3.zip
...
models/cube_solver.zip
models/curriculum_progress.json
```

`cube_solver.zip` 始终指向当前最高深度的合格模型。未达标阶段只保存 `_latest.zip`，课程不会继续升级。再次运行相同命令会验证并跳过已经达标的阶段，也可以通过 `--resume-from` 显式指定续训模型。

模型缺失、观测维度不是 363 或动作空间不是 `Discrete(12)` 时，GUI 会显示错误并禁用 AI，不会使用随机动作冒充模型推理。所有旧观测接口的模型均不兼容。
