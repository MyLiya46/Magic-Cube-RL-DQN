# Magic Cube AI

一个使用主流 12 动作空间的三阶魔方 3D 演示程序。用户可以手动执行六个面的顺时针和逆时针 90°动作，也可以加载本地 Stable-Baselines3 DQN 模型逐步复原。

## 快速开始

需要 Python 3.11+ 和 `uv`：

```bash
uv sync
uv run pytest
uv run python -m magic_cube
```

也可以使用项目脚本：

```bash
uv run magic-cube
```

## 操作

动作空间固定为：`U、U'、F、F'、R、R'、L、L'、B、B'、D、D'`。无撇号表示从对应面观察时顺时针旋转 90°，撇号表示逆时针旋转 90°。普通动作使用对应字母快捷键，逆时针动作使用 `Shift+字母`。不把 180°旋转作为独立动作。

## 使用强化学习模型进行AI推导

GUI 默认从 `models/cube_solver.zip` 加载 DQN 模型。当前项目不包含伪造的权重文件。推荐使用课程学习训练：

```bash
uv run python scripts/train_rl.py --depths 1,2,3,4,5 --target-success-rate 0.80 --max-redundancy-rate 0.05
uv run python scripts/evaluate_rl.py --checkpoint models/cube_solver.zip --episodes 100 --scramble-depth 5
```

继续挑战 10 步和 20 步：

```bash
uv run python scripts/train_rl.py --depths 10,20 --target-success-rate 0.80 --max-redundancy-rate 0.05
```

暴力

```bash
uv run python scripts/train_rl.py --depths 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20 --target-success-rate 0.90 --max-redundancy-rate 0.05
```

训练策略：

- 深度 `N` 阶段从 `1..N` 步打乱中采样，保留之前深度的能力。
- 进入每个难度时先立即验证；需要训练时首次在 1 万环境步后验证。
- 若成功率远离晋级线，验证间隔按 1 万、2 万、4 万、8 万逐步增大，最高 10 万；达到目标成功率的 75% 后恢复为每 1 万步验证。
- 目标深度验证成功率达到 80%，且冗余动作率不超过 5%，才进入下一阶段。
- 合格模型保存为 `models/cube_dqn_depth_N.zip`，并同步更新 `models/cube_solver.zip`。
- 未达标模型保存为 `models/cube_dqn_depth_N_latest.zip`，下次可以继续训练。
- 进度和验证指标写入 `models/curriculum_progress.json`。

模型接口要求：

- 动作空间：`Discrete(12)`。
- 动作顺序：`[U, U', F, F', R, R', L, L', B, B', D, D']`。
- 观测：324 维魔方 one-hot 状态，加最近 3 个动作的 39 维 one-hot 历史，共 363 维。

动作历史让冗余惩罚保持马尔可夫性质。每次开始 AI 推导都开启一个新的策略回合，历史只包含该回合内 AI 已执行的动作；用户操作仍保留在右侧记录中，但不会污染模型上下文。统一奖励策略如下：

- 每个未完成动作：`-0.01`。
- 立即执行上一步的逆动作：额外 `-0.10`。
- 同一面连续操作第三次：额外 `-0.08`；连续两次仍被允许，用于表达 180°旋转。
- 连续四次完全相同动作：额外 `-0.15`。
- 贴纸位置改善：每增加一个正确位置奖励 `+0.01`，减少则对称扣分。
- 完全复原：`+1.0`。

应用启动时不会自动训练模型。模型缺失、加载失败、观测维度不匹配或动作空间不匹配时，`AI推导` 会显示具体错误并停止，不会用随机动作代替模型。

旧的 6 动作、54 维或 324 维模型与当前 363 维接口不兼容，必须重新训练。课程晋级同时检查成功率和冗余率；基础 DQN 对 10 步、20 步状态仍可能无法达标，脚本不会把失败模型标记为合格。

## 项目结构

```text
magic_cube/
  core/       魔方状态、动作和几何映射
  rl/         Gymnasium 环境和模型适配器
  ui/         PySide6 + OpenGL 界面
scripts/      RL 训练和评估脚本
models/       本地模型说明
tests/        核心状态和环境测试
```
