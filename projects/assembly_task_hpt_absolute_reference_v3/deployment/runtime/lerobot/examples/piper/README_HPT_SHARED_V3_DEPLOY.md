# HPT Shared-Action V3 真机部署说明

本文只适用于 `hpt-shared-v3-15k`。所有命令均在机器人主机
`/mnt/wd10/kerun/lerobot` 下运行；`/mnt/wd10/tongxi` 只提供现有 Python
环境，禁止修改。

## 1. 部署契约

- checkpoint：`/mnt/wd10/kerun/checkpoints/hpt_shared_action_v3_best_step15000.ckpt`
- SHA-256：`04acf3339fe53b22dd59a491e4c6ab5f5e325689c217a0de646c651db7fb9de2`
- 加载对象：训练阶段完整 Lightning `ModelWrapper`
- 推理分支：V3 `SharedActionPolicyHPTV3.predict_all()` 的
  `actions_joints`
- 输入：单帧全局 RGB、单帧腕部 RGB、当前 7D PIPER proprio
- 模型输出：原生 `30 × 7`
  - 前 6 维：六关节绝对角，单位 rad
  - 第 7 维：二值夹爪状态，`0=close`、`1=open`
- 驱动输出：六关节绝对 driver command 加独立二值夹爪命令
- 部署阶段不调用 IK 或 FK

V3 的 30 个模型点与 30 个控制时刻一一对应。`--query-frequency 10`
每次执行模型索引 `0..9` 后重新观察和规划，不再使用 Clean V2 的
`100↔30` 重采样映射。

## 2. 登录与环境

```bash
ssh -p 9966 master-ht@192.168.1.86

source /mnt/wd10/tongxi/egoverse_piper/EgoVerse/emimic/bin/activate
export PYTHONPATH=/mnt/wd10/kerun/egoverse_piper/EgoVerse:/mnt/wd10/kerun/lerobot/src
export PYTHONDONTWRITEBYTECODE=1
cd /mnt/wd10/kerun/lerobot
```

运行部署前必须退出 `lerobot-teleoperate`，并保持急停可触及。

## 3. 不动真机的检查阶梯

### 3.1 只检查 checkpoint

```bash
python examples/piper/deploy_egoverse_hpt_joint.py \
  --mode checkpoint \
  --checkpoint-profile hpt-shared-v3-15k \
  --query-frequency 10 \
  --device cuda
```

该模式不构造机器人，也不连接 CAN 或相机。

### 3.2 录制数据离线推理

```bash
python examples/piper/deploy_egoverse_hpt_joint.py \
  --mode offline \
  --checkpoint-profile hpt-shared-v3-15k \
  --offline-npz /mnt/wd10/kerun/offline_fixtures/teddy_0912_7_ep0_first50.npz \
  --query-frequency 10 \
  --max-relative-target 1.5 \
  --max-joint-command-step 1.5 \
  --max-joint-command-accel 1.5 \
  --log-dir /mnt/wd10/kerun/deploy_logs/hpt_shared_v3 \
  --device cuda
```

检查日志中的以下字段：

- `checkpoint_contract.policy_contract_name == hpt_mixed_shared_action_v3`
- `output_head == shared:actions_joints`
- `model_chunk_shape == [30, 7]`
- `query_indices == [0,1,2,3,4,5,6,7,8,9]`
- 所有输出、proprio 和图像均有限值

### 3.3 连接真机但绝不发送命令

先恢复示范起始位姿、打开夹爪，并把熊和篮子放回训练分布。然后运行：

```bash
python examples/piper/deploy_egoverse_hpt_joint.py \
  --mode live-dry-run \
  --checkpoint-profile hpt-shared-v3-15k \
  --can-port can_follower \
  --query-frequency 10 \
  --max-steps 120 \
  --speed-percent 5 \
  --max-relative-target 0.5 \
  --max-joint-command-step 0.5 \
  --max-joint-command-accel 0.1 \
  --max-initial-joint-delta-deg 5 \
  --gripper-threshold 0.5 \
  --gripper-prediction-source full-horizon \
  --gripper-control-mode latched \
  --gripper-close-confirm-frames 3 \
  --gripper-open-confirm-frames 5 \
  --gripper-min-close-hold-frames 60 \
  --gripper-close-feedback-threshold 10 \
  --gripper-open-feedback-threshold 90 \
  --gripper-terminal-open-after-release \
  --log-dir /mnt/wd10/kerun/deploy_logs/hpt_shared_v3 \
  --device cuda \
  --save-inference-images
```

该模式以 observation-only 方式连接 PIPER，日志中每一帧都必须满足
`command_sent=false`。还需人工检查保存的两路图像：

- `435if`：全局相机
- `405`：腕部相机
- 两路均应为 `480×640×3`，物体清晰、无互换、无严重偏色或过曝

## 4. 首次正式执行前的通过条件

以下条件必须同时满足：

1. checkpoint 哈希一致，完整 `ModelWrapper` 成功加载。
2. dry-run 全程 `command_sent=false`，无其它 teleoperate/deploy 进程。
3. 初始首动作最大关节差不超过 `5°`。
4. 两路相机身份、尺寸和画面正确。
5. `model_chunk_shape=[30,7]`，输出分支为
   `shared:actions_joints`。
6. 起始阶段夹爪意图应与现场阶段一致；标准起始场景通常应先保持打开。
7. 重点审计 `joint_limit_clipped_steps`。若每个 chunk 都持续发生裁剪，先恢复
   示范起始位姿并重新 dry-run，不直接执行。
8. 热身后的单次模型推理应明显低于 30 Hz 控制周期；首次 CUDA 热身耗时不计。

## 5. 首次低速执行模板

只有在第 4 节全部通过后才使用。首次最多 120 帧，全程人工看守：

```bash
python examples/piper/deploy_egoverse_hpt_joint.py \
  --mode execute \
  --checkpoint-profile hpt-shared-v3-15k \
  --can-port can_follower \
  --query-frequency 10 \
  --max-steps 120 \
  --speed-percent 5 \
  --max-relative-target 0.5 \
  --max-joint-command-step 0.5 \
  --max-joint-command-accel 0.1 \
  --max-initial-joint-delta-deg 5 \
  --gripper-threshold 0.5 \
  --gripper-prediction-source full-horizon \
  --gripper-control-mode latched \
  --gripper-close-confirm-frames 3 \
  --gripper-open-confirm-frames 5 \
  --gripper-min-close-hold-frames 60 \
  --gripper-close-feedback-threshold 10 \
  --gripper-open-feedback-threshold 90 \
  --gripper-terminal-open-after-release \
  --log-dir /mnt/wd10/kerun/deploy_logs/hpt_shared_v3 \
  --device cuda \
  --allow-motion
```

正式运动时不要加入 `--save-inference-images`：同步 PNG 编码会把部分控制帧
拉长到约 `178 ms`；模型本身热身后约 `10 ms`，不保存图片时才适合控制循环。

出现反向运动、连续限位裁剪、非预期夹爪闭合、相机中断或控制延迟突增时，
立即急停或 `Ctrl+C`，不要通过放宽保护参数继续运行。

## 6. 已完成的验证记录

- 单元与静态测试：`29 passed`
- V3 checkpoint、Piper Clean V2、Mixed Clean V2 均能由同一部署入口加载
- V3 录制数据离线推理：`[30,7]`，热身后平均约 `14.7 ms`
- V3 60 帧 live dry-run：零控制命令，热身后平均约 `10.0 ms`
- live 相机：`435if` 全局、`405` 腕部，均为 `640×480 RGB`
- 当前一次 live dry-run 处于熊已贴近夹爪的中间阶段，模型 30 步均预测闭合；
  同时关节 2/3 每个已选点都触发安全限位裁剪。因此这次结果只证明部署链路
  正确，不能作为正式首动放行依据。应恢复标准起点后重新执行第 3.3 节。
