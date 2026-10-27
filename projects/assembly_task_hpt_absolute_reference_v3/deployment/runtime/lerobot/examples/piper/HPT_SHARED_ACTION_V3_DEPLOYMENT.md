# HPT Shared-Action V3 真机部署说明

## 1. 部署契约

部署入口：`deploy_egoverse_hpt_shared_action_v3_joint.py`

该入口加载训练阶段完整 Lightning `ModelWrapper`，使用同一套：

- ResNet18 front/wrist 图像编码器；
- PIPER 关节 proprio 编码器；
- 16-block HPT shared trunk；
- Shared-Action V3 原生 30 步时序 decoder；
- PIPER 6D absolute-joint projection；
- 独立 binary gripper projection。

模型同时产生 `30×9D` camera-frame shared pose 和
`30×(6D joint + 1D gripper)`。真机只执行后者；shared pose 只写入诊断日志。
整个部署链不使用 IK、FK、Flow Matching 或 30↔100 插值。

## 2. Checkpoint profiles

- `hpt-v3-trim-val-best-5k`：验证集综合 physical score 选择的正式模型。
- `hpt-v3-trim-joint-candidate-15k`：验证集 joint-only 候选；用于对照，不替代正式选择。

每个 profile 固定 checkpoint 路径和 SHA-256，校验不一致时拒绝加载。

## 3. 强制分阶段检查

以下四个阶段不得跳级。

### 阶段 A：只加载 checkpoint

```bash
python examples/piper/deploy_egoverse_hpt_shared_action_v3_joint.py \
  --mode checkpoint \
  --checkpoint-profile hpt-v3-trim-val-best-5k \
  --device cuda
```

该模式不构造机器人，不打开相机或 CAN。

### 阶段 B：离线 NPZ

```bash
python examples/piper/deploy_egoverse_hpt_shared_action_v3_joint.py \
  --mode offline \
  --checkpoint-profile hpt-v3-trim-val-best-5k \
  --offline-npz /path/to/recorded_observation.npz \
  --query-frequency 10 \
  --max-steps 120 \
  --device cuda
```

NPZ 必须包含：

- `observation.state`
- `observation.images.435if`
- `observation.images.405`

### 阶段 C：live dry-run

```bash
python examples/piper/deploy_egoverse_hpt_shared_action_v3_joint.py \
  --mode live-dry-run \
  --checkpoint-profile hpt-v3-trim-val-best-5k \
  --can-port can_follower \
  --query-frequency 10 \
  --max-steps 120 \
  --speed-percent 5 \
  --max-relative-target 0.5 \
  --max-gripper-relative-target 100 \
  --max-joint-command-step 0.5 \
  --max-joint-command-accel 0.1 \
  --max-initial-joint-delta-deg 5 \
  --gripper-prediction-source full-horizon \
  --gripper-control-mode latched \
  --device cuda
```

该模式连接相机和 CAN feedback，但运动命令保持禁用。

### 阶段 D：execute

只有 checkpoint、offline、live-dry-run 的日志均通过人工检查后才允许执行。
首次执行必须低速、短时、有人看守并保持急停可触及。正式 execute 命令在完成
live-dry-run 审核后再生成，避免把未经确认的参数复制到真机。

## 4. 部署时必须核对

- front serial：`327122075831`，640×480 RGB 30 Hz；
- wrist serial：`427622273943`，640×480 RGB 30 Hz；
- calibration：`/mnt/wd10/kerun/calibration/my_piper_follower.json`；
- CAN：必须显式指定并再次确认 `can_follower`；
- 两个相机的视角、色彩、曝光和训练数据一致；
- 起始关节状态与第一个预测动作的最大差值通过 startup gate；
- full 30-step gripper logits、binary 命令、关节 clipping 数和模型延迟均写入 JSONL；
- teleoperate 和部署进程不得同时运行。

## 5. 当前推荐策略

初始验证使用 `qf=10`、不启用 temporal ensemble、不启用 chunk smoothing。
这与独立测试评测的 `qf=10` 合同一致。滤波或 ensemble 只能在基本轨迹正确后做
独立 A/B，避免把模型问题与部署后处理混在一起。
