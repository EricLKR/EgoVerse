# Assembly Task — HPT Absolute-Reference V3

本目录保存 **Assembly Task New1** 当前 HPT 训练与真机部署代码。对应正式部署
checkpoint profile 为：

```text
assembly-new1-hpt-absolute-reference-v3-30k
```

## 目录

- `training/`：数据划分、相机契约、训练前检查、HPC smoke/formal/eval 脚本与评测代码。
- `deployment/`：同步/异步 RTC/retimed/delayed-release 部署、Home、PIPER runtime、配置和测试。
- 仓库顶层 `egomimic/`：训练该 checkpoint 时使用的完整 EgoVerse 代码快照。

## 模型契约

- 单帧、三相机输入：front、wrist、side。
- PIPER + human mixed HPT 共享 9D camera-frame pose 表征。
- PIPER 专用 `AbsoluteReferenceJointPolicyV3` 输出 6D 绝对关节轨迹。
- 独立二值夹爪 head。
- 原生控制 horizon 为 30；兼容接口 action horizon 为 100。
- 真机部署不使用 IK；FK 只出现在训练损失中。

## Checkpoint

Git 分支只保存代码、配置与权重校验信息，不上传大型 checkpoint。

```text
file: assembly_new1_hpt_absolute_reference_v3_step30000.ckpt
sha256: 2c0fb0c724de01a24bdfd8c29e6e22de14d784c64a2bc8de275593035589e992
robot path: /mnt/wd10/kerun/assembly_task_new1_deployment_v2/checkpoints/
```

## 正式部署命令（2026-10-27 保存）

这是会驱动真机的命令。运行前必须确认三相机、CAN、六轴使能、Home、清空工作区，
并保持急停可触及。

```bash
cd /mnt/wd10/kerun/assembly_task_new1_deployment_v2

PYTHON_BIN=/mnt/wd10/tongxi/egoverse_piper/EgoVerse/emimic/bin/python \
./run_hpt_mixed_v2_async_rtc_retimed_delayed_release_current.sh execute \
  --retime-advance-mode command-substep \
  --retime-max-joint-step-deg 1.5 \
  --retime-max-feedback-lead-deg 3.0 \
  --model-limit-soft-clamp-tolerance-deg 1.0 \
  --max-steps 6000 \
  --speed-percent 10 \
  --j2-command-offset-deg 5.0 \
  --j3-command-offset-deg 0.0 \
  --j5-command-offset-deg -0.5 \
  --gripper-effort 5000 \
  --gripper-delayed-release-max-nearest-pose-rms-deg 1.8 \
  --no-home-at-end \
  --allow-motion
```

命令行参数位于 wrapper 默认参数之后，因此同名参数采用这里保存的最后一个值。

## Homing 命令（2026-10-27 保存）

这同样会驱动真机。先托稳机械臂、清空扫掠空间并确认急停可触及。

```bash
cd /mnt/wd10/kerun/assembly_task_new1_deployment_v2

./run_home_only.sh \
  --mode execute \
  --can-port can_leader \
  --speed-percent 25 \
  --max-relative-target 2.5 \
  --max-gripper-relative-target 100 \
  --home-velocity-deg-s 25 \
  --home-acceleration-deg-s2 50 \
  --home-timeout-s 45 \
  --home-joint-tolerance-deg 1.0 \
  --home-gripper-tolerance-mm 2.0 \
  --j2-feedback-offset-deg 0 \
  --j2-command-offset-deg 3.8 \
  --j3-command-offset-deg -0.5 \
  --j5-command-offset-deg 1.5 \
  --allow-motion
```

## 不在 Git 中的内容

- 训练数据与 Zarr episode
- checkpoints
- 真机运行日志和全程图像
- replay/offline NPZ
- Python cache、core dump 与历史 `.pre-*`/backup 文件

