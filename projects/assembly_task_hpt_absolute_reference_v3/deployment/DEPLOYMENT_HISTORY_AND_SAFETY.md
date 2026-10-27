# Assembly New1 部署包

这是 Assembly New1 新数据上的单帧、三相机、直接关节空间模型部署包：

1. `assembly-new1-hpt-mixed-v2-30k`：HPT Mixed Clean V2，100 点模型轨迹映射到 30 个控制 tick。
2. `assembly-new1-hpt-expert-action-v3-30k`：HPT Expert-Action V3，原生 30 点轨迹。
3. `assembly-new1-hpt-absolute-reference-v3-25k`：HPT Absolute-Reference V3，验证集候选，100 点接口映射到 30 个控制 tick。
4. `assembly-new1-hpt-absolute-reference-v3-30k`：HPT Absolute-Reference V3，独立测试集首选，100 点接口映射到 30 个控制 tick。
5. `assembly-new1-pi05-mixed-v2-30k`：pi0.5 Mixed V2，完整训练 `ModelWrapper`、Gemma action expert与三相机前向，原生 30 点轨迹。

三者均输出六维绝对关节弧度和独立二值夹爪，不使用 FK 或 IK。训练视觉输入固定为：

- 全局相机 `337322071261` → `front_img_1`
- 腕部相机 `427622273943` → `wrist_img`
- 侧面相机 `427622272585` → `side_img`

## 原 leader 作为部署机械臂

当前真机是数据采集时的原 leader，而 checkpoint 的状态和动作语义属于原
follower。机器人数据转换器已经把记录的 follower 百分位动作还原为物理关节
弧度，因此 checkpoint 学到的是 follower 的物理关节角，不是两台机械臂之间的
遥操作量程百分位。本包因此保留两个
明确分离的空间：

- 模型空间：`assets/piper_follower_calibration.json`；checkpoint、proprio 和
  解码轨迹始终保持训练时的 follower 语义。
- 真机空间：`assets/piper_original_leader_calibration.json`；SDK 限位、相对限幅、
  驱动编码、物理限位和实际 CAN 命令使用原 leader 的 calibration。

`robot/joint_space_adapter.py` 现在严格恢复采集时的量程百分位关系。模型输出先按
follower calibration 归一化，再按原 leader calibration 反归一化：
`q_hardware = D_leader(N_follower(q_model)) + offset`。反馈 proprio 使用严格逆变换：
`q_model = D_follower(N_leader(q_hardware - offset))`。因此原 leader 在采集时产生的
每一个百分位命令都能被当前这台同一原 leader 重建；不再错误地假设两台机械臂
相同的物理角读数代表相同的遥操作命令。若以后批准了残余零位偏置，它只在已知
百分位映射之后应用。
夹爪仍按开口百分比映射，因为两台夹爪行程不同。`reset_control.json` 保存的旧
leader/follower 百分位 Home 配对只用于审计采集链路，不再用于策略动作或相机
Home。`assets/joint_zero_offset.json` 当前为 `inactive` 的全零映射；只有状态为
`approved_active` 且至少三个姿态的验证记录通过时，非零偏置才允许加载。任何模型
目标若超出原 leader 的物理关节限位都会在发送前拒绝。

## 异步推理 + RTC 入口

同步部署文件保持不变；异步入口包括：

- `run_hpt_mixed_v2_async_rtc.sh`
- `run_hpt_mixed_v2_async_rtc_first_grasp_guard.sh`
- `run_hpt_expert_action_v3_async_rtc.sh`
- `run_hpt_absolute_reference_v3_25k_async_rtc.sh`
- `run_hpt_absolute_reference_v3_30k_async_rtc.sh`
- `run_hpt_absolute_reference_v3_30k_async_rtc_gripper_deadline.sh`
- `run_pi05_mixed_v2_async_rtc.sh`

异步版先同步生成并检查首个 30-step chunk，之后由单个后台线程推理；控制线程继续执行旧计划仍有效的后缀。新结果以观测捕获时的 successful-action tick 为起点，安装时跳过已过期前缀，并在解码后的物理 6D 关节弧度空间对重叠段做余弦融合。夹爪不参与连续值融合，仍使用独立的二值绝对时间轴和反馈状态机。过期结果、剩余后缀不足和计划耗尽都有显式日志或安全停止，不会把过时 chunk 从头重放。

`first_grasp_guard` 是隔离的首次抓取修复入口。它不再让最老的预测覆盖新视觉结果：首次闭合必须由最新两个不同 chunk 同意，并且当前模型空间 6D 关节姿态必须进入由 90 条训练 episode 的首次闭合帧生成的姿态分布。门控失败时夹爪保持打开、手臂继续重规划，日志记录最近训练样本 RMS 和逐关节包络结果。原入口行为不变。

`gripper_deadline` 是 Absolute-Reference V3 30k 的隔离修复入口。若连续两个预测 chunk 都在未来给出首次闭合，它会锁存两者中最早的绝对 action tick；后续重规划只能把该期限提前，不能反复把闭合推迟到新 chunk 尾部。到期后仍必须通过上述训练姿态门，最多等待 10 个成功 action tick，超时则取消而不是在错误姿态闭合。旧入口仍保持原行为。

异步 RTC 固定要求 `--query-frequency 30`，且不得与旧的 `--temporal-ensemble` 或 `--chunk-smoothing` 叠加。HPT 的离线延迟低于一个 30-tick chunk；pi0.5 在当前离线测量中预热后约 1.37 秒，超过 30 Hz 下一个 chunk 的 1 秒跨度，因此新增文件不等于 pi0.5 已经通过实时门。pi0.5 必须在目标机器人主机重新测量延迟；若仍超过时域，RTC 会出现 plan starvation 并安全停止，而不是掩盖延迟。

## 安全边界

- `checkpoint` 只加载权重，不构造机器人、不打开相机、不访问 CAN。
- `offline` 只读取 NPZ。
- `live-dry-run` 会连接相机和 CAN，但保持 observation-only。
- `execute` 才能运动，并且必须显式提供 `--allow-motion`。
- `execute` 不再自行 quick-stop/重新使能。必须先用 `run_home_only.sh execute`
  回到起点并保持六轴使能；部署只接管这个已使能状态。六轴全禁用或部分禁用时，
  部署会在发送任何策略动作前拒绝启动。
- HPT 正式脚本和独立 Home 都使用
  `assets/initial_pose_follower_physical_on_original_leader.json`。模型输出已经是 follower
  物理关节角；同型 PIPER 上必须保持该物理角度，仅在驱动边界用当前机械臂 calibration
  编码，不能把 calibration 范围当成两台机械臂之间的仿射运动学映射。
- 接管后每个控制周期都会检查控制器、驱动故障和六轴使能状态；任一轴掉使能即停止发送后续动作。
- 机器人未开机时只能完成静态检查和 checkpoint load；不能把 `live-dry-run` 或 `execute` 当作已验证。

## 不加载模型的独立回 Home

`run_home_only.sh` 不导入 Torch 或策略类，不读取 checkpoint，也不连接三台相机；它只连接指定的 PIPER CAN 口。目标严格读取本部署包的 `assets/initial_pose_follower_physical_on_original_leader.json`，限位来自 `assets/piper_original_leader_calibration.json`。

先只读检查当前反馈和规划时长（不会使能或运动）：

```bash
./run_home_only.sh --mode check --can-port can_follower
```

现场确认工作空间清空、急停可触及并托稳机械臂后，才执行：

```bash
./run_home_only.sh \
  --mode execute \
  --can-port can_follower \
  --speed-percent 5 \
  --max-relative-target 0.5 \
  --home-velocity-deg-s 5 \
  --home-acceleration-deg-s2 10 \
  --home-gripper-velocity-mm-s 10 \
  --home-joint-tolerance-deg 1.0 \
  --home-gripper-tolerance-mm 2.0 \
  --allow-motion
```

脚本会先 observation-only 读取反馈，再显式使能并锁存当前位置，然后以五次曲线回到 Assembly New1 起点。正常完成或异常退出均保持最后实测位置；断开连接不会自动解除电机使能。

看到 `HOME COMPLETE` 后才能另起部署进程。部署进程的接管路径不会调用
`EmergencyStop`、`EnablePiper` 或 `DisablePiper`，只把当前实测位置重新写为保持目标。

## Assembly New1 采集起点

这次数据的起点与上一版 `piper_assembly_1012_zarr` 不同，禁止复用旧任务的复位目标：

| 数据 | J1 | J2 | J3 | J4 | J5 | J6 |
|---|---:|---:|---:|---:|---:|---:|
| 上一版 1012 起点（deg） | 0.608 | 58.151 | -46.543 | -7.334 | 52.797 | 2.721 |
| 本版模型/follower 物理起点（deg） | 6.460 | 42.912 | -27.554 | -11.212 | 41.809 | 7.905 |
| 当前真机原 leader 起点（deg） | 6.460 | 42.493 | -28.753 | -11.511 | 40.529 | 7.839 |

模型/follower 目标直接来自 100 个新机器人 episode 的首帧 `cmd_joint`；100 条记录完全一致。首帧实测 `obs_joint` 的平均值为 `[6.4882, 42.9580, -27.5889, -11.2263, 41.8133, 7.9255]` deg，任一 episode、任一关节相对上述 follower 目标的最大绝对偏差为 0.098°。当前真机是原采集 leader，因此 Home 和整段动作改为使用采集链路的严格逆映射，而不是 follower 物理角 identity。

## 无模型 Episode Replay

`replay/assembly_new1_ep0000_original_leader.npz` 从原始 LeRobot `action` 导出。它同时保存
follower 模型角和原 leader 重放角，并绑定两份 calibration 的 SHA-256。重放不加载
checkpoint、不打开相机、不使用 FK/IK。首先只做离线检查：

```bash
./run_episode_replay.sh \
  --mode check \
  --trajectory replay/assembly_new1_ep0000_original_leader.npz \
  --start-frame 0 \
  --end-frame 60 \
  --time-scale 4 \
  --max-joint-step-deg 0.25
```

回到原 leader Home 后，用 observation-only CAN 检查起点误差：

```bash
./run_episode_replay.sh \
  --mode live-dry-run \
  --trajectory replay/assembly_new1_ep0000_original_leader.npz \
  --can-port can_follower \
  --start-frame 0 \
  --end-frame 60 \
  --time-scale 4 \
  --max-joint-step-deg 0.25
```

只有 dry-run 报告的起点最大误差不超过 1°、六轴使能且工作区清空时，才允许同一
60 帧前缀加 `--mode execute --allow-motion`。禁止第一次就重放完整 1952 帧。

### 腕部真实零位对齐

若原 leader 在上述读数下的真实腕部姿态仍与训练 follower 不一致，不应移动或
旋转固定的 D405 支架，也不能仅凭相同角度读数假设两台机械臂具有完全相同的
机械零位。`tools/check_wrist_visual_alignment.py` 只打开 D405 RGB，不导入机器人
驱动、不访问 CAN、不发送动作；它将训练首帧特征映射到当前画面并报告固定误差。
理想结果为 `tx=0 px, ty=0 px, rotation=0 deg, scale=1`。只读检查可以运行：

```bash
source /mnt/wd10/tongxi/egoverse_piper/EgoVerse/emimic/bin/activate
python tools/check_wrist_visual_alignment.py \
  --output-dir validation/wrist_alignment_manual
```

人工同时调 J4/J5/J6 很难收敛，因此另提供独立的
`run_wrist_zero_calibration.sh`。其 `check` 模式会读取 CAN 反馈和 D405，但保持
observation-only，不使能、不发送动作：

```bash
./run_wrist_zero_calibration.sh \
  --mode check \
  --can-port can_follower
```

`execute` 只允许从保存的 Home 附近开始。它对 J4/J5/J6 做默认 ±0.5° 中央差分
探测，数值估计“关节角变化 → 图像平移/旋转误差”的 Jacobian，再用有阻尼最小
二乘和实测 line search 求解。每轮单轴修正默认不超过 1°，累计单轴偏置不超过
8°；视觉 Jacobian 退化、实测无改善、超限或未收敛都会停止，默认返回起始 Home
并锁存当前位置。现场清空工作区、急停可触及并持续看守后，才可执行：

```bash
./run_wrist_zero_calibration.sh \
  --mode execute \
  --can-port can_follower \
  --active-joints 4,5,6 \
  --probe-step-deg 0.5 \
  --max-update-step-deg 1.0 \
  --max-total-offset-deg 8.0 \
  --speed-percent 5 \
  --max-relative-target 0.25 \
  --allow-motion
```

即使成功，程序也只在
`validation/wrist_zero_calibration/execute_*/candidate_joint_zero_offset.json`
保存候选 `hardware = model + zero_offset`，不会自动修改 Home、adapter 或部署命令。
必须先审查候选偏置和最终图像残差，再显式激活；在偏置尚未确认前不得启动正式
部署。

### 只有一台 PIPER 时的三姿态映射验证

旧 follower 已损坏且当前只剩一台 PIPER，不需要也不能再做 leader-follower
遥操作采集。本包从现有 100 条 Assembly New1 示范中固定提取了 episode 0 的三个
夹爪打开、接触前姿态（frame 0、30、45），并保存三相机参考图。三个姿态均位于
第一次物体接触之前，物体布局没有变化，避免把任务执行后的场景变化误判成机械臂
零位误差。它们
只用于验证上面的 Home 候选偏置能否作为**整段轨迹恒定映射**：

```text
q_hardware = q_model + [0, 0, 0, +7.6702, -0.6841, -5.1376] deg
q_model    = q_hardware - same_offset
```

先运行完全静态检查；这一步不打开相机、CAN 或模型：

```bash
./run_joint_mapping_validation.sh --mode check
```

只有静态检查通过、机械臂已经用当前 inactive 映射回到保存的未偏置 Home、夹爪
完全打开、整个扫掠空间清空且急停可触及时，才可人工执行三姿态验证：

```bash
./run_joint_mapping_validation.sh \
  --mode execute \
  --can-port can_follower \
  --speed-percent 5 \
  --max-relative-target 0.25 \
  --move-velocity-deg-s 3 \
  --move-acceleration-deg-s2 6 \
  --allow-motion
```

工具在发送动作前重新检查关节限位，并用官方 PIPER FK 对 Home 到每个目标的
末端路径做至少 200 mm 高度预检；FK 只检查末端，因此现场仍必须清空整条机械臂
扫掠区域。每个姿态都会保存当前三相机图。腕部图使用 ORB/RANSAC 自动门控：
平移不超过 5 px、旋转不超过 0.75°、尺度误差不超过 0.02；全局和侧面图仅保存
供人工审核，不用背景仿射结果替代人工检查。每个姿态后都会回到保存的 Home。

三个姿态全部通过时，输出也只是
`validated_joint_zero_offset.json`（`validated_pending_approval`），不会修改
`assets/joint_zero_offset.json`。审核三组 montage 后，才可用报告路径和精确确认
口令执行独立的静态批准步骤：

```bash
./run_joint_mapping_validation.sh \
  --mode approve \
  --validation-report validation/joint_mapping_multi_pose/execute_UTC/validation_report.json \
  --approval-token APPROVE_VALIDATED_MAPPING
```

批准操作会先备份旧映射，再写入 `approved_active`；adapter、Home、模型 proprio
和每一帧动作随后统一使用同一个双向偏置。如果任一姿态失败，不得批准；这说明
单一零位偏置不足以解释两台机械臂的差异，在没有旧 follower 和新增数据的条件下
不能靠放宽保护安全修复。

## 准备隔离运行时

部署主机已有 LeRobot 工程时，先建立隔离副本。此命令不会访问硬件：

```bash
./tools/prepare_runtime.sh \
  --lerobot-root /path/to/lerobot \
  --pi05-base-weights /path/to/pi05_base_pytorch \
  --pi05-tokenizer /path/to/paligemma_tokenizer.model \
  --pi05-python-overlay /path/to/python_overlay
```

HPT 不依赖后三个 pi0.5 资源；pi0.5 需要全部三个资源。脚本拒绝覆盖已经存在的 `runtime/lerobot`。

## 无硬件验证

```bash
python tools/validate_bundle.py
python -m unittest discover -s tests -v

./run_hpt_mixed_v2.sh checkpoint --device cuda
./run_hpt_expert_action_v3.sh checkpoint --device cuda
./run_hpt_absolute_reference_v3_25k.sh checkpoint --device cuda
./run_hpt_absolute_reference_v3_30k.sh checkpoint --device cuda
./run_pi05_mixed_v2.sh checkpoint --device cuda

./run_hpt_mixed_v2_async_rtc.sh checkpoint --device cuda
./run_hpt_expert_action_v3_async_rtc.sh checkpoint --device cuda
./run_hpt_absolute_reference_v3_25k_async_rtc.sh checkpoint --device cuda
./run_hpt_absolute_reference_v3_30k_async_rtc.sh checkpoint --device cuda
./run_hpt_absolute_reference_v3_30k_async_rtc_gripper_deadline.sh checkpoint --device cuda
./run_pi05_mixed_v2_async_rtc.sh checkpoint --device cuda
```

最后三条只有在对应 Python/CUDA 运行时和 pi0.5 外部资源就绪后才执行。它们仍不会访问机器人。

部署包还带有一个从 held-out 测试 episode 0019 第 0 帧生成的真实三相机样本：
`offline_samples/assembly_new1_test_ep0019_frame000.npz`。以下命令只做一帧模型前向，
不会构造机器人、打开相机或访问 CAN：

```bash
./run_hpt_mixed_v2.sh offline --device cuda \
  --offline-npz offline_samples/assembly_new1_test_ep0019_frame000.npz --max-steps 1
./run_hpt_expert_action_v3.sh offline --device cuda \
  --offline-npz offline_samples/assembly_new1_test_ep0019_frame000.npz --max-steps 1
./run_pi05_mixed_v2.sh offline --device cuda \
  --offline-npz offline_samples/assembly_new1_test_ep0019_frame000.npz --max-steps 1
```

该 NPZ 的状态是 live SDK 的原始单位（关节 0.001°、夹爪 0.001 mm），图像为
640×480 RGB。可用 `tools/create_offline_sample.py` 从别的 Zarr 帧重建；脚本要求
Python 3.11+、zarr 3 和 Pillow。

## 2026-10-20 HPT 真机只读门

目标主机 `kth-psa-i0001218` 上已经完成以下只读验证：

1. 三台相机序列号与训练合同一致，均能提供 640×480@30 RGB；
2. `can_follower` 为 1 Mbps、`ERROR-ACTIVE`，RX 持续增加且零 CAN error/bus-off；
3. HPT Mixed V2 异步 RTC `live-dry-run` 完成 120 tick，11 个重规划全部接受，平均模型推理约 30.9 ms；
4. HPT Expert-Action V3 异步 RTC `live-dry-run` 完成 120 tick，11 个重规划全部接受，平均模型推理约 55.6 ms；
5. 两次 dry-run 均为 observation-only，日志确认没有发送任何运动命令。

色彩审计发现 front 当前比训练起始帧平均亮约 6.2%、饱和度低约 13.3%；side 饱和度低约 11.8%；wrist 的平均亮度和饱和度接近训练分布，但当前构图中绿色背景占比偏大。完整证据见
`validation/CAMERA_COLOR_AUDIT_20261020.md`。本次没有根据场景相关的全图统计强行添加色彩增益。

HPT 两版已经通过执行前的只读门，但仍未发送真机动作。正式 `execute` 前必须由现场人员确认机械臂周围清空、急停可触及；程序会先执行受保护的采集起点 homing。pi0.5 的目标主机实时门仍未在本节宣称通过。

本次无硬件静态检查、真实测试帧前向结果和仍未通过的实时门见
`validation/OFFLINE_VALIDATION_20261020.md`。
