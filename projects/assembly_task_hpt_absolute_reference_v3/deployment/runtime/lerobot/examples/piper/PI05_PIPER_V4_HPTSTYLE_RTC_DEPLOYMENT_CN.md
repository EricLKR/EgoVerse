# EgoVerse pi0.5 PIPER V4 HPT-style RTC 真机部署说明

## 正式部署命令

> 下面是当前确认并记录的基准命令。该命令会让真机运动；运行前必须确认遥操作程序已退出、工作区无人、机械臂周围无障碍物，并保持急停可触及。

```bash
python examples/piper/deploy_egoverse_pi05_piper_v3_v4_hptstyle_rtc_joint.py \
  --mode execute \
  --checkpoint-profile pi05-piper-v4-10k \
  --can-port can_follower \
  --seed 42 \
  --query-frequency 30 \
  --inference-steps 10 \
  --max-steps 2500 \
  --speed-percent 15 \
  --max-relative-target 3.5 \
  --max-gripper-relative-target 100 \
  --max-initial-joint-delta-deg 5 \
  --max-joint-command-step 3.5 \
  --max-joint-command-accel 1.5 \
  --rtc-max-joint-command-jerk 0.5 \
  --rtc-command-deadband 0.1 \
  --rtc-intrachunk-smoothing \
  --rtc-intrachunk-window 7 \
  --rtc-intrachunk-sigma 1.5 \
  --rtc-replan-interval 10 \
  --rtc-blend-actions 8 \
  --rtc-min-suffix-actions 8 \
  --rtc-max-result-age-ms 1500 \
  --rtc-max-starvation-hold-frames 90 \
  --gripper-threshold 0.5 \
  --gripper-prediction-source full-horizon \
  --gripper-control-mode latched \
  --gripper-close-confirm-frames 3 \
  --gripper-open-confirm-frames 5 \
  --gripper-min-close-hold-frames 60 \
  --gripper-min-open-hold-frames 60 \
  --gripper-close-feedback-threshold 10 \
  --gripper-open-feedback-threshold 90 \
  --no-gripper-terminal-open-after-release \
  --gripper-contact-detection \
  --gripper-contact-max-feedback 70 \
  --gripper-contact-stall-frames 15 \
  --gripper-contact-motion-tolerance 0.5 \
  --gripper-contact-min-close-frames 10 \
  --gripper-contact-min-drop 10 \
  --wrist-color-normalization startup \
  --device cuda \
  --allow-motion
```

## 1. 登录与环境准备

在机器人主机上执行：

```bash
ssh -p 9966 master-ht@192.168.1.86

source /mnt/wd10/tongxi/egoverse_piper/EgoVerse/emimic/bin/activate

export PYTHONPATH=/mnt/wd10/kerun/egoverse_piper/EgoVerse:/mnt/wd10/kerun/lerobot/src
export PYTHONDONTWRITEBYTECODE=1

cd /mnt/wd10/kerun/lerobot
```

本部署只使用 `/mnt/wd10/kerun` 下的代码和模型文件。不要修改 `/mnt/wd10/tongxi`；该目录在这里只用于复用现有 Python 虚拟环境。

## 2. 当前部署版本

- 部署入口：`examples/piper/deploy_egoverse_pi05_piper_v3_v4_hptstyle_rtc_joint.py`
- 模型配置：`pi05-piper-v4-10k`
- 模型类型：EgoVerse pi0.5 PIPER-only Expert-Action V4 bounded LoRA
- 训练 checkpoint：验证集选出的 10,000 step checkpoint
- checkpoint 路径：`/mnt/wd10/kerun/pi05_piper_v3_v4/checkpoints/v4_selected_step10000_trainable.ckpt`
- 动作空间：6 维 PIPER 绝对关节角（弧度）加 1 维二值夹爪
- 动作 horizon：原生 30 步，不进行 `30→100` 插值
- 观测历史：单帧
- 控制频率：30 Hz
- 坐标转换：关节空间直接控制，不使用 IK，也不需要用 FK 生成控制指令

部署加载的是完整训练阶段 `ModelWrapper`，并通过：

```text
ModelWrapper.model.forward_joint_deploy
```

执行 joint-only 部署前向。归一化统计、模型配置、pi0.5 基础权重和 LoRA/可训练权重均由 checkpoint profile 固定并校验，不能随意替换成未审计的 checkpoint。

## 3. 输入、模型输出与执行链路

### 3.1 模型输入

每次推理使用：

- `435if` 全局相机 RGB 图像；
- `405` 腕部相机 RGB 图像；
- 当前 PIPER proprio/关节状态；
- 固定任务文本：`Pick up the teddy bear and place it in the basket.`；
- rollout 内固定的 Flow Matching 推理噪声，随机种子由 `--seed 42` 指定。

腕部 D405 图像在启动阶段估计颜色增益，随后整个 rollout 固定使用该校正值。它不会在每一帧重新改变色彩。

### 3.2 模型原生输出

模型一次生成：

```text
30 × 7
```

其中：

- 前 6 维是未来 30 个控制时刻的绝对关节角，单位为弧度；
- 第 7 维是独立二值夹爪预测，`0` 表示闭合，`1` 表示张开。

6 维关节角随后被转换为 PIPER 驱动坐标。夹爪二值输出转换为 `0%` 或 `100%` 命令，但不会参与连续数值平滑。

### 3.3 完整执行链路

```text
单帧双相机图像 + proprio + 文本
        ↓
训练阶段完整 ModelWrapper
        ↓
pi0.5 / Gemma Action Expert
        ↓
原生 30 步：6D 绝对关节弧度 + 二值夹爪
        ↓
6D 手臂 chunk 内高斯时间平滑
        ↓
decoded-joint 绝对时间 RTC 余弦重叠融合
        ↓
转换为 PIPER driver coordinates
        ↓
因果速度 / 加速度 / jerk 限制
        ↓
PIPER 驱动层物理安全剪裁
        ↓
can_follower 真机
```

夹爪走独立路径：

```text
完整 30 步二值夹爪预测
        ↓
绝对时间选择当前仍有效的最早预测
        ↓
latched 状态机 + 连续帧确认
        ↓
最短开/闭保持时间 + 接触检测
        ↓
独立夹爪命令
```

## 4. HPT-style decoded-joint RTC

该部署没有在 Flow Matching 采样器内部再加一层 RTC guidance。连续性只在解码后的 6D 关节空间处理，避免两套 RTC 相互干扰。

主要逻辑如下：

1. 后台线程执行模型推理，主控制循环继续执行现有 chunk；
2. 每个预测绑定其观测时刻对应的绝对 action tick；
3. 推理完成时，已经过期的 chunk 前缀会被跳过；
4. 新旧 chunk 在相同绝对时间上对齐；
5. 重叠部分使用余弦权重融合；
6. 只有剩余有效动作不少于 `--rtc-min-suffix-actions 8` 时才安装新计划；
7. 超过 `--rtc-max-result-age-ms 1500` 的过旧推理结果会被拒绝；
8. 如果队列断档，最多保持 90 个控制帧，然后安全终止。

当前 RTC 参数：

- 每执行 10 个 action tick 请求一次新预测；
- 新旧 chunk 融合 8 个动作；
- chunk 内高斯平滑窗口为 7，标准差为 1.5；
- 平滑只作用于 6D 手臂轨迹，夹爪二值序列原样保留。

## 5. 关节命令平滑和安全限制

当前命令使用以下限制：

- 单 tick 最大关节命令步长：`3.5` driver-coordinate units；
- 最大命令加速度：`1.5` driver-coordinate units/tick²；
- 最大 jerk：`0.5` driver-coordinate units/tick³；
- 小于 `0.1` 的目标误差进入 deadband；
- PIPER 驱动层另有 `--max-relative-target 3.5` 的最终物理安全剪裁。

这些数值是 PIPER 归一化驱动坐标，不是角度，也不是弧度。

驱动安全剪裁不会被关闭。如果模型给出的命令超过当前反馈附近的安全范围，驱动仍会发送剪裁后的目标。当前版本已经修正 action clock 行为：

- 剪裁后的命令只要成功发送，就会推进到下一个 action tick；
- RTC 滤波状态同步为驱动实际接受的命令；
- 不会再因为持续出现 `arm_command_was_clipped=true` 而永久重复同一个 action；
- 如果夹爪尚未物理到位，action tick 仍会按设计等待夹爪完成。

## 6. 夹爪控制逻辑

当前夹爪配置含义如下：

- `--gripper-threshold 0.5`：模型概率/连续输出以 0.5 二值化；
- `--gripper-prediction-source full-horizon`：状态机读取完整 30 步夹爪预测；
- `--gripper-control-mode latched`：确认状态切换后锁存，不随单帧噪声立即反转；
- 连续 3 帧闭合请求才开始闭合；
- 连续 5 帧张开请求才开始张开；
- 闭合后至少保持 60 帧；
- 张开后至少保持 60 帧；
- 反馈小于等于 10 视为完全闭合；
- 反馈大于等于 90 视为完全张开。

当前使用：

```text
--no-gripper-terminal-open-after-release
```

因此，夹爪完成一次“闭合→张开”后，模型如果再次给出足够稳定的闭合预测，状态机仍允许再次闭合。这与“松开后永久保持张开”的 terminal-open 模式不同。

### 6.1 抓住物体时的接触判断

夹爪可能因为夹住小熊而无法到达完全闭合反馈。接触判断用于把“夹住物体后停止继续闭合”识别为成功完成闭合阶段：

- 必须已经执行至少 10 帧闭合；
- 闭合反馈必须相对起点下降至少 10；
- 反馈需要处于 70 以下；
- 连续 15 帧内夹爪运动幅度不超过 0.5。

满足条件后，状态机会认为物体已被夹住并允许手臂轨迹继续执行，而不会永远等待夹爪达到反馈 10。

## 7. 关键运行参数说明

- `--seed 42`：固定同一次 rollout 的推理噪声，使相同输入下的输出更可复现。
- `--query-frequency 30`：本脚本强制要求 30，与模型原生 30 步动作 horizon 一致。
- `--inference-steps 10`：Flow Matching 使用 10 个数值积分步。
- `--max-steps 2500`：最多执行 2500 个 30 Hz 控制循环，理论上约 83 秒。
- `--speed-percent 15`：PIPER 控制器速度百分比。
- `--max-initial-joint-delta-deg 5`：首次模型目标与当前关节状态的最大允许初始差异。
- `--device cuda`：在机器人主机 GPU 上推理。
- `--allow-motion`：显式授权 execute 模式发送运动命令；没有该参数时脚本拒绝运动。

## 8. 运行前检查

### 8.1 确认没有其他控制进程

```bash
pgrep -af 'lerobot-teleoperate|deploy_egoverse|reset_piper'
```

如果有正在控制 `can_follower` 的 teleoperate 或旧部署进程，先在对应终端按 `Ctrl+C` 正常退出。不要让两个进程同时控制机械臂。

### 8.2 检查 CAN

```bash
ip -br link show type can
```

应看到：

```text
can_follower  UP
```

### 8.3 检查相机

```bash
lerobot-find-cameras realsense
```

需要同时识别训练/部署所使用的全局相机和腕部 D405。相机序列号必须与当前 PIPER 配置一致。

### 8.4 检查 GPU

```bash
nvidia-smi
```

确认 CUDA 可用，并且没有其他高占用训练任务抢占显存或计算资源。

### 8.5 机械安全检查

- 急停按钮可立即触及；
- 小熊、篮子和相机摆放与训练分布接近；
- 机械臂周围无人员和硬质障碍物；
- 夹爪没有卡住线缆；
- 相机 USB 线和 CAN 线不会进入机械臂运动范围；
- 当前位姿接近训练起始分布，并能通过 5° initial-delta 检查。

## 9. 日志位置与运行后检查

日志默认写入：

```text
/mnt/wd10/kerun/deploy_logs/joint_space/
```

查找最新日志：

```bash
find /mnt/wd10/kerun/deploy_logs/joint_space \
  -type f \
  -name 'pi05_piper_v3_v4_hptstyle_rtc_execute_*.jsonl' \
  -printf '%T@ %p\n' | sort -nr | head
```

重点检查：

- `command_sent`：execute 模式下是否成功发送命令；
- `action_advanced`：action tick 是否持续推进；
- `action_advance_commit`：滤波目标和驱动实际接受值是否一致；
- `arm_command_was_clipped`：最终驱动安全层是否发生剪裁；
- `rtc_controller_before.action_tick`：策略绝对时间是否单调递增；
- `rtc_starvation_hold`：是否因后台推理未完成而等待；
- `causal_filter.velocity_limited_axes`：速度限制涉及多少轴；
- `causal_filter.acceleration_limited_axes`：加速度限制涉及多少轴；
- `causal_filter.jerk_limited_axes`：jerk 限制涉及多少轴；
- `gripper_state_machine.transition`：夹爪何时发生开闭切换；
- `gripper_state_machine.contact_reached`：是否通过物体接触完成闭合阶段；
- `gripper_state_machine.release_latched`：是否已完成释放。

允许出现少量 `arm_command_was_clipped=true`，尤其是模型目标靠近关节标定边界时。异常情况是：大量帧剪裁，同时 `action_advanced=false` 且 action tick 不增长。当前版本已经针对这种冻结问题进行了修复。

## 10. 常见问题

### 10.1 机械臂完全不动

依次检查：

1. 是否误用了 `live-dry-run`；
2. 是否缺少 `--allow-motion`；
3. `can_follower` 是否为 `UP`；
4. 是否有 teleoperate 或旧部署进程占用 CAN；
5. 日志中的 `command_sent` 是否为 `true`；
6. `action_advanced` 和 action tick 是否增长；
7. 夹爪是否处于未完成状态，导致动作时钟等待；
8. 电机电源、PIPER controller enable 状态是否正常。

### 10.2 机械臂运动慢或动作很小

检查日志中的速度、加速度、jerk 限制计数和驱动剪裁。不要仅通过继续增大参数解决问题；先确认模型目标、真机反馈和 calibration 使用同一关节定义。

### 10.3 机械臂抖动

检查：

- chunk 内原始与平滑后的 adjacent joint RMS；
- RTC plan switch 前后的 joint RMS；
- 因果滤波后的速度换向率；
- 是否存在频繁新 chunk 安装；
- 相机画面是否抖动、曝光突变或颜色漂移。

当前部署已同时使用 chunk 内平滑、absolute-time RTC overlap 和因果 V/A/J 限制。不要再叠加旧的 `--chunk-smoothing` 或 legacy temporal ensemble。

### 10.4 夹爪夹住物体后不继续

检查 `contact_reached`、反馈下降量和 stall window。如果接触检测没有触发，确认真实夹爪反馈单位仍为 0–100，并检查接触阈值是否适合当前夹爪标定。

### 10.5 放入篮子后夹爪不张开

检查模型是否给出了至少连续 5 帧 open 请求，并检查最短闭合保持 60 帧是否已经结束。当前版本不会让任意未来 close 预测覆盖绝对时间上更早且仍有效的 open 命令。

### 10.6 松开后又闭合

当前命令明确使用 `--no-gripper-terminal-open-after-release`，因此允许模型再次闭合。这是当前基准配置的预期行为。如果未来需要“完成一次释放后永久保持张开”，应改用：

```text
--gripper-terminal-open-after-release
```

## 11. 停止与故障处置

- 正常停止：在部署终端按 `Ctrl+C`；
- 出现快速异常运动、碰撞风险或线缆拉扯：立即使用急停；
- 相机断连、CAN 报错或控制程序异常退出后，不要直接反复重启 execute；先确认机械臂状态、相机和 CAN，再从 observation-only 检查开始；
- execute 模式禁止同步保存推理图像，因此不要加入 `--save-inference-images`。需要保存图像时使用 `live-dry-run`。

## 12. 当前版本边界

这份文档对应且只对应：

```text
deploy_egoverse_pi05_piper_v3_v4_hptstyle_rtc_joint.py
checkpoint profile: pi05-piper-v4-10k
```

其他部署入口、V3 checkpoint、Mixed Clean V2、旧 HPT 或 generative RTC 的参数和控制语义可能不同，不能直接套用本命令。
