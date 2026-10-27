# the folder dataset introduction 
```bash
/home/sl/sl/lerobot/dataset
├── pick_sponge # pick sponge and place it into the plate (with 100 episode)
│   ├── data
│   │   └── chunk-000
│   │       ├── file-000.parquet
│   │       └── file-001.parquet
│   ├── meta
│   │   ├── episodes
│   │   │   └── chunk-000
│   │   ├── info.json
│   │   ├── stats.json
│   │   └── tasks.parquet
│   └── videos
│       ├── observation.images.low
│       │   └── chunk-000
│       ├── observation.images.right
│       │   └── chunk-000
│       └── observation.images.top
│           └── chunk-000
├── pusht # push T block into the T marker (with 100 episode)
   ├── data
   │   └── chunk-000
   │       └── file-000.parquet
   ├── meta
   │   ├── episodes
   │   │   └── chunk-000
   │   ├── info.json
   │   ├── stats.json
   │   └── tasks.parquet
   └── videos
       ├── observation.images.low
       │   └── chunk-000
       ├── observation.images.right
       │   └── chunk-000
       └── observation.images.top
           └── chunk-000


52 directories, 21 files
```

#  Huggingface LerobotDataset and Robot Learning: A Tutorial
```bash
https://huggingface.co/spaces/lerobot/robot-learning-tutorial

- LeRobotDataset (Importance; 重点看)
- Robot (Reinforcement) Learning
  - A (Concise) Introduction to RL
  - Real-world RL for Robotics
  - Code Example- Real-world RL
  - Limitations of RL in Real-World Robotics: Simulators and Reward Design

- Robot (Imitation) Learning
  - A (Concise) Introduction to Generative Models
  - Variational Auto-Encoders
  - Diffusion Models
  - Flow Matching
  - Action Chunking with Transformers
  - Code Example: Training and Using ACT in Practice
  - Diffusion Policy
  - Code Example: Training and Using Diffusion Policies in Practice
  - Optimized Inference
  - Code Example- Using Async Inference
```
# ViperX dataset from EPFL robot platform

```bash
https://moon-ci-docs.huggingface.co/docs/lerobot/pr_2123/en/integrate_hardware#looking-for-an-example-

<!-- Task： control robot to push T block -->

inside /data/sichao/lerobot/dataset/pusht/meta/info.json

Para "features" and "observation.state" inddicate the robot state, and "observation.images.top","observation.images.low","observation.images.right", which mean "get_observation()" These typically include motor states, camera frames. 

# # ViperX robot

Go to /data/sichao/lerobot/src/lerobot/robots/viperx/viperx.py

get_observation function collect motor states and camera frames

send_action fucntion send robot motor states. 
```


# Real-to-Sim and Policy training

```bash
# Policy training used imitation learning policies: ACT, DP, Pi-0, and SmolVLA. The real-world setup consists of a single UFactory xArm 7 robot arm equipped with two calibrated Intel RealSense RGB-D cameras: a D405 mounted on the robot wrist and a D455 mounted on the table as a fixed external camera. All policies take as input images from both "camera views, along with the current end-effector state". For "push-T", the end-effector state includes only the 2D position (x,y); for the other tasks, it additionally includes the position, rotation, and gripper openness. Across all tasks, we collect 39-60 successful demonstrations via teleoperation using GELLO [85]. Training is performed using the open-source LeRobot [86] implementation, except for Pi-0, where we adopt the original implementation [3] for better performance.

#  Real-to-Sim Robot Policy Evaluation with Gaussian Splatting Simulation of Soft-Body Interactions github repository: https://github.com/kywind/real2sim-eval

policy training github repository:  https://github.com/shuosha/policy_training 

# policy_training/configs/inference

configs/inference/pusht.json 
  - "action_dim": 2, # mean 2D posistion (x,y), relect robot state is 2d 

configs/inference/insert_rope.json
  - "action_dim": 8, # mean 8D posistion (x_pos,y_pos,z_pos,q_x,q_y,q_z,q_w,gripper_qpos)

/data/sichao/.cache/huggingface/lerobot/shashuo0104/xarm7_insert_rope/meta/info.jsonl shows 
    observation.state	
    dtype	"float32"
    shape	
    0	8
    names	
    waypoint	
    0	"x_pos"
    1	"y_pos"
    2	"z_pos"
    3	"q_x"
    4	"q_y"
    5	"q_z"
    6	"q_w"
    7	"gripper_qpos"

# policy_training/configs/training 
configs/training/act_pusht.json
  "observation.state": {
                  "type": "STATE",
                  "shape": [
                      2
                  ]
              }
          },

configs/training/act_insert_rope.json 

  "observation.state": {
                  "type": "STATE",
                  "shape": [
                      8
                  ]
              }
          },

https://github.com/shuosha/policy_training/blob/main/configs/training/act_pack_sloth.json

configs/training/act_pack_sloth.json

# If you need to knwo the exact dataset for policy training, go to https://github.com/shuosha/policy_training

Training Your Own Models has the dataset. 

you can compare the dataset collected by policy training and LeRobotDataset "lerobot/dataset/pusht" download from https://kth-my.sharepoint.com/:u:/g/personal/sicliu_ug_kth_se/IQA-YrMzIa4IQpaD1MQ0HsWFAYLUSh8d3kGyZHJNFtmaA-c?e=fxczgd , repsectovely.

```
# LeRobotDataset v3.0 详细解析（完整讲解版）

```bash
# LeRobotDataset v3.0 是 Hugging Face LeRobot 框架中的 统一机器人数据格式（新一代标准），用于表示从多机器人、多模态系统采集的数据。它是专门为 模仿学习、泛化策略、跨机器人共享而设计的轻量结构

# 1. LeRobotDataset v3 = “三个核心概念”

v3.0 格式本质上是：
一个序列（episode） → 由多个 step 组成 → 每个 step 包含 obs（状态）+ action（动作）
结构极简、统一、类似 RLDS / D4RL，但更适合真实机器人。

# 2. 数据结构总览
# v3 数据集由下列组成：
/data/sichao/.cache/huggingface/lerobot/lerobot/pick_sponge
├── data
│   └── chunk-000
│       └── file-001.parquet
├── meta
│   ├── episodes
│   │   └── chunk-000
│   │       └── file-001.parquet
│   ├── info.json
│   ├── stats.json
│   └── tasks.parquet
└── videos
    ├── observation.images.low
    │   └── chunk-000
    │       └── file-001.mp4
    ├── observation.images.right
    │   └── chunk-000
    │       └── file-001.mp4
    └── observation.images.top
        └── chunk-000
            └── file-001.mp4

# 3. v3 的作用与优势（为什么 LeRobot 都迁移到 v3）
3.1 极简数据接口
只需要 “obs + action”，不再需要复杂类型。
3.2 可扩展的模态系统
每个 obs 模态都独立，比如：
RGB
Depth
Segmentation
Joint states
Force/torque
Language instruction
Point cloud
Camera pose
Robot state（FK/EE pose）
3.3 “Lazy loading”（按需加载）# Huggingface LerobotDataset and Robot Learning: A Tutorial
```bash
https://huggingface.co/spaces/lerobot/robot-learning-tutorial

- LeRobotDataset (Importance; 重点看)
- Robot (Reinforcement) Learning
  - A (Concise) Introduction to RL
  - Real-world RL for Robotics
  - Code Example- Real-world RL
  - Limitations of RL in Real-World Robotics: Simulators and Reward Design

- Robot (Imitation) Learning
  - A (Concise) Introduction to Generative Models
  - Variational Auto-Encoders
  - Diffusion Models
  - Flow Matching
  - Action Chunking with Transformers
  - Code Example: Training and Using ACT in Practice
  - Diffusion Policy
  - Code Example: Training and Using Diffusion Policies in Practice
  - Optimized Inference
  - Code Example- Using Async Inference
```
# ViperX dataset from EPFL robot platform

```bash
https://moon-ci-docs.huggingface.co/docs/lerobot/pr_2123/en/integrate_hardware#looking-for-an-example-

<!-- Task： control robot to push T block -->

inside /data/sichao/lerobot/dataset/pusht/meta/info.json

Para "features" and "observation.state" inddicate the robot state, and "observation.images.top","observation.images.low","observation.images.right", which mean "get_observation()" These typically include motor states, camera frames. 

# # ViperX robot

Go to /data/sichao/lerobot/src/lerobot/robots/viperx/viperx.py

get_observation function collect motor states and camera frames

send_action fucntion send robot motor states. 
```


# Real-to-Sim and Policy training

```bash
# Policy training used imitation learning policies: ACT, DP, Pi-0, and SmolVLA. The real-world setup consists of a single UFactory xArm 7 robot arm equipped with two calibrated Intel RealSense RGB-D cameras: a D405 mounted on the robot wrist and a D455 mounted on the table as a fixed external camera. All policies take as input images from both "camera views, along with the current end-effector state". For "push-T", the end-effector state includes only the 2D position (x,y); for the other tasks, it additionally includes the position, rotation, and gripper openness. Across all tasks, we collect 39-60 successful demonstrations via teleoperation using GELLO [85]. Training is performed using the open-source LeRobot [86] implementation, except for Pi-0, where we adopt the original implementation [3] for better performance.

#  Real-to-Sim Robot Policy Evaluation with Gaussian Splatting Simulation of Soft-Body Interactions github repository: https://github.com/kywind/real2sim-eval

policy training github repository:  https://github.com/shuosha/policy_training 

# policy_training/configs/inference

configs/inference/pusht.json 
  - "action_dim": 2, # mean 2D posistion (x,y), relect robot state is 2d 

configs/inference/insert_rope.json
  - "action_dim": 8, # mean 8D posistion (x_pos,y_pos,z_pos,q_x,q_y,q_z,q_w,gripper_qpos)

/data/sichao/.cache/huggingface/lerobot/shashuo0104/xarm7_insert_rope/meta/info.jsonl shows 
    observation.state	
    dtype	"float32"
    shape	
    0	8
    names	
    waypoint	
    0	"x_pos"
    1	"y_pos"
    2	"z_pos"
    3	"q_x"
    4	"q_y"
    5	"q_z"
    6	"q_w"
    7	"gripper_qpos"

# policy_training/configs/training 
configs/training/act_pusht.json
  "observation.state": {
                  "type": "STATE",
                  "shape": [
                      2
                  ]
              }
          },

configs/training/act_insert_rope.json 

  "observation.state": {
                  "type": "STATE",
                  "shape": [
                      8
                  ]
              }
          },

https://github.com/shuosha/policy_training/blob/main/configs/training/act_pack_sloth.json

configs/training/act_pack_sloth.json

# If you need to knwo the exact dataset for policy training, go to https://github.com/shuosha/policy_training

Training Your Own Models has the dataset. 

you can compare the dataset collected by policy training and LeRobotDataset "lerobot/dataset/pusht" download from https://kth-my.sharepoint.com/:u:/g/personal/sicliu_ug_kth_se/IQA-YrMzIa4IQpaD1MQ0HsWFAYLUSh8d3kGyZHJNFtmaA-c?e=fxczgd , repsectovely.

```
# LeRobotDataset v3.0 详细解析（完整讲解版）

```bash
# 中文介绍
https://huggingface.co/blog/zh/lerobot-release-v040

# LeRobotDataset v3.0 是 Hugging Face LeRobot 框架中的 统一机器人数据格式（新一代标准），用于表示从多机器人、多模态系统采集的数据。它是专门为 模仿学习、泛化策略、跨机器人共享而设计的轻量结构

# 1. LeRobotDataset v3 = “三个核心概念”

v3.0 格式本质上是：
一个序列（episode） → 由多个 step 组成 → 每个 step 包含 obs（状态）+ action（动作）
结构极简、统一、类似 RLDS / D4RL，但更适合真实机器人。

# 2. 数据结构总览
# v3 数据集由下列组成：
/data/sichao/.cache/huggingface/lerobot/lerobot/pick_sponge
├── data
│   └── chunk-000
│       └── file-001.parquet
├── meta
│   ├── episodes
│   │   └── chunk-000
│   │       └── file-001.parquet
│   ├── info.json
│   ├── stats.json
│   └── tasks.parquet
└── videos
    ├── observation.images.low
    │   └── chunk-000
    │       └── file-001.mp4
    ├── observation.images.right
    │   └── chunk-000
    │       └── file-001.mp4
    └── observation.images.top
        └── chunk-000
            └── file-001.mp4

# 3. v3 的作用与优势（为什么 LeRobot 都迁移到 v3）
3.1 极简数据接口
只需要 “obs + action”，不再需要复杂类型。
3.2 可扩展的模态系统
每个 obs 模态都独立，比如：
RGB
Depth
Segmentation
Joint states
Force/torque
Language instruction
Point cloud
Camera pose
Robot state（FK/EE pose）
3.3 “Lazy loading”（按需加载）
不会一次性把所有图像加载到内存。
3.4 强调一致性和跨机器人共享
例如：
训练 ACT/RT-2 风格策略
多机器人政策合并
不同机器人数据可以共享（aloha + UR5+ panda）

# 4. Episode 结构（最重要部分）
每个 episode 包含：
项目	含义
meta.json	episode 级元信息（机器人类型、相机、任务等）
timestamps.json	每个 step 的时间戳（同步用）
/obs/	所有观测模态（每个模态可有子文件夹）
/action/	动作序列（一般 shape = [T, action_dim]）
# 5. 观测 obs 的格式（最关键）
obs = 一个 dict：每个 key 表示一个模态
例如：
obs = {
  "camera_front_image": ...,
  "camera_wrist_depth": ...,
  "joint_positions": ...,
  "joint_velocities": ...,
  "ee_pose": ...,
  "gripper_open": ...
}
v3 中 每个模态必须是 step-major 的列表
即：
camera_front_image = ["000001.jpeg", "000002.jpeg", ...]
joint_positions = [[...], [...], ...]

# 6. 动作 action 的格式
动作必须是一个矩阵：
actions.shape = (T, action_dim)
例如对于 Interbotix ViperX / WidowX：
action_dim = n_joints + 1 (gripper)
对于 UR5：
n_dim = 6 或 7

# 7. metadata（元信息）
meta.json 中常包含：
{
  "episode_id": "episode_0001",
  "robot": "ur5",
  "control_frequency": 30,
  "action_dim": 7,
  "cameras": {
     "wrist": { "intrinsics": [...], "extrinsics": [...] },
     "overhead": { ... }
  },
  "notes": ""
}

手眼标定外参、D405/D435 内参可以直接写入这里，非常标准。

# 8. timestamps.json

确保所有模态同步，例如：
{
  "camera_front_image": [...timestamps...],
  "camera_wrist_image": [...timestamps...],
  "joint_positions": [...],
  "actions": [...]
}

用于：
插值同步
对齐 Realsense + robot state

# 9. statistics.json
用于训练：
{
  "mean": {
    "joint_positions": [...],
    "joint_velocities": [...],
    "action": [...]
  },
  "std": {...}
}
在训练时自动 normalize。

# 10. LeRobot 如何加载 v3 数据集？

只需要：

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset(path="path/to/dataset", split="train")
sample = dataset[0]


你会得到：

{
  "obs": {...},
  "action": ...,
  "next_obs": ...
}

# 11. v3 的典型训练流程（你训练 ACT、PI0、PI05 都用这个）
dataset = LeRobotDataset(...)
train_dataloader = DataLoader(dataset, ...)
policy = ACT/PI0/PI05(...)
policy.train(train_dataloader)

# 12. 结合你的实际情况（UR5 + RealSense + Interbotix）
采集系统完全可以无缝对接 v3：
可以有如下模态：
obs = {
  "wrist_rgb": /obs/wrist/rgb/*.jpeg
  "wrist_depth": /obs/wrist/depth/*.png
  "static_rgb": /obs/static/rgb/*.jpeg
  "joint_pos": np.ndarray[T, 6]
  "joint_vel": np.ndarray[T, 6]
  "ee_pose": np.ndarray[T, 7]
  "gripper": np.ndarray[T, 1]
}

# 13. LeRobotDataset v3 的最佳实践（我基于你现在的项目总结）

使用统一时间戳（你现在 RealSense + UR5 必须同步）

动作必须是连续的（不要写 None/NaN）

图像路径使用相对路径（防止移动 dataset）

尽量写入相机内参外参（你正在做手眼标定）

保存中英文任务描述（用于语言模型训练）

保存 pointmap（你现在用 D405 扫描物体，非常重要）

例如：

obs/pointcloud/*.ply
obs/pointmap/*.npy

# 总结（一句话版本）

LeRobotDataset v3 就是：一个统一、轻量、多模态的真实机器人数据格式，用于训练 ACT / PI / Diffusion / VLA、支持任意相机、任意机器人、任意模态，适合你现在 UR5 + RealSense 的全流程。
```

不会一次性把所有图像加载到内存。
3.4 强调一致性和跨机器人共享
例如：
训练 ACT/RT-2 风格策略
多机器人政策合并
不同机器人数据可以共享（aloha + UR5+ panda）

# 4. Episode 结构（最重要部分）
每个 episode 包含：
项目	含义
meta.json	episode 级元信息（机器人类型、相机、任务等）
timestamps.json	每个 step 的时间戳（同步用）
/obs/	所有观测模态（每个模态可有子文件夹）
/action/	动作序列（一般 shape = [T, action_dim]）
# 5. 观测 obs 的格式（最关键）
obs = 一个 dict：每个 key 表示一个模态
例如：
obs = {
  "camera_front_image": ...,
  "camera_wrist_depth": ...,
  "joint_positions": ...,
  "joint_velocities": ...,
  "ee_pose": ...,
  "gripper_open": ...
}
v3 中 每个模态必须是 step-major 的列表
即：
camera_front_image = ["000001.jpeg", "000002.jpeg", ...]
joint_positions = [[...], [...], ...]

# 6. 动作 action 的格式
动作必须是一个矩阵：
actions.shape = (T, action_dim)
例如对于 Interbotix ViperX / WidowX：
action_dim = n_joints + 1 (gripper)
对于 UR5：
n_dim = 6 或 7

# 7. metadata（元信息）
meta.json 中常包含：
{
  "episode_id": "episode_0001",
  "robot": "ur5",
  "control_frequency": 30,
  "action_dim": 7,
  "cameras": {
     "wrist": { "intrinsics": [...], "extrinsics": [...] },
     "overhead": { ... }
  },
  "notes": ""
}

手眼标定外参、D405/D435 内参可以直接写入这里，非常标准。

# 8. timestamps.json

确保所有模态同步，例如：
{
  "camera_front_image": [...timestamps...],
  "camera_wrist_image": [...timestamps...],
  "joint_positions": [...],
  "actions": [...]
}

用于：
插值同步
对齐 Realsense + robot state

# 9. statistics.json
用于训练：
{
  "mean": {
    "joint_positions": [...],
    "joint_velocities": [...],
    "action": [...]
  },
  "std": {...}
}
在训练时自动 normalize。

# 10. LeRobot 如何加载 v3 数据集？

只需要：

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset(path="path/to/dataset", split="train")
sample = dataset[0]


你会得到：

{
  "obs": {...},
  "action": ...,
  "next_obs": ...
}

# 11. v3 的典型训练流程（你训练 ACT、PI0、PI05 都用这个）
dataset = LeRobotDataset(...)
train_dataloader = DataLoader(dataset, ...)
policy = ACT/PI0/PI05(...)
policy.train(train_dataloader)

# 12. 结合你的实际情况（UR5 + RealSense + Interbotix）
采集系统完全可以无缝对接 v3：
可以有如下模态：
obs = {
  "wrist_rgb": /obs/wrist/rgb/*.jpeg
  "wrist_depth": /obs/wrist/depth/*.png
  "static_rgb": /obs/static/rgb/*.jpeg
  "joint_pos": np.ndarray[T, 6]
  "joint_vel": np.ndarray[T, 6]
  "ee_pose": np.ndarray[T, 7]
  "gripper": np.ndarray[T, 1]
}

# 13. LeRobotDataset v3 的最佳实践（我基于你现在的项目总结）

使用统一时间戳（你现在 RealSense + UR5 必须同步）

动作必须是连续的（不要写 None/NaN）

图像路径使用相对路径（防止移动 dataset）

尽量写入相机内参外参（你正在做手眼标定）

保存中英文任务描述（用于语言模型训练）

保存 pointmap（你现在用 D405 扫描物体，非常重要）

例如：

obs/pointcloud/*.ply
obs/pointmap/*.npy

# 总结（一句话版本）

LeRobotDataset v3 就是：一个统一、轻量、多模态的真实机器人数据格式，用于训练 ACT / PI / Diffusion / VLA、支持任意相机、任意机器人、任意模态，适合你现在 UR5 + RealSense 的全流程。
```
