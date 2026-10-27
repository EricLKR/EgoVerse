# 来源与必要的本地改动

| 源码 | 锁定 commit |
|---|---|
| https://github.com/agilexrobotics/piper_ros （humble） | `017ffefa64511bc6325bd77ddc4e16065c152051` |
| https://github.com/marcoesposito1988/easy_handeye2 | `29bd50a7939e853d768d203d2abc61eb6e50facc` |
| https://github.com/agilexrobotics/agx_arm_urdf （piper_l） | `f6642ce0d7872c686f29c99e9e10cd23d1d49313` |

原电脑已验证：ROS 2 Humble；realsense2_camera 4.55.1；apriltag_ros 3.4.0。当前电脑已迁移到 Jazzy，安装 realsense2_camera 4.58.4、apriltag_ros 3.4.0；apt 构建号未锁定；本机硬件运行已验证，新机器仍需现场验证。Python 依赖见 config/python-requirements.txt。

保留的改动：

- `piper_feedback.patch`：使用精确角度换算、关闭连接时初始化命令、按 Piper L 模型发布夹爪反馈。
- `agx_arm_description`：安装官方 Piper L 模型的轻量包装包，未改几何参数。
- `easy_handeye2_integration.patch`：仅使样本/外参保存路径可通过环境变量指定到本工程。

推荐运行 `bash scripts/fetch_sources.sh`，自动获取上述版本并幂等应用补丁。以下仅供理解手动操作；不要向已应用补丁的源码重复执行：

```bash
git -C ros2_ws/src/piper_ros apply ../../../docs/piper_feedback.patch
git -C ros2_ws/src/easy_handeye2 apply ../../../docs/easy_handeye2_integration.patch
bash scripts/build.sh
```

项目 calibrate.launch.py 直接启动上游 handeye_server 和 rqt_calibrator.py，没有自定义求解器。
没有包含上游启动文件附带的 dummy camera TF，避免它与 RealSense 的光学坐标链产生重复父节点。
GUI 与上游标定启动文件均已恢复为锁定 commit 的原版。

## Ubuntu 24.04 / Jazzy 移植

2026-10-13 用户明确选择本机 ROS 2 Jazzy。三个官方源码仍保持以上锁定 commit；Piper 官方源码原分支为 humble，本机已验证 Jazzy 构建和运行，未覆盖该上游仓库的其他控制功能。系统依赖改用 `ros-jazzy-*`；Python 环境使用系统 3.12，site-packages 路径由 sysconfig 解析。NumPy 1.26.4 / SciPy 1.11.4 提供 Python 3.12 Linux wheels；wrapt 从 1.13.3 升为 1.16.0，以支持 Python 3.12。SDK 与 GUI/算法不变。

本机 Jazzy 完整 setup/verify 已通过：rosdep 全部满足、6 包构建、服务/原生 GUI plugin/widget/SDK 导入、模型、GUI 资源离屏加载、标定文件及方向检查、launch 解析均通过，结果见 PROCESS.md。本机相机、从臂实时反馈、原生 GUI 采样与图像窗口已现场验证；新机器仍需现场验收。旧 Humble 验证不能替代。安装入口为 `scripts/install_system.sh` / `scripts/setup.sh`，非硬件验证入口为 `scripts/verify.sh`。

硬件运行调试补充：驱动选机 serial 校正为 337322071261；USB 描述符 serial 不用于选机。新增项目 Fast DDS 32 MiB SHM 配置，消除默认小段带来的图像丢帧；实际帧率和现场验收范围见 PROCESS.md。参考 [Fast DDS 2.14.7 Shared Memory Transport](https://fast-dds.docs.eprosima.com/en/v2.14.7/fastdds/transport/shared_memory/shared_memory.html)。

## Aria Gen 2 流程依赖（2026-10-17）

Aria 头部相机标定是独立的新流程，D435f 流程不变。projectaria-client-sdk 2.5.1 / projectaria-tools 2.3.0 有 cp312 wheel，装入同一 `ros2_ws/.venv`，桥接节点可以直接使用 rclpy。完整依赖会经 rerun-sdk 0.33 把 numpy 升到 2.x，破坏 Jazzy 系统模块（cv_bridge 等）所依赖的 numpy 1.26。因此按 `config/aria/python-requirements*.txt` 分两步安装：先装普通依赖，再 `--no-deps` 安装 Aria 包。这样不安装 rerun/jupyter 查看器，`pip check` 会报这些查看器依赖缺失，属预期。已验证 numpy 1.26.4 下 sdk_gen2、stream_receiver、去畸变、cv_bridge 均可用；scripts/verify.sh 与原有测试通过。
