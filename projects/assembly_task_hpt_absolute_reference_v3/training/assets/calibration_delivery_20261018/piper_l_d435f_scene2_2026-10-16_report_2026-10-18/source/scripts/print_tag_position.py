#!/usr/bin/env python3
"""读取一次新鲜的 AprilTag TF，打印其中心在相机系和机械臂基座系中的坐标。

依赖本工程已启动的 RealSense 和 apriltag_ros 节点，不直接打开相机设备。
程序不连接机械臂、不发送运动指令、不发布 TF，也不添加目标位置偏移。
"""
# 标准库：argparse 解析命令行参数，time 提供等待超时所需的单调时钟。
import argparse
import time
from pathlib import Path
import yaml

# NumPy 处理向量和矩阵；rclpy 提供 ROS 2 节点及消息回调。
import numpy as np
import rclpy
# ROS Time 用于查询 TF；Rotation 将四元数转换为旋转矩阵。
from rclpy.time import Time
from scipy.spatial.transform import Rotation
# Buffer 保存收到的变换，TransformListener 订阅 TF，异常类型用于处理查询失败。
from tf2_ros import Buffer, TransformException, TransformListener


# 坐标系名称须与相机和 AprilTag 节点发布的 TF 一致。
# 相机 optical frame 的轴向为：X 向右、Y 向下、Z 向前。
CAMERA_FRAME = "camera_color_optical_frame"
# tag_0 的原点位于 ID 0 标记的中心。
TAG_FRAME = "tag_0"
# 基座坐标只通过下面的固定外参计算，因此无需启动机器人 TF 节点。
BASE_FRAME = "base_link"

def load_calibration(path):
    """读取明确指定的新标定文件；拒绝方向或坐标系不匹配的结果。"""
    data = yaml.safe_load(Path(path).read_text())
    params = data["parameters"]
    expected = {
        "calibration_type": "eye_on_base",
        "robot_base_frame": BASE_FRAME,
        "robot_effector_frame": "link6",
        "tracking_base_frame": CAMERA_FRAME,
        "tracking_marker_frame": TAG_FRAME,
    }
    for key, value in expected.items():
        if params.get(key) != value:
            raise ValueError(f"{key} 应为 {value}，实际为 {params.get(key)}")
    transform = data["transform"]
    translation = np.array([transform["translation"][k] for k in "xyz"], dtype=float)
    quaternion = np.array([transform["rotation"][k] for k in "xyzw"], dtype=float)
    if not np.all(np.isfinite(translation)) or not np.all(np.isfinite(quaternion)):
        raise ValueError("标定文件包含非有限值")
    if abs(np.linalg.norm(quaternion) - 1.0) > 1e-3:
        raise ValueError("标定四元数不是单位四元数")
    # eye_on_base 的原生结果就是 T_base_cam，无需取逆；平移单位为米。
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
    matrix[:3, 3] = translation
    return matrix


def camera_to_base(point, transform):
    """p_base = R_base_cam @ p_cam + t_base_cam；输入和输出均为米。"""
    return transform[:3, :3] @ np.asarray(point, dtype=float) + transform[:3, 3]


def main():
    """等待有效检测，打印两组坐标，然后释放 ROS 资源并退出。"""
    # 默认最多等待 10 秒；可用 --timeout 指定其他等待时间。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", required=True, type=Path,
                        help="当前相机安装位置重新标定后保存的 .calib 文件")
    parser.add_argument("--timeout", type=float, default=10.0,
                        help="Seconds to wait for a fresh detection (default: 10).")
    args = parser.parse_args()
    # 拒绝负数、零、无穷大及 NaN，避免无效的等待期限。
    if not np.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be positive and finite")

    # 在连接 ROS 前检查结果文件；不默认加载任何旧外参。
    try:
        base_camera = load_calibration(args.calibration)
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as exc:
        parser.error(f"无法读取标定结果：{exc}")

    # 初始化 ROS 2，创建只用于接收 TF 的节点及变换缓存。
    rclpy.init()
    node = rclpy.create_node("print_tag_position")
    buffer = Buffer()
    listener = TransformListener(buffer, node)
    # 超时使用单调时钟，不受系统校时导致的时间跳变影响。
    deadline = time.monotonic() + args.timeout
    try:
        while rclpy.ok() and time.monotonic() < deadline:
            # 处理订阅回调，让 TransformListener 将新 TF 写入缓存。
            rclpy.spin_once(node, timeout_sec=0.1)
            try:
                # 参数顺序是目标坐标系、源坐标系、查询时刻。
                # Time() 表示查询最新可用变换，得到 T_camera_tag。
                # 其平移分量就是 Tag 原点（中心）在相机系下的坐标。
                transform = buffer.lookup_transform(CAMERA_FRAME, TAG_FRAME, Time())
            except TransformException:
                # 刚启动或尚未检测到 Tag 时可能没有 TF，继续等候即可。
                continue
            # 计算 TF 时间戳距当前 ROS 时间的秒数；1 秒等于 10^9 纳秒。
            stamp = transform.header.stamp
            age = (node.get_clock().now().nanoseconds
                   - stamp.sec * 1_000_000_000 - stamp.nanosec) / 1e9
            # Tag 离开视野后，TF 缓存仍可能保留旧检测结果。
            # 只接受最近 0.5 秒内的 TF，并排除时间戳在未来的记录。
            if not 0 <= age <= 0.5:
                continue
            # 提取 Tag 中心在相机系中的 XYZ，单位为米。
            t = transform.transform.translation
            camera_point = np.array([t.x, t.y, t.z])
            if not np.all(np.isfinite(camera_point)):
                # 检测结果若含 NaN 或无穷大，就跳过这条记录。
                continue
            # 用固定手眼外参计算同一个 Tag 中心在基座系中的坐标。
            # 此处没有加入 +Z 方向 1 cm，也没有做夹指到法兰的偏移换算。
            base_point = camera_to_base(camera_point, base_camera)
            # 打印检测时间和数据年龄，便于判断输出是否来自当前观测。
            print(f"Tag: {TAG_FRAME}; stamp: {stamp.sec}.{stamp.nanosec:09d}; age: {age:.3f} s")
            # 两组坐标分别以米和毫米显示；换算关系为 1 m = 1000 mm。
            for frame, point in [(CAMERA_FRAME, camera_point), (BASE_FRAME, base_point)]:
                print(f"{frame} [m]:  x={point[0]:.6f}, y={point[1]:.6f}, z={point[2]:.6f}")
                print(f"{frame} [mm]: x={point[0]*1000:.3f}, y={point[1]*1000:.3f}, z={point[2]*1000:.3f}")
            # 得到一条有效检测后立即成功退出，不连续采样。
            return 0
        # 等待结束仍未获得有效检测时返回失败，不输出旧坐标作为替代。
        print("No fresh tag_0 TF received. Check camera/apriltag nodes and tag visibility.")
        return 1
    except KeyboardInterrupt:
        # 用户按 Ctrl+C 时退出，130 是此类中断常用的退出码。
        return 130
    finally:
        # 无论成功、超时还是中断，都取消 TF 订阅并清理节点和 ROS 上下文。
        listener.unregister()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    # 直接执行此文件时运行 main，并把返回值作为进程退出码。
    # 被其他 Python 文件导入时不会创建 ROS 节点或读取硬件数据。
    raise SystemExit(main())
