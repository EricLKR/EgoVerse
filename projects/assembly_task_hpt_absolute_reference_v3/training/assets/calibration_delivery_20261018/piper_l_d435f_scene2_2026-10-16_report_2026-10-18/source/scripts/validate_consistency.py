#!/usr/bin/env python3
"""Read-only new-pose consistency checks using a frozen eye-on-base calibration."""
import argparse
import fcntl
import hashlib
import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rclpy
import yaml
from rclpy.time import Time
from scipy.spatial.transform import Rotation
from tf2_ros import Buffer, TransformException, TransformListener

from print_tag_position import BASE_FRAME, CAMERA_FRAME, TAG_FRAME, load_calibration

EFFECTOR_FRAME = 'link6'
FORMAT = 'piper_handeye_consistency_v1'


def checked_matrix(value):
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError('Transform must be a finite 4x4 matrix')
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-9):
        raise ValueError('Invalid homogeneous transform')
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(rotation), 1, atol=1e-6):
        raise ValueError('Invalid rotation matrix')
    return matrix


def tf_matrix(transform):
    t, q = transform.translation, transform.rotation
    quaternion = np.array([q.x, q.y, q.z, q.w])
    if not np.all(np.isfinite(quaternion)) or abs(np.linalg.norm(quaternion) - 1) > 1e-3:
        raise ValueError('Invalid TF quaternion')
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
    matrix[:3, 3] = [t.x, t.y, t.z]
    return checked_matrix(matrix)


def installation_transform(base_hand, base_camera, camera_tag):
    return np.linalg.inv(checked_matrix(base_hand)) @ checked_matrix(base_camera) @ checked_matrix(camera_tag)


def residual(reference, value):
    reference, value = checked_matrix(reference), checked_matrix(value)
    delta_mm = (value[:3, 3] - reference[:3, 3]) * 1000
    degrees = np.degrees(Rotation.from_matrix(reference[:3, :3].T @ value[:3, :3]).magnitude())
    return delta_mm, float(np.linalg.norm(delta_mm)), float(degrees)


def read_session(path):
    data = json.loads(path.read_text())
    if data.get('format') != FORMAT or not data.get('poses'):
        raise ValueError('Not a supported consistency session')
    checked_matrix(data['T_base_cam'])
    for pose in data['poses']:
        expected = installation_transform(pose['T_base_link6'], data['T_base_cam'], pose['T_cam_tag'])
        if not np.allclose(expected, checked_matrix(pose['T_link6_tag']), atol=1e-9):
            raise ValueError('Session transform does not match its recorded inputs')
    return data


def atomic_write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def render_report(data):
    poses = data['poses']
    reference = poses[0]['T_link6_tag']
    rows, positions, angles = [], [], []
    for index, pose in enumerate(poses):
        delta, position, angle = residual(reference, pose['T_link6_tag'])
        label = str(pose['label']).replace('|', '/').replace('\n', ' ').replace('\r', ' ')
        rows.append(f'| {index + 1} | {label} | {delta[0]:.3f} | {delta[1]:.3f} | {delta[2]:.3f} | {position:.3f} | {angle:.3f} |')
        if index:
            positions.append(position)
            angles.append(angle)
    lines = [
        '# 新姿态一致性验证', '',
        f"- 外参文件：`{data['calibration_file']}`",
        f"- 外参 SHA-256：`{data['calibration_sha256']}`",
        f'- 参考：第 1 个新姿态；验证姿态数：{len(poses) - 1}。参考姿态不计入汇总。',
        '- 计算：`T_link6_tag = inverse(T_base_link6) × T_base_cam × T_cam_tag`。',
        '- 每点两组 TF 使用同一 Tag 时间戳；ΔXYZ 在 link6 系，单位 mm；旋转差为相对旋转角，单位 °。',
        '- 本报告衡量固定 Tag 安装变换的一致性，包含视觉、机器人与参考姿态噪声；没有绝对真值，不声明绝对精度或自动达标。', '',
        '| 点 | 标签 | ΔX mm | ΔY mm | ΔZ mm | 位置差 mm | 角度差 ° |',
        '|---|---|---:|---:|---:|---:|---:|', *rows, '',
    ]
    if positions:
        lines += ['| 指标 | 位置差 mm | 角度差 ° |', '|---|---:|---:|']
        for name, function in [('RMS', lambda a: np.sqrt(np.mean(np.square(a)))), ('中位数', np.median), ('P95', lambda a: np.percentile(a, 95)), ('最大值', np.max)]:
            lines.append(f'| {name} | {function(positions):.3f} | {function(angles):.3f} |')
    else:
        lines.append('仅建立参考，尚无验证姿态，不能计算误差汇总。')
    return '\n'.join(lines) + '\n'


def report_path(session):
    return session.with_suffix('.report.md')


def capture_pair(timeout, max_age):
    rclpy.init()
    node = rclpy.create_node('handeye_consistency_reader')
    buffer = Buffer()
    listener = TransformListener(buffer, node)
    deadline = time.monotonic() + timeout
    last_error = 'No fresh Tag TF'
    try:
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
            try:
                tag = buffer.lookup_transform(CAMERA_FRAME, TAG_FRAME, Time())
                stamp = Time.from_msg(tag.header.stamp)
                age = (node.get_clock().now().nanoseconds - stamp.nanoseconds) / 1e9
                if stamp.nanoseconds <= 0 or not 0 <= age <= max_age:
                    last_error = f'Tag TF is stale or future-dated (age={age:.3f}s)'
                    continue
                hand = buffer.lookup_transform(BASE_FRAME, EFFECTOR_FRAME, stamp)
                # Recheck freshness after TF lookup, using the exact same time for both.
                age = (node.get_clock().now().nanoseconds - stamp.nanoseconds) / 1e9
                if not 0 <= age <= max_age:
                    continue
                return tf_matrix(hand.transform), tf_matrix(tag.transform), stamp.nanoseconds, age
            except TransformException as error:
                last_error = str(error)
        raise ValueError(f'未保存姿态：{last_error}。检查 Tag 可见性和机器人 TF，停稳后重试。')
    finally:
        listener.unregister()
        node.destroy_node()
        rclpy.try_shutdown()


def capture(args):
    source = args.calibration.resolve()
    session = args.session.resolve()
    if source in (session, report_path(session)):
        raise ValueError('验证输出不能覆盖外参文件')
    original = source.read_bytes()
    base_camera = load_calibration(source)
    if source.read_bytes() != original:
        raise ValueError('加载时外参文件发生改变，请重试')
    digest = hashlib.sha256(original).hexdigest()
    session.parent.mkdir(parents=True, exist_ok=True)
    with session.with_suffix('.json.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('同一验证文件有另一个采集正在运行') from None
        if session.exists():
            data = read_session(session)
            if data['calibration_sha256'] != digest or not np.allclose(data['T_base_cam'], base_camera, atol=1e-12):
                raise ValueError('本次外参与验证会话不一致；请使用原文件或新建 --session')
        else:
            data = dict(format=FORMAT, calibration_file=str(source), calibration_sha256=digest,
                        T_base_cam=base_camera.tolist(), poses=[])
        hand, tag, stamp, age = capture_pair(args.timeout, args.max_age)
        if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
            raise ValueError('外参文件在采集期间改变，未保存此点')
        if data['poses'] and stamp <= data['poses'][-1]['tag_stamp_ns']:
            raise ValueError('TF 时间戳未前进，未保存此点；检查时钟或新建验证会话')
        installed = installation_transform(hand, base_camera, tag)
        pose = dict(label=args.label or f"pose_{len(data['poses']) + 1:02d}",
                    recorded_at=datetime.now(timezone.utc).isoformat(), tag_stamp_ns=stamp,
                    tag_age_s=age, T_base_link6=hand.tolist(), T_cam_tag=tag.tolist(),
                    T_link6_tag=installed.tolist())
        data['poses'].append(pose)
        atomic_write(session, json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        atomic_write(report_path(session), render_report(data))
        delta, position, angle = residual(data['poses'][0]['T_link6_tag'], installed)
        print(f"已保存第 {len(data['poses'])} 点 ({pose['label']})；TF age={age:.3f}s")
        if len(data['poses']) == 1:
            print('该点为固定参考；后续新姿态不改变此参考。')
        else:
            print(f'ΔXYZ [mm]: {delta.round(3).tolist()}；位置差 {position:.3f} mm；角度差 {angle:.3f}°')
        print(f'数据：{session}\n报告：{report_path(session)}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='command', required=True)
    collect = subparsers.add_parser('capture', help='停稳后读取一个新姿态；首次建立固定参考')
    collect.add_argument('--calibration', required=True, type=Path)
    collect.add_argument('--session', required=True, type=Path, help='单独的验证 JSON 文件；同名文件继续追加')
    collect.add_argument('--label', default='')
    collect.add_argument('--timeout', type=float, default=10)
    collect.add_argument('--max-age', type=float, default=0.5, help='允许的 TF 数据年龄，不是精度门限')
    summary = subparsers.add_parser('report', help='离线查看已保存数据的误差，不启动 ROS 节点')
    summary.add_argument('--session', required=True, type=Path)
    args = parser.parse_args()
    if args.session.suffix != '.json':
        parser.error('--session 必须使用 .json 后缀')
    if args.command == 'capture':
        for name in ('timeout', 'max_age'):
            if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
                parser.error(f'--{name.replace("_", "-")} 必须为正的有限值')
    try:
        if args.command == 'capture':
            capture(args)
        else:
            print(render_report(read_session(args.session)), end='')
        return 0
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError, KeyError, TypeError, AttributeError, yaml.YAMLError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    raise SystemExit(main())
