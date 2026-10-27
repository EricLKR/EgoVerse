#!/usr/bin/env python3
"""Frozen-extrinsic corner checks: three mounting poses, seven held-out poses.

Only subscribes to rectified RGB, CameraInfo, AprilTag corners and robot TF.
No Tag pose TF enters the measured residual. PnP is used only to initialize
the single rigid mounting fit, which uses the first three poses exclusively.
"""
import argparse
from collections import OrderedDict
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import threading
import time

import cv2
import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, Image
from apriltag_msgs.msg import AprilTagDetectionArray
from tf2_ros import Buffer, TransformException, TransformListener

from print_tag_position import BASE_FRAME, CAMERA_FRAME, load_calibration
from validate_consistency import atomic_write, checked_matrix, tf_matrix

FORMAT = 'piper_corner_validation_v1'
SIZE = 0.042
POINTS = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], float) * SIZE / 2


def matrix(parameters):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_rotvec(parameters[:3]).as_matrix()
    result[:3, 3] = parameters[3:]
    return result


def parameters(transform):
    return np.r_[Rotation.from_matrix(transform[:3, :3]).as_rotvec(), transform[:3, 3]]


def project(camera_base, hand, mounting, intrinsics):
    transform = checked_matrix(camera_base) @ checked_matrix(hand) @ checked_matrix(mounting)
    points = POINTS @ transform[:3, :3].T + transform[:3, 3]
    if np.any(points[:, 2] <= 0.01):
        raise ValueError('Predicted Tag is behind/too close to camera')
    pixels = points @ np.asarray(intrinsics).T
    return pixels[:, :2] / pixels[:, 2:3]


def fit_mounting(camera_base, frames):
    """Fit six mounting parameters, never the camera extrinsic or held-out data."""
    def residual(value):
        installed = matrix(value)
        try:
            return np.concatenate([(project(camera_base, f['T_base_link6'], installed, f['K'])
                                    - np.asarray(f['corners'])).ravel() for f in frames])
        except ValueError:
            return np.full(len(frames) * 8, 1e6)
    candidates = []
    for frame in frames[::max(1, len(frames) // 3)]:
        ok, rotation, translation = cv2.solvePnP(POINTS, np.asarray(frame['corners'], float),
                                               np.asarray(frame['K'], float), None,
                                               flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            continue
        camera_tag = matrix(np.r_[rotation.ravel(), translation.ravel()])
        initial = np.linalg.inv(frame['T_base_link6']) @ np.linalg.inv(camera_base) @ camera_tag
        fit = least_squares(residual, parameters(initial), x_scale='jac', max_nfev=1000,
                            ftol=1e-11, xtol=1e-11, gtol=1e-11)
        if fit.success and np.all(np.isfinite(fit.x)) and np.max(np.abs(fit.fun)) < 1e6:
            candidates.append(fit)
    if not candidates:
        raise ValueError('Mounting fit failed; no validation verdict')
    fit = min(candidates, key=lambda candidate: np.sum(candidate.fun ** 2))
    singular = np.linalg.svd(fit.jac, compute_uv=False)
    if singular[-1] <= singular[0] * 1e-8:
        raise ValueError('Mounting fit is degenerate; use varied positions/orientations')
    return matrix(fit.x), dict(rms_corner_px=float(np.sqrt(np.mean(fit.fun.reshape(-1, 2)**2) * 2)),
                               jacobian_singular_values=singular.tolist(), fit_poses=3)


def error_values(data, pose):
    camera_base = np.linalg.inv(data['T_base_cam'])
    return np.concatenate([np.linalg.norm(project(camera_base, frame['T_base_link6'],
                           data['T_link6_tag'], frame['K']) - np.asarray(frame['corners']), axis=1)
                           for frame in pose['frames']])


def report(data):
    candidate = data.get('mode') == 'candidate_held_out'
    lines = ['# 四角重投影验证', '', f"- 外参 SHA256：`{data['calibration_sha256']}`",
             ('- 固定候选外参及随候选保存的 Tag 安装关系；本会话所有新姿态均为留出验证，不重新拟合。'
              if candidate else '- 固定原外参；前 3 个姿态仅拟合 Tag 安装关系，后 7 个为留出验证。'),
             '- 测量采用 apriltag_ros 的四角坐标；不使用每帧 Tag 三维姿态计算误差。',
             '- 单位 px；该结果包含角点、内参、Tag 几何及机器人反馈误差，不是毫米级绝对精度。', '']
    if data['T_link6_tag'] is None:
        return '\n'.join(lines + [f"已记录 {len(data['poses'])}/3 个安装参考，尚未冻结安装关系。", ''])
    lines += ['| 点 | 用途 | 标签 | 四角 RMS px | 最大 px |', '|---|---|---|---:|---:|']
    checks = []
    for index, pose in enumerate(data['poses'], 1):
        errors = error_values(data, pose)
        label = pose['label'].replace('|', '/').replace('\n', ' ')
        lines.append(f"| {index} | {pose['role']} | {label} | {np.sqrt(np.mean(errors**2)):.3f} | {np.max(errors):.3f} |")
        if pose['role'] == 'held_out':
            checks.extend(errors)
    if checks:
        lines += ['', f'留出汇总：RMS {np.sqrt(np.mean(np.square(checks))):.3f} px；'
                  f'中位数 {np.median(checks):.3f} px；最大 {np.max(checks):.3f} px。',
                  '安装参考不计入留出汇总；不自动宣布达标。']
    return '\n'.join(lines) + '\n'


class Reader:
    def __init__(self):
        rclpy.init()
        self.node = rclpy.create_node('frozen_corner_reader')
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self.node)
        self.lock = threading.Lock()
        self.images, self.infos, self.detections = OrderedDict(), OrderedDict(), OrderedDict()
        self.subscriptions = [self.node.create_subscription(kind, topic,
                              lambda msg, cache=cache: self.store(cache, msg), qos_profile_sensor_data)
                              for kind, topic, cache in [(Image, '/camera/color/image_rect', self.images),
                              (CameraInfo, '/camera/color/camera_info', self.infos),
                              (AprilTagDetectionArray, '/detections', self.detections)]]
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)
        self.thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.thread.start()

    def store(self, cache, message):
        stamp = Time.from_msg(message.header.stamp).nanoseconds
        with self.lock:
            cache[stamp] = message
            while len(cache) > 40:
                cache.popitem(last=False)

    def close(self):
        self.executor.shutdown(timeout_sec=3)
        self.thread.join(timeout=3)
        self.listener.unregister()
        self.node.destroy_node()
        rclpy.try_shutdown()

    def collect(self, timeout, check_only=False):
        deadline = time.monotonic() + timeout
        frames, last_image, last_error = [], None, 'Waiting for matched image/detection/TF'
        last_stamp = 0
        while time.monotonic() < deadline:
            time.sleep(0.01)
            with self.lock:
                stamps = sorted(set(self.images) & set(self.infos), reverse=True)
                if not check_only:
                    stamps = [s for s in stamps if s in self.detections]
                if not stamps:
                    continue
                stamp = stamps[0]
                image, info = self.images[stamp], self.infos[stamp]
                detection = self.detections.get(stamp)
            if stamp - last_stamp < 60_000_000:
                continue
            age = (self.node.get_clock().now().nanoseconds - stamp) / 1e9
            if not 0 <= age <= 0.5:
                last_error = f'Stale/future image: {age:.3f}s'
                continue
            try:
                if image.header.frame_id != CAMERA_FRAME or info.header.frame_id != CAMERA_FRAME:
                    raise ValueError('Wrong camera optical frame')
                if image.encoding not in ('rgb8', 'bgr8') or (image.width, image.height) != (info.width, info.height):
                    raise ValueError('Image encoding/resolution mismatch')
                if not np.allclose(np.asarray(info.r).reshape(3, 3), np.eye(3), atol=1e-8):
                    raise ValueError('Nonidentity rectification unsupported')
                K = np.asarray(info.p).reshape(3, 4)[:, :3]
                if not np.all(np.isfinite(K)) or K[0, 0] <= 0 or K[1, 1] <= 0:
                    raise ValueError('Invalid intrinsics')
                hand = tf_matrix(self.buffer.lookup_transform(BASE_FRAME, 'link6', Time(nanoseconds=stamp)).transform)
                tags = [] if detection is None else [d for d in detection.detections if d.id == 0 and d.hamming == 0]
                if check_only:
                    return dict(image_age_s=age, tag_detected=len(tags) == 1,
                                image_size=[image.width, image.height], robot_tf_matched=True)
                if detection.header.frame_id != CAMERA_FRAME or len(tags) != 1:
                    raise ValueError('ID 0 Tag must be fully visible and correctly decoded')
                corners = np.array([[p.x, p.y] for p in tags[0].corners])
                if corners.shape != (4, 2) or not np.all(np.isfinite(corners)):
                    raise ValueError('Invalid Tag corners')
                if np.any(corners < 8) or np.any(corners[:, 0] >= image.width - 8) or np.any(corners[:, 1] >= image.height - 8):
                    raise ValueError('Tag too close to image border')
                pixels = np.frombuffer(bytes(image.data), np.uint8).reshape(image.height, image.step)
                pixels = pixels[:, :image.width * 3].reshape(image.height, image.width, 3).copy()
                if image.encoding == 'rgb8':
                    pixels = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
                frame = dict(stamp_ns=stamp, image_age_s=age, K=K.tolist(), corners=corners.tolist(),
                             T_base_link6=hand.tolist(), decision_margin=float(tags[0].decision_margin))
                frames.append(frame)
                last_image = pixels
                last_stamp = stamp
                if len(frames) >= 8:
                    baseline = np.asarray(frames[0]['T_base_link6'])
                    translation = max(np.linalg.norm(np.asarray(f['T_base_link6'])[:3, 3] - baseline[:3, 3]) for f in frames) * 1000
                    angle = max(np.degrees(Rotation.from_matrix(baseline[:3, :3].T @ np.asarray(f['T_base_link6'])[:3, :3]).magnitude()) for f in frames)
                    jitter = max(np.max(np.linalg.norm(np.asarray(f['corners']) - frames[0]['corners'], axis=1)) for f in frames)
                    if translation > 1 or angle > 0.3 or jitter > 1:
                        last_error = f'Not stationary: {translation:.2f}mm / {angle:.2f}deg / {jitter:.2f}px'
                        frames.clear()
                        continue
                    return frames, last_image, dict(translation_range_mm=translation,
                                                    rotation_range_deg=angle, corner_range_px=jitter)
            except (TransformException, ValueError) as error:
                last_error = str(error)
                frames.clear()
        raise ValueError(f'未采集：{last_error}')


def session_data(source):
    original = source.read_bytes()
    camera = load_calibration(source)
    if source.read_bytes() != original:
        raise ValueError('Calibration changed while loading')
    return dict(format=FORMAT, calibration_file=str(source), calibration_sha256=hashlib.sha256(original).hexdigest(),
                T_base_cam=camera.tolist(), tag_size_m=SIZE, T_link6_tag=None, poses=[])


def save_previews(data, folder):
    if data['T_link6_tag'] is None:
        return
    for pose in data['poses']:
        image = cv2.imread(str(folder / pose['raw_image']))
        frame = pose['frames'][-1]
        predicted = project(np.linalg.inv(data['T_base_cam']), frame['T_base_link6'], data['T_link6_tag'], frame['K'])
        observed = np.asarray(frame['corners'])
        for number, (a, b) in enumerate(zip(observed, predicted)):
            a, b = tuple(np.rint(a).astype(int)), tuple(np.rint(b).astype(int))
            cv2.circle(image, a, 4, (0, 255, 0), 1)
            cv2.drawMarker(image, b, (255, 255, 0), cv2.MARKER_CROSS, 10, 1)
            cv2.line(image, a, b, (0, 0, 255), 1)
            cv2.putText(image, str(number), a, cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 255, 0), 1)
        cv2.putText(image, f"{pose['role']} | green measured / cyan predicted", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 255, 255), 1)
        if not cv2.imwrite(str(folder / pose['comparison_image']), image):
            raise OSError('Comparison image save failed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['init', 'check', 'capture', 'report'])
    parser.add_argument('--calibration', type=Path)
    parser.add_argument('--session', type=Path)
    parser.add_argument('--label', default='')
    parser.add_argument('--timeout', type=float, default=10)
    parser.add_argument('--mounting-from', type=Path, help='init only: freeze a solved candidate JSON mounting transform')
    parser.add_argument('--validation-count', type=int, default=5, help='Number of fresh candidate validation poses')
    args = parser.parse_args()
    reader = None
    try:
        if not np.isfinite(args.timeout) or args.timeout <= 0:
            raise ValueError('Timeout must be positive and finite')
        if args.mounting_from is not None and args.command != 'init':
            raise ValueError('--mounting-from is allowed only at init, never during capture')
        if not 1 <= args.validation_count <= 30:
            raise ValueError('--validation-count must be between 1 and 30')
        if args.command == 'check':
            reader = Reader()
            print(json.dumps(reader.collect(args.timeout, check_only=True)))
            return 0
        if args.session is None or args.session.suffix != '.json':
            raise ValueError('--session must be a separate .json file')
        session = args.session.resolve()
        session.parent.mkdir(parents=True, exist_ok=True)
        with session.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if args.command == 'report':
                data = json.loads(session.read_text())
                if data['format'] != FORMAT:
                    raise ValueError('Unsupported session')
                print(report(data))
                return 0
            if args.calibration is None:
                raise ValueError('--calibration required')
            frozen = session_data(args.calibration.resolve())
            if session in (args.calibration.resolve(), session.with_suffix('.report.md')):
                raise ValueError('Output must not overwrite calibration')
            if args.command == 'init':
                if session.exists():
                    raise ValueError('Session already exists; will not overwrite')
                data = frozen
                if args.mounting_from is not None:
                    candidate_bytes = args.mounting_from.read_bytes()
                    candidate = json.loads(candidate_bytes)
                    if candidate['format'] != 'piper_corner_candidate_v1' or candidate['tag_size_m'] != SIZE:
                        raise ValueError('Unsupported candidate or Tag size')
                    if candidate['candidate_calibration_sha256'] != data['calibration_sha256'] or not np.allclose(candidate['T_base_cam'], data['T_base_cam'], atol=1e-12):
                        raise ValueError('Candidate mounting does not belong to this calibration')
                    data.update(mode='candidate_held_out', max_poses=args.validation_count,
                                T_link6_tag=checked_matrix(candidate['T_link6_tag']).tolist(),
                                candidate_metadata_file=str(args.mounting_from.resolve()),
                                candidate_metadata_sha256=hashlib.sha256(candidate_bytes).hexdigest(),
                                candidate_training_session_sha256=candidate['source_session_sha256'])
            else:
                data = json.loads(session.read_text())
                if data['format'] != FORMAT or data['calibration_sha256'] != frozen['calibration_sha256'] or not np.allclose(data['T_base_cam'], frozen['T_base_cam'], atol=1e-12):
                    raise ValueError('Calibration/session mismatch')
                index = len(data['poses']) + 1
                maximum = data.get('max_poses', 10)
                candidate_mode = data.get('mode') == 'candidate_held_out'
                if candidate_mode:
                    metadata = Path(data['candidate_metadata_file']).read_bytes()
                    if hashlib.sha256(metadata).hexdigest() != data['candidate_metadata_sha256']:
                        raise ValueError('Candidate metadata changed since initialization')
                    if not np.allclose(json.loads(metadata)['T_link6_tag'], data['T_link6_tag'], atol=1e-12):
                        raise ValueError('Frozen mounting changed since initialization')
                if index > maximum:
                    raise ValueError('Requested validation session complete')
                reader = Reader()
                frames, pixels, stability = reader.collect(args.timeout)
                if data['poses'] and frames[0]['stamp_ns'] <= data['poses'][-1]['frames'][-1]['stamp_ns']:
                    raise ValueError('Timestamps did not advance')
                if hashlib.sha256(args.calibration.read_bytes()).hexdigest() != data['calibration_sha256']:
                    raise ValueError('Calibration changed during capture')
                pose = dict(label=args.label or f'pose_{index:02d}', role='mounting_fit' if not candidate_mode and index <= 3 else 'held_out',
                            recorded_at=datetime.now(timezone.utc).isoformat(), frames=frames, stability=stability,
                            raw_image=f'{session.stem}_{index:02d}_raw.png',
                            comparison_image=f'{session.stem}_{index:02d}_comparison.png')
                data['poses'].append(pose)
                if index == 3 and not candidate_mode:
                    mounting, diagnostics = fit_mounting(np.linalg.inv(data['T_base_cam']),
                                                        [f for p in data['poses'][:3] for f in p['frames']])
                    data['T_link6_tag'], data['mounting_fit'] = mounting.tolist(), diagnostics
                if not cv2.imwrite(str(session.parent / pose['raw_image']), pixels):
                    raise OSError('Raw image save failed')
                print(f"已保存 {index}/{maximum}，{pose['role']}，8 帧；停稳检查：{stability}")
                if data['T_link6_tag'] is not None:
                    errors = error_values(data, pose)
                    print(f'四角 RMS {np.sqrt(np.mean(errors**2)):.3f} px；最大 {np.max(errors):.3f} px')
            atomic_write(session, json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
            atomic_write(session.with_suffix('.report.md'), report(data))
            save_previews(data, session.parent)
            print(f'数据：{session}')
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    finally:
        if reader is not None:
            reader.close()


if __name__ == '__main__':
    raise SystemExit(main())
