#!/usr/bin/env python3
"""Offline, explicitly requested joint camera/mounting fit from raw corners.

Fits T_cam_base and T_link6_tag with fixed intrinsics and 42 mm Tag geometry.
Writes an exclusive new candidate; never updates the source calibration/session.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import yaml

from print_tag_position import load_calibration
from validate_consistency import checked_matrix
from validate_reprojection import FORMAT, SIZE, POINTS, matrix, parameters


def objective(frames):
    hands = np.array([checked_matrix(f['T_base_link6']) for f in frames])
    intrinsics = np.array([f['K'] for f in frames], float)
    measured = np.array([f['corners'] for f in frames], float)
    if intrinsics.shape != (len(frames), 3, 3) or measured.shape != (len(frames), 4, 2):
        raise ValueError('Invalid intrinsics/corner dimensions')
    if not np.all(np.isfinite(intrinsics)) or not np.all(np.isfinite(measured)):
        raise ValueError('Nonfinite observations')
    def residual(value):
        transforms = matrix(value[:6]) @ hands @ matrix(value[6:])
        camera = np.einsum('nij,kj->nki', transforms[:, :3, :3], POINTS) + transforms[:, :3, 3][:, None, :]
        if np.any(camera[:, :, 2] <= .01):
            return np.full(measured.size, 1e6)
        projected = np.einsum('nij,nkj->nki', intrinsics, camera)
        return (projected[:, :, :2] / projected[:, :, 2:] - measured).ravel()
    return residual


def optimize(frames, initial):
    fit = least_squares(objective(frames), initial, x_scale='jac', max_nfev=2000,
                        ftol=1e-12, xtol=1e-12, gtol=1e-12)
    if not fit.success or not np.all(np.isfinite(fit.x)) or np.max(np.abs(fit.fun)) >= 1e6:
        raise ValueError('Joint optimization failed')
    singular = np.linalg.svd(fit.jac, compute_uv=False)
    if singular[-1] <= singular[0] * 1e-8:
        raise ValueError('Joint fit is degenerate; candidate not saved')
    return fit


def stats(errors):
    return dict(rms_px=float(np.sqrt(np.mean(errors ** 2))), median_px=float(np.median(errors)),
                max_px=float(np.max(errors)))


def fit_candidate(data, starts=7, leave_one_out=True):
    if data['format'] != FORMAT or data['tag_size_m'] != SIZE or len(data['poses']) < 6:
        raise ValueError('Expected at least six poses with the configured 42 mm Tag')
    frames = [frame for pose in data['poses'] for frame in pose['frames']]
    initial = np.r_[parameters(np.linalg.inv(checked_matrix(data['T_base_cam']))),
                     parameters(checked_matrix(data['T_link6_tag']))]
    rng = np.random.default_rng(20261013)
    fits = []
    for index in range(starts):
        seed = initial if index == 0 else initial + rng.normal(size=12) * np.tile([.08, .08, .08, .03, .03, .03], 2)
        try:
            fits.append(optimize(frames, seed))
        except ValueError:
            pass
    if not fits:
        raise ValueError('No valid candidate found')
    best = min(fits, key=lambda fit: np.sum(fit.fun ** 2))
    camera = np.linalg.inv(matrix(best.x[:6]))
    mounting = matrix(best.x[6:])
    counts = [len(pose['frames']) for pose in data['poses']]
    errors = np.linalg.norm(best.fun.reshape(-1, 4, 2), axis=2)
    pose_errors = np.split(errors, np.cumsum(counts)[:-1])
    diagnostics = dict(training=stats(errors), per_pose=[stats(e) for e in pose_errors],
                       successful_starts=len(fits), requested_starts=starts,
                       start_rms_px=[stats(np.linalg.norm(f.fun.reshape(-1, 2), axis=1))['rms_px'] for f in fits],
                       jacobian_singular_values=np.linalg.svd(best.jac, compute_uv=False).tolist(),
                       solver='scipy.optimize.least_squares; linear loss; all recorded observations retained',
                       num_parameters=12, pose_count=len(data['poses']), frame_count=len(frames))
    old = np.asarray(data['T_base_cam'])
    diagnostics['delta_from_original'] = dict(translation_mm=float(np.linalg.norm(camera[:3, 3] - old[:3, 3]) * 1000),
        rotation_deg=float(np.degrees(Rotation.from_matrix(old[:3, :3].T @ camera[:3, :3]).magnitude())))
    if leave_one_out:
        results = []
        for index, pose in enumerate(data['poses']):
            subset = [f for number, other in enumerate(data['poses']) if number != index for f in other['frames']]
            fit = optimize(subset, best.x)
            reduced_camera = np.linalg.inv(matrix(fit.x[:6]))
            excluded_error = np.linalg.norm(objective(pose['frames'])(fit.x).reshape(-1, 2), axis=1)
            results.append(dict(excluded_pose=index + 1, excluded_label=pose['label'], **stats(excluded_error),
                camera_shift_mm=float(np.linalg.norm(reduced_camera[:3, 3] - camera[:3, 3]) * 1000),
                camera_shift_deg=float(np.degrees(Rotation.from_matrix(camera[:3, :3].T @ reduced_camera[:3, :3]).magnitude()))))
        diagnostics['leave_one_pose_out'] = results
    return camera, mounting, diagnostics


def render_report(result):
    fit = result['diagnostics']
    lines = ['# 四角联合求解候选外参', '',
             '- 状态：候选，尚未完成全新姿态验证；不得据训练误差认定绝对精度。',
             f"- 数据：{fit['pose_count']} 个姿态、{fit['frame_count']} 帧。全部用于本次求解，原留出标签不再表示对新候选的留出验证。",
             '- 求解：固定内参、42 mm Tag；联合优化 T_cam_base 与 T_link6_tag 共 12 个参数；输出 T_base_cam。',
             '- 原始角点直接进入像素残差；未使用每帧 Tag 三维姿态作拟合观测，未删除任何姿态。',
             f"- 训练 RMS：{fit['training']['rms_px']:.3f} px；最大：{fit['training']['max_px']:.3f} px。",
             f"- 多初值：{fit['successful_starts']}/{fit['requested_starts']} 个成功。",
             f"- 与原外参差异：相机位置 {fit['delta_from_original']['translation_mm']:.3f} mm；旋转 {fit['delta_from_original']['rotation_deg']:.3f}°。", '',
             '## T_base_cam（平移单位 m）', '', '```text',
             *[' '.join(f'{v: .10f}' for v in row) for row in result['T_base_cam']], '```', '',
             '| 训练姿态 | 四角 RMS px | 最大 px |', '|---|---:|---:|',
             *[f"| {i + 1} | {s['rms_px']:.3f} | {s['max_px']:.3f} |" for i, s in enumerate(fit['per_pose'])], '']
    if 'leave_one_pose_out' in fit:
        lines += ['## 留一姿态敏感性（诊断，不替代新数据验证）', '',
                  '| 排除姿态 | 被排除姿态 RMS px | 相机位置变化 mm | 旋转变化 ° |', '|---|---:|---:|---:|',
                  *[f"| {v['excluded_pose']} | {v['rms_px']:.3f} | {v['camera_shift_mm']:.3f} | {v['camera_shift_deg']:.3f} |"
                    for v in fit['leave_one_pose_out']], '']
    lines += [f"- 输入会话 SHA256：`{result['source_session_sha256']}`",
              f"- 候选 .calib SHA256：`{result['candidate_calibration_sha256']}`",
              '- 原外参/原会话保持不变；新验证须冻结本候选的外参和安装关系。', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path, help='New .calib path; existing files are refused')
    args = parser.parse_args()
    try:
        session, target = args.session.resolve(), args.output.absolute()
        if target.suffix != '.calib':
            raise ValueError('Candidate must have .calib suffix')
        targets = [target, target.with_suffix('.json'), target.with_suffix('.report.md')]
        if any(path.exists() or path.is_symlink() for path in targets):
            raise ValueError('Output already exists; will not overwrite')
        original_session = session.read_bytes()
        data = json.loads(original_session)
        source = Path(data['calibration_file'])
        original_calibration = source.read_bytes()
        if hashlib.sha256(original_calibration).hexdigest() != data['calibration_sha256'] or not np.allclose(load_calibration(source), data['T_base_cam'], atol=1e-12):
            raise ValueError('Source calibration no longer matches captured data')
        camera, mounting, diagnostics = fit_candidate(data)
        native = copy.deepcopy(yaml.safe_load(original_calibration))
        native['parameters']['name'] = target.stem
        native['transform'] = dict(translation=dict(zip('xyz', map(float, camera[:3, 3]))),
                                   rotation=dict(zip('xyzw', map(float, Rotation.from_matrix(camera[:3, :3]).as_quat()))))
        native_text = yaml.safe_dump(native, sort_keys=False)
        result = dict(format='piper_corner_candidate_v1', status='candidate_pending_fresh_validation',
                      created_at=datetime.now(timezone.utc).isoformat(), candidate_calibration_file=str(target),
                      candidate_calibration_sha256=hashlib.sha256(native_text.encode()).hexdigest(),
                      source_session=str(session), source_session_sha256=hashlib.sha256(original_session).hexdigest(),
                      source_calibration_file=str(source), source_calibration_sha256=data['calibration_sha256'],
                      tag_size_m=SIZE, T_base_cam=camera.tolist(), T_link6_tag=mounting.tolist(), diagnostics=diagnostics)
        if source.read_bytes() != original_calibration or session.read_bytes() != original_session:
            raise ValueError('Source changed during solving; no candidate saved')
        target.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation ensures a rerun cannot overwrite any existing result.
        contents = [native_text, json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', render_report(result)]
        for path, content in zip(targets, contents):
            with path.open('x') as stream:
                stream.write(content)
        print(render_report(result))
        print('Candidate:', target)
        return 0
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    raise SystemExit(main())
