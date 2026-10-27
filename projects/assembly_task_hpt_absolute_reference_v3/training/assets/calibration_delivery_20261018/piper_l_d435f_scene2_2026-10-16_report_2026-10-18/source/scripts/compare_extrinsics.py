#!/usr/bin/env python3
"""Record one shared observation and compare two frozen camera/mounting pairs.

No calibration fitting, Tag pose TF, CAN connection or robot control.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from print_tag_position import load_calibration
from validate_consistency import atomic_write, checked_matrix
from validate_reprojection import Reader, project

FORMAT = 'piper_frozen_extrinsic_comparison_v1'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_sources(data):
    for source in data['sources']:
        if digest(source['path']) != source['sha256']:
            raise ValueError(f"Frozen input changed: {source['path']}")
    for model in data['models'].values():
        checked_matrix(model['T_base_cam'])
        checked_matrix(model['T_link6_tag'])


def errors(model, pose):
    return np.concatenate([np.linalg.norm(project(np.linalg.inv(model['T_base_cam']), frame['T_base_link6'],
                           model['T_link6_tag'], frame['K']) - frame['corners'], axis=1)
                           for frame in pose['frames']])


def statistics(values):
    return dict(rms_px=float(np.sqrt(np.mean(values ** 2))), median_px=float(np.median(values)),
                max_px=float(np.max(values)))


def report(data):
    lines = ['# 随机新姿态：原外参与候选外参比较', '',
             '- 两套相机外参和各自安装关系均在新姿态采集前固定；本轮不进行任何拟合。',
             '- 两套误差来自同一组图像角点、内参和同时间戳机器人 TF；单位 px。',
             '- 原外参的安装关系沿用此前仅由 10 个训练姿态确定的离线比较结果。',
             f"- 已采集 {len(data['poses'])}/{data['max_poses']} 个随机新姿态，每点 8 帧。", '',
             '| 点 | 原外参 RMS px | 候选 RMS px | 原最大 px | 候选最大 px |', '|---|---:|---:|---:|---:|']
    collected = {'original': [], 'candidate': []}
    improved = 0
    for index, pose in enumerate(data['poses'], 1):
        values = {name: errors(model, pose) for name, model in data['models'].items()}
        old, new = [statistics(values[name]) for name in ('original', 'candidate')]
        for name in collected:
            collected[name].extend(values[name])
        improved += new['rms_px'] < old['rms_px']
        lines.append(f"| {index} | {old['rms_px']:.3f} | {new['rms_px']:.3f} | {old['max_px']:.3f} | {new['max_px']:.3f} |")
    if data['poses']:
        lines += ['', '| 外参 | RMS px | 中位数 px | 最大 px |', '|---|---:|---:|---:|']
        for name, label in [('original', '原外参'), ('candidate', '候选外参')]:
            value = statistics(np.array(collected[name]))
            lines.append(f"| {label} | {value['rms_px']:.3f} | {value['median_px']:.3f} | {value['max_px']:.3f} |")
        lines += ['', f"候选在 {improved}/{len(data['poses'])} 个姿态的 RMS 较小。"]
    lines += ['', '- 本报告是像素预测误差比较，不是毫米级绝对精度或自动达标结论。', '']
    return '\n'.join(lines)


def initialize(baseline_path, candidate_path, count):
    comparison = json.loads(baseline_path.read_text())
    candidate = json.loads(candidate_path.read_text())
    if candidate['format'] != 'piper_corner_candidate_v1' or candidate['tag_size_m'] != .042:
        raise ValueError('Unsupported candidate')
    if comparison['training_sha256'] != candidate['source_session_sha256']:
        raise ValueError('Comparison and candidate use different training data')
    if not np.allclose(comparison['results']['candidate']['T_link6_tag'], candidate['T_link6_tag'], atol=1e-12):
        raise ValueError('Candidate mounting mismatch')
    sources = [dict(path=str(baseline_path), sha256=digest(baseline_path)),
               dict(path=str(candidate_path), sha256=digest(candidate_path))]
    models = {}
    for name, path_key, hash_key, mounting in [
        ('original', 'source_calibration_file', 'source_calibration_sha256', comparison['results']['original']['T_link6_tag']),
        ('candidate', 'candidate_calibration_file', 'candidate_calibration_sha256', candidate['T_link6_tag'])]:
        path = Path(candidate[path_key]).resolve()
        sources.append(dict(path=str(path), sha256=candidate[hash_key]))
        models[name] = dict(T_base_cam=load_calibration(path).tolist(),
                            T_link6_tag=checked_matrix(mounting).tolist(), calibration_file=str(path))
    data = dict(format=FORMAT, created_at=datetime.now(timezone.utc).isoformat(), models=models,
                sources=sources, max_poses=count, poses=[])
    verify_sources(data)
    return data


def save_comparison(pixels, pose, model, name, path):
    image = pixels.copy()
    frame = pose['frames'][-1]
    predicted = project(np.linalg.inv(model['T_base_cam']), frame['T_base_link6'], model['T_link6_tag'], frame['K'])
    for measured, expected in zip(frame['corners'], predicted):
        a, b = tuple(np.rint(measured).astype(int)), tuple(np.rint(expected).astype(int))
        cv2.circle(image, a, 4, (0, 255, 0), 1)
        cv2.drawMarker(image, b, (255, 255, 0), cv2.MARKER_CROSS, 10, 1)
        cv2.line(image, a, b, (0, 0, 255), 1)
    cv2.putText(image, f'{name}: green observed / cyan predicted', (8, 20), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 255, 255), 1)
    if not cv2.imwrite(str(path), image):
        raise OSError('Comparison image save failed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['init', 'capture', 'report'])
    parser.add_argument('--session', required=True, type=Path)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--candidate', type=Path)
    parser.add_argument('--count', type=int, default=5)
    parser.add_argument('--timeout', type=float, default=10)
    args = parser.parse_args()
    reader = None
    try:
        if args.session.suffix != '.json' or not 1 <= args.count <= 30 or not np.isfinite(args.timeout) or args.timeout <= 0:
            raise ValueError('Invalid session suffix, pose count or timeout')
        session = args.session.resolve()
        session.parent.mkdir(parents=True, exist_ok=True)
        with session.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if args.command == 'init':
                if session.exists() or session.with_suffix('.report.md').exists():
                    raise ValueError('Existing session/output will not be overwritten')
                if args.baseline is None or args.candidate is None:
                    raise ValueError('--baseline and --candidate required at init')
                data = initialize(args.baseline.resolve(), args.candidate.resolve(), args.count)
            else:
                data = json.loads(session.read_text())
                if data['format'] != FORMAT:
                    raise ValueError('Unsupported comparison session')
                verify_sources(data)
                if args.command == 'report':
                    print(report(data))
                    return 0
                index = len(data['poses']) + 1
                if index > data['max_poses']:
                    raise ValueError('Requested comparison session complete')
                reader = Reader()
                frames, pixels, stability = reader.collect(args.timeout)
                if data['poses'] and frames[0]['stamp_ns'] <= data['poses'][-1]['frames'][-1]['stamp_ns']:
                    raise ValueError('Timestamp did not advance')
                verify_sources(data)
                prefix = f'{session.stem}_{index:02d}'
                pose = dict(label=f'random_{index:02d}', role='held_out', recorded_at=datetime.now(timezone.utc).isoformat(),
                            frames=frames, stability=stability, raw_image=prefix + '_raw.png',
                            comparisons={name: prefix + '_' + name + '.png' for name in data['models']})
                # Evaluate both models on exactly the same frames before any result is saved.
                values = {name: statistics(errors(model, pose)) for name, model in data['models'].items()}
                if not cv2.imwrite(str(session.parent / pose['raw_image']), pixels):
                    raise OSError('Raw image save failed')
                for name, model in data['models'].items():
                    save_comparison(pixels, pose, model, name, session.parent / pose['comparisons'][name])
                pose['errors'] = values
                data['poses'].append(pose)
                print(f"已保存 {index}/{data['max_poses']}；同一组 8 帧；停稳检查：{stability}")
                print(f"原外参 RMS {values['original']['rms_px']:.3f} px，最大 {values['original']['max_px']:.3f} px")
                print(f"候选 RMS {values['candidate']['rms_px']:.3f} px，最大 {values['candidate']['max_px']:.3f} px")
            atomic_write(session, json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
            atomic_write(session.with_suffix('.report.md'), report(data))
            print('数据：', session)
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    finally:
        if reader is not None:
            reader.close()


if __name__ == '__main__':
    raise SystemExit(main())
