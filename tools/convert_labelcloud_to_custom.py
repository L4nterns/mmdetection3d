#!/usr/bin/env python3
import argparse
from collections import Counter
import hashlib
import json
import math
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.labelcloud.utils import (
    aligned_point_cloud_range, class_stats, load_classes, output_is_stale,
    decode_label_name, points_in_rotated_box, read_label_file,
    read_pcd_as_xyzrgb, safe_name, split_scenes, write_bin_atomic,
    write_json_atomic, write_text_atomic)


SPARSE_RGB_MODELS = ('tr3d', 'fcaf3d')
MODEL_CHOICES = ('pv_rcnn', 'pointpillars') + SPARSE_RGB_MODELS
AUG_MODE_CHOICES = ('full', 'safe', 'none')
GT_DATABASE_CHOICES = ('auto', 'on', 'off')
POINT_FEATURE_CHOICES = ('xyzrgb', 'xyzi')
DEFAULT_SPARSE_REPEAT_TIMES = 5
DEFAULT_LIDAR_REPEAT_TIMES = 2
DEFAULT_VAL_INTERVAL = 5


def load_env_file(path=None):
    """加载 .env 到环境变量；不覆盖已存在的系统环境变量。"""
    candidates = [Path.cwd() / '.env', ROOT / '.env'] if path is None else [Path(path)]
    seen = set()
    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if candidate in seen or not candidate.exists():
            continue
        seen.add(candidate)
        for raw_line in candidate.read_text(encoding='utf-8').splitlines():
            line = raw_line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('export '):
                line = line[7:].strip()
            if '=' not in line:
                continue
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip()
            if not key or key in os.environ:
                continue
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
                value = value[1:-1]
            os.environ[key] = value


load_env_file()


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert a labelCloud project to MMDetection3D format.')
    parser.add_argument(
        '--labelcloud-root',
        '--labelcloud_root',
        default=os.environ.get('LABELCLOUD_DIR'))
    parser.add_argument('--class-file', default=None)
    parser.add_argument('--pointcloud-dir', default=None)
    parser.add_argument('--label-dir', default=None)
    parser.add_argument(
        '--out-dir',
        '--out_dir',
        default=os.environ.get('OUT_DIR', 'data/labelcloud'))
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=float(os.environ.get('TRAIN_RATIO', '0.8')))
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=float(os.environ.get('VAL_RATIO', '0.2')))
    parser.add_argument(
        '--seed',
        type=int,
        default=int(os.environ.get('SPLIT_SEED', '42')))
    parser.add_argument(
        '--model',
        choices=MODEL_CHOICES,
        default=os.environ.get('MODEL', 'tr3d'))
    parser.add_argument('--model-cfg', default=os.environ.get('CFG_FILE'))
    parser.add_argument('--extra-tag', default=os.environ.get('EXTRA_TAG'))
    parser.add_argument(
        '--epochs', type=int, default=int(os.environ.get('EPOCHS', '80')))
    parser.add_argument(
        '--train-repeat-times',
        default=os.environ.get('TRAIN_REPEAT_TIMES', 'auto'),
        help='RepeatDataset times. "auto" uses model-specific defaults.')
    parser.add_argument(
        '--val-interval',
        default=os.environ.get('VAL_INTERVAL', str(DEFAULT_VAL_INTERVAL)),
        help='Validation interval in epochs; clamped to max_epochs.')
    parser.add_argument('--voxel-xy', type=float, default=0.1)
    parser.add_argument('--range-margin', type=float, default=5.0)
    parser.add_argument('--min-points-filter', type=int, default=5)
    parser.add_argument('--sample-group-size', type=int, default=15)
    parser.add_argument(
        '--sample-points',
        type=int,
        default=int(os.environ.get('SAMPLE_POINTS', '500000')))
    parser.add_argument(
        '--sparse-voxel-size',
        type=float,
        default=float(os.environ.get('SPARSE_VOXEL_SIZE', '0.015')),
        help='Cubic voxel size for TR3D/FCAF3D sparse RGB backbones.')
    parser.add_argument(
        '--point-features',
        choices=POINT_FEATURE_CHOICES,
        default=os.environ.get('POINT_FEATURES', 'xyzrgb'),
        help='xyzrgb uses all 6 saved dimensions; xyzi uses x/y/z/red only.')
    parser.add_argument(
        '--aug-mode',
        choices=AUG_MODE_CHOICES,
        default=os.environ.get('AUG_MODE', 'full'),
        help='full: ObjectSample+flip+rotation+scale; safe: scale only; '
        'none: no geometric augmentation.')
    parser.add_argument(
        '--gt-database',
        choices=GT_DATABASE_CHOICES,
        default=os.environ.get('GT_DATABASE', 'auto'),
        help='auto creates GT database only when aug-mode=full; on/off force it.')
    parser.add_argument('--rebuild-gt-database', action='store_true')
    parser.add_argument(
        '--gt-database-target-elements',
        type=int,
        default=int(os.environ.get('GT_DATABASE_TARGET_ELEMENTS', '4000000')))
    parser.add_argument(
        '--gt-database-min-chunk-size',
        type=int,
        default=int(os.environ.get('GT_DATABASE_MIN_CHUNK_SIZE', '50000')))
    parser.add_argument(
        '--gt-database-max-chunk-size',
        type=int,
        default=int(os.environ.get('GT_DATABASE_MAX_CHUNK_SIZE', '500000')))
    parser.add_argument(
        '--gt-database-max-points',
        type=int,
        default=int(os.environ.get('GT_DATABASE_MAX_POINTS', '0')))
    parser.add_argument(
        '--min-box-dim',
        type=float,
        default=float(os.environ.get('MIN_BOX_DIM', '0.0001')),
        help='过滤任意边长小于该阈值的 3D 框；<=0 表示不按边长过滤。')
    parser.add_argument(
        '--min-points-in-gt',
        type=int,
        default=int(os.environ.get('MIN_POINTS_IN_GT', '1')),
        help='过滤内部点数小于该阈值的 3D 框；<=0 表示不按点数过滤。')
    parser.add_argument(
        '--train-lr',
        type=float,
        default=float(os.environ.get('TRAIN_LR', '0.0003')),
        help='生成配置中的 AdamW 基础学习率。')
    parser.add_argument(
        '--train-lr-peak-ratio',
        type=float,
        default=float(os.environ.get('TRAIN_LR_PEAK_RATIO', '1.0')),
        help='第一段学习率调度的峰值倍率；1.0 表示不升高学习率。')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument(
        '--skip-gt-database',
        action='store_true',
        help='Do not create labelcloud_dbinfos_train.pkl and gt database.')
    return parser.parse_args()


def resolve_inputs(args):
    if args.labelcloud_root:
        root = Path(args.labelcloud_root)
        class_file = Path(args.class_file) if args.class_file else root / '_classes.json'
        pointcloud_dir = Path(args.pointcloud_dir) if args.pointcloud_dir else root / 'pointClouds'
        label_dir = Path(args.label_dir) if args.label_dir else root / 'labels'
    else:
        if not (args.class_file and args.pointcloud_dir and args.label_dir):
            raise ValueError(
                'Provide either --labelcloud-root or all of --class-file, '
                '--pointcloud-dir and --label-dir.')
        class_file = Path(args.class_file)
        pointcloud_dir = Path(args.pointcloud_dir)
        label_dir = Path(args.label_dir)

    for path, desc in [(class_file, 'class file'), (pointcloud_dir, 'pointcloud dir'),
                       (label_dir, 'label dir')]:
        if not path.exists():
            raise FileNotFoundError(f'Missing {desc}: {path}')
    if not class_file.is_file():
        raise FileNotFoundError(f'Class file is not a file: {class_file}')
    if not pointcloud_dir.is_dir():
        raise NotADirectoryError(f'Pointcloud dir is not a directory: {pointcloud_dir}')
    if not label_dir.is_dir():
        raise NotADirectoryError(f'Label dir is not a directory: {label_dir}')
    return class_file, pointcloud_dir, label_dir


def tagged_name(model, extra_tag=None):
    name = f'labelcloud_{model}'
    if extra_tag:
        tag = re.sub(r'[^A-Za-z0-9_.-]+', '_', extra_tag).strip('._-')
        if tag:
            name = f'{name}_{tag}'
    return name


def scene_id_from_pcd_name(name):
    return Path(name).stem


def scheduler_epochs(max_epochs):
    if max_epochs < 2:
        raise ValueError('--epochs must be at least 2 for two-phase scheduling.')
    first = max(1, int(round(max_epochs * 15 / 40)))
    first = min(first, max_epochs - 1)
    second = max_epochs - first
    return first, second


def default_train_repeat_times(model):
    return (DEFAULT_SPARSE_REPEAT_TIMES
            if model in SPARSE_RGB_MODELS else DEFAULT_LIDAR_REPEAT_TIMES)


def resolve_train_repeat_times(value, model):
    if value is None or str(value).strip().lower() == 'auto':
        return default_train_repeat_times(model)
    repeat_times = int(value)
    if repeat_times <= 0:
        raise ValueError('--train-repeat-times must be "auto" or a positive integer.')
    return repeat_times


def resolve_val_interval(value, max_epochs):
    val_interval = int(value)
    if val_interval <= 0:
        raise ValueError('--val-interval must be a positive integer.')
    return min(val_interval, max_epochs)


def epoch_cfg_options(max_epochs):
    first, second = scheduler_epochs(max_epochs)
    return [
        f'train_cfg.max_epochs={max_epochs}',
        f'param_scheduler.0.T_max={first}',
        f'param_scheduler.0.end={first}',
        f'param_scheduler.1.T_max={second}',
        f'param_scheduler.1.begin={first}',
        f'param_scheduler.1.end={max_epochs}',
        f'param_scheduler.2.T_max={first}',
        f'param_scheduler.2.end={first}',
        f'param_scheduler.3.T_max={second}',
        f'param_scheduler.3.begin={first}',
        f'param_scheduler.3.end={max_epochs}',
    ]


def fmt(value):
    return f'{float(value):.8f}'


def py_list(values):
    return '[' + ', '.join(repr(v) for v in values) + ']'


def num_points_in_bin(path, load_dim=6):
    path = Path(path)
    if not path.exists():
        return None
    try:
        points = np.fromfile(path, dtype=np.float32)
    except OSError:
        return None
    if points.size % load_dim != 0:
        return None
    return points.size // load_dim


def read_bin_points(path, load_dim=6):
    points = np.fromfile(path, dtype=np.float32)
    if points.size % load_dim != 0:
        raise ValueError(
            f'Point bin size is not divisible by load_dim={load_dim}: {path}')
    return points.reshape(-1, load_dim)


def stats_from_rows(rows, class_names):
    extents = []
    dims_by_class = {name: [] for name in class_names}
    bottoms_by_class = {name: [] for name in class_names}
    for row in rows:
        x, y, z, dx, dy, dz, heading, name = row
        dims_by_class[name].append((dx, dy, dz))
        bottoms_by_class[name].append(z - dz / 2.0)

        radius = math.sqrt(dx * dx + dy * dy) / 2.0
        extents.append((x - radius, y - radius, z - dz / 2.0))
        extents.append((x + radius, y + radius, z + dz / 2.0))
    return extents, dims_by_class, bottoms_by_class


def filter_label_rows(rows, class_names, points_xyz, min_box_dim,
                      min_points_in_gt):
    if min_box_dim <= 0 and min_points_in_gt <= 0:
        extents, dims_by_class, bottoms_by_class = stats_from_rows(
            rows, class_names)
        return rows, extents, dims_by_class, bottoms_by_class, {}, {}

    kept = []
    dropped_by_reason = {}
    dropped_by_class = {name: 0 for name in class_names}
    for row in rows:
        dims = np.asarray(row[3:6], dtype=np.float64)
        reasons = []
        if min_box_dim > 0 and np.any(dims < min_box_dim):
            reasons.append('small_box_dim')
        if min_points_in_gt > 0:
            if points_xyz is None:
                raise ValueError(
                    '--min-points-in-gt requires point data to be loaded.')
            num_points = int(points_in_rotated_box(points_xyz, row[:7]).sum())
            if num_points < min_points_in_gt:
                reasons.append('few_points_in_gt')

        if reasons:
            for reason in reasons:
                dropped_by_reason[reason] = dropped_by_reason.get(reason, 0) + 1
            dropped_by_class[row[7]] += 1
            continue
        kept.append(row)

    extents, dims_by_class, bottoms_by_class = stats_from_rows(
        kept, class_names)
    dropped_by_class = {name: count for name, count in dropped_by_class.items()
                        if count}
    return kept, extents, dims_by_class, bottoms_by_class, dropped_by_reason, dropped_by_class


def total_boxes_by_class(dims_by_class):
    return sum(len(values) for values in dims_by_class.values())


def label_class_counts(label_files):
    counts = Counter()
    for label_path in label_files:
        data = json.loads(label_path.read_text(encoding='utf-8'))
        for obj in data.get('objects', []):
            counts[decode_label_name(obj.get('name'))] += 1
    return counts


def validate_label_classes(label_files, class_names):
    class_set = set(class_names)
    counts = label_class_counts(label_files)
    unknown = Counter({
        name: count
        for name, count in counts.items()
        if name not in class_set
    })
    if not unknown:
        return

    top_unknown = ', '.join(
        f'{name}={count}' for name, count in unknown.most_common(20))
    raise ValueError(
        'Found labelCloud classes that are not declared in _classes.json: '
        f'{sum(unknown.values())} boxes across {len(unknown)} classes. '
        f'Top unknown classes: {top_unknown}. '
        'Update _classes.json or normalize label names before conversion.')


def point_feature_dims(point_features):
    if point_features == 'xyzrgb':
        return 6, [0, 1, 2, 3, 4, 5], 3
    if point_features == 'xyzi':
        return 6, [0, 1, 2, 3], 1
    raise ValueError(f'Unsupported point feature mode: {point_features}')


def sparse_rgb_voxel_size(args):
    if args.sparse_voxel_size <= 0:
        raise ValueError('--sparse-voxel-size must be positive.')
    return [
        args.sparse_voxel_size,
        args.sparse_voxel_size,
        args.sparse_voxel_size,
    ]


def should_create_gt_database(args):
    if getattr(args, 'model', None) in SPARSE_RGB_MODELS:
        if args.gt_database == 'on':
            raise ValueError(
                '--gt-database on is not supported for TR3D/FCAF3D; these '
                'models use sparse scene sampling instead of object database '
                'sampling.')
        return False
    if args.skip_gt_database:
        if args.gt_database == 'on':
            raise ValueError('--skip-gt-database conflicts with --gt-database on.')
        return False
    if args.gt_database == 'on':
        return True
    if args.gt_database == 'off':
        return False
    return args.aug_mode == 'full'


def build_data_info(scene_id, num_features, rows, class_to_idx, source_mtime_ns):
    instances = []
    for row in rows:
        x, y, z, dx, dy, dz, heading, name = row
        instances.append({
            'bbox_3d': [
                float(x), float(y), float(z), float(dx), float(dy),
                float(dz), float(heading)
            ],
            'bbox_label_3d': class_to_idx[name],
        })
    return {
        'sample_idx': scene_id,
        'lidar_points': {
            'num_pts_feats': num_features,
            'lidar_path': f'points/{scene_id}.bin',
        },
        'instances': instances,
        'source_mtime_ns': int(source_mtime_ns),
    }


def save_info(path, class_names, data_list):
    import mmengine

    path.parent.mkdir(parents=True, exist_ok=True)
    mmengine.dump(
        {
            'metainfo': {
                'classes': tuple(class_names),
                'dataset': 'labelcloud'
            },
            'data_list': data_list,
        },
        path)


def db_file_num_points(path, load_dim=6):
    try:
        return num_points_in_bin(path, load_dim=load_dim)
    except OSError:
        return None


def resolve_chunk_size(num_boxes, target_elements, min_chunk_size,
                       max_chunk_size):
    if min_chunk_size <= 0 or max_chunk_size <= 0:
        raise ValueError('GT database chunk sizes must be positive.')
    if min_chunk_size > max_chunk_size:
        raise ValueError('GT database min chunk size cannot exceed max chunk size.')
    if target_elements <= 0 or num_boxes <= 0:
        return int(max_chunk_size)
    chunk_size = int(target_elements) // max(1, int(num_boxes))
    return max(int(min_chunk_size), min(int(max_chunk_size), chunk_size))


def subsample_points(points, max_points, scene_id, gt_idx):
    if max_points <= 0 or points.shape[0] <= max_points:
        return points
    seed_bytes = hashlib.blake2b(
        f'{scene_id}:{gt_idx}'.encode('utf-8'), digest_size=8).digest()
    seed = int.from_bytes(seed_bytes, byteorder='little') % (2**32)
    rng = np.random.default_rng(seed)
    choice = rng.choice(points.shape[0], max_points, replace=False)
    return points[choice]


def gt_database_manifest(class_names,
                         target_elements=4000000,
                         min_chunk_size=50000,
                         max_chunk_size=500000,
                         max_points=0,
                         load_dim=6):
    return {
        'version': 1,
        'class_names': list(class_names),
        'target_elements': int(target_elements),
        'min_chunk_size': int(min_chunk_size),
        'max_chunk_size': int(max_chunk_size),
        'max_points': int(max_points),
        'load_dim': int(load_dim),
    }


def read_gt_database_manifest(path):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return None


def create_gt_database(out_dir,
                       train_infos,
                       class_names,
                       rebuild=False,
                       target_elements=4000000,
                       min_chunk_size=50000,
                       max_chunk_size=500000,
                       max_points=0,
                       load_dim=6):
    import mmengine

    out_dir = Path(out_dir)
    db_dir = out_dir / 'labelcloud_gt_database'
    db_info_path = out_dir / 'labelcloud_dbinfos_train.pkl'
    manifest_path = out_dir / 'labelcloud_gt_database_manifest.json'
    manifest = gt_database_manifest(
        class_names,
        target_elements=target_elements,
        min_chunk_size=min_chunk_size,
        max_chunk_size=max_chunk_size,
        max_points=max_points,
        load_dim=load_dim)
    existing_manifest = read_gt_database_manifest(manifest_path)
    manifest_mismatch = existing_manifest != manifest
    if manifest_mismatch and (db_dir.exists() or db_info_path.exists()):
        rebuild = True
    if rebuild and db_dir.exists():
        shutil.rmtree(db_dir)
    if rebuild and db_info_path.exists():
        db_info_path.unlink()
    if rebuild and manifest_path.exists():
        manifest_path.unlink()
    db_dir.mkdir(parents=True, exist_ok=True)

    all_db_infos = {name: [] for name in class_names}
    group_id = 0
    for info in train_infos:
        scene_id = info['sample_idx']
        points = np.fromfile(out_dir / info['lidar_points']['lidar_path'],
                             dtype=np.float32).reshape(-1, load_dim)
        source_mtime_ns = int(info.get('source_mtime_ns', 0))
        missing = []
        boxes = []
        for gt_idx, instance in enumerate(info['instances']):
            label = int(instance['bbox_label_3d'])
            name = class_names[label]
            box = np.asarray(instance['bbox_3d'], dtype=np.float64)
            db_box = box.copy()
            db_box[2] -= db_box[5] / 2.0
            filename = f'{scene_id}_{safe_name(name)}_{gt_idx}.bin'
            rel_path = Path('labelcloud_gt_database') / filename
            abs_path = db_dir / filename

            existing_num_points = db_file_num_points(abs_path, load_dim=load_dim)
            if (existing_num_points is not None and
                    abs_path.stat().st_mtime_ns >= source_mtime_ns):
                num_points = int(existing_num_points)
            else:
                missing.append((gt_idx, label, name, db_box, rel_path, abs_path))
                boxes.append(box)
                continue

            all_db_infos[name].append({
                'name': name,
                'path': rel_path.as_posix(),
                'image_idx': scene_id,
                'gt_idx': gt_idx,
                'box3d_lidar': db_box.astype(np.float32),
                'num_points_in_gt': num_points,
                'difficulty': 0,
                'group_id': group_id,
            })
            group_id += 1

        if missing:
            chunk_size = resolve_chunk_size(
                len(missing), target_elements, min_chunk_size, max_chunk_size)
            gt_points_parts = [[] for _ in missing]
            for start in range(0, points.shape[0], chunk_size):
                chunk = points[start:start + chunk_size]
                for local_idx, box in enumerate(boxes):
                    mask = points_in_rotated_box(chunk[:, :3], box)
                    if mask.any():
                        gt_points_parts[local_idx].append(chunk[mask].copy())

            for local_idx, item in enumerate(missing):
                gt_idx, label, name, db_box, rel_path, abs_path = item
                if gt_points_parts[local_idx]:
                    gt_points = np.concatenate(gt_points_parts[local_idx], axis=0)
                else:
                    gt_points = np.zeros((0, load_dim), dtype=np.float32)
                gt_points = subsample_points(gt_points, max_points, scene_id, gt_idx)
                gt_points[:, :3] -= db_box[:3].astype(np.float32)
                write_bin_atomic(abs_path, gt_points)
                all_db_infos[name].append({
                    'name': name,
                    'path': rel_path.as_posix(),
                    'image_idx': scene_id,
                    'gt_idx': gt_idx,
                    'box3d_lidar': db_box.astype(np.float32),
                    'num_points_in_gt': int(gt_points.shape[0]),
                    'difficulty': 0,
                    'group_id': group_id,
                })
                group_id += 1

    tmp_db_info_path = db_info_path.with_suffix('.tmp.pkl')
    mmengine.dump(all_db_infos, tmp_db_info_path)
    tmp_db_info_path.replace(db_info_path)
    write_json_atomic(manifest_path, manifest)
    return {name: len(items) for name, items in all_db_infos.items()}


def output_shape(pc_range, voxel_size):
    x_cells = int(round((pc_range[3] - pc_range[0]) / voxel_size[0]))
    y_cells = int(round((pc_range[4] - pc_range[1]) / voxel_size[1]))
    return [y_cells, x_cells]


def assigners_for_classes(class_names):
    return '[' + ', '.join(
        "dict(type='Max3DIoUAssigner', "
        "iou_calculator=dict(type='BboxOverlapsNearest3D'), "
        "pos_iou_thr=0.5, neg_iou_thr=0.35, min_pos_iou=0.35, "
        "ignore_iof_thr=-1)" for _ in class_names) + ']'


def assigners_3d_for_classes(class_names):
    return '[' + ', '.join(
        "dict(type='Max3DIoUAssigner', "
        "iou_calculator=dict(type='BboxOverlaps3D', coordinate='lidar'), "
        "pos_iou_thr=0.55, neg_iou_thr=0.55, min_pos_iou=0.55, "
        "ignore_iof_thr=-1, match_low_quality=False)" for _ in class_names) + ']'


def validation_config_blocks(has_validation, val_interval):
    if has_validation:
        checkpoint_hook = """    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        save_best='labelcloud/mAP@0.5',
        rule='greater'),"""
        train_cfg = (
            f"train_cfg = dict(by_epoch=True, max_epochs={{epochs}}, "
            f"val_interval={val_interval})")
        val_block = """val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='labelcloud_infos_val.pkl',
        data_prefix=dict(pts=''),
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        metainfo=metainfo,
        box_type_3d={box_type!r},
        {axis_align_line}backend_args=backend_args))
test_dataloader = val_dataloader
val_evaluator = dict(
    type='LabelCloudMetric',
    classes=class_names,
    iou_thresholds=[0.25, 0.5])
test_evaluator = val_evaluator"""
        loop_block = """val_cfg = dict()
test_cfg = dict()"""
    else:
        checkpoint_hook = """    checkpoint=dict(type='CheckpointHook', interval=1),"""
        train_cfg = "train_cfg = dict(by_epoch=True, max_epochs={epochs})"
        val_block = """val_dataloader = None
val_evaluator = None
test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='labelcloud_infos_train.pkl',
        data_prefix=dict(pts=''),
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        metainfo=metainfo,
        box_type_3d={box_type!r},
        {axis_align_line}backend_args=backend_args))
test_evaluator = None"""
        loop_block = """val_cfg = None
test_cfg = dict()"""
    return checkpoint_hook, train_cfg, val_block, loop_block


def write_sparse_rgb_model_config(path, args, class_names, pc_range, load_dim,
                                  use_dim_cfg, first_sched_epochs,
                                  second_sched_epochs, has_validation=True):
    if args.point_features != 'xyzrgb':
        raise ValueError('TR3D/FCAF3D require --point-features xyzrgb.')

    data_root = Path(args.out_dir).as_posix().rstrip('/') + '/'
    sample_points = args.sample_points
    sparse_voxel_size = args.sparse_voxel_size
    train_repeat_times = getattr(
        args, 'train_repeat_times', default_train_repeat_times(args.model))
    val_interval = getattr(args, 'val_interval', DEFAULT_VAL_INTERVAL)
    use_gt_database = should_create_gt_database(args)
    imports = [
        'projects.labelcloud.labelcloud_dataset',
        'projects.labelcloud.metrics',
    ]
    if args.model == 'tr3d':
        imports.append('projects.TR3D.tr3d')

    sample_transform = ''
    test_sample_transform = ''
    if sample_points > 0:
        sample_transform = (
            f"    dict(type='TR3DPointSample', num_points={sample_points}),\n"
            if args.model == 'tr3d' else
            f"    dict(type='PointSample', num_points={sample_points}),\n")
        test_sample_transform = (
            f"    dict(type='PointSample', num_points=sample_points),\n")

    aug_steps = ''
    if args.aug_mode == 'full':
        aug_steps = """    dict(type='RandomFlip3D', sync_2d=False,
         flip_ratio_bev_horizontal=0.5, flip_ratio_bev_vertical=0.5),
    dict(type='GlobalRotScaleTrans', rot_range=[-0.087266, 0.087266],
         scale_ratio_range=[0.9, 1.1], translation_std=[0.1, 0.1, 0.1],
         shift_height=False),
"""
    elif args.aug_mode == 'safe':
        aug_steps = """    dict(type='GlobalRotScaleTrans', rot_range=[0, 0],
         scale_ratio_range=[0.95, 1.05], translation_std=[0, 0, 0],
         shift_height=False),
"""

    if args.model == 'tr3d':
        model_block = f"""
model = dict(
    type='MinkSingleStage3DDetector',
    data_preprocessor=dict(type='Det3DDataPreprocessor'),
    backbone=dict(
        type='TR3DMinkResNet',
        in_channels=3,
        depth=34,
        norm='batch',
        num_planes=(64, 128, 128, 128)),
    neck=dict(
        type='TR3DNeck',
        in_channels=(64, 128, 128, 128),
        out_channels=128),
    bbox_head=dict(
        type='TR3DHead',
        in_channels=128,
        voxel_size=sparse_voxel_size,
        pts_center_threshold=6,
        num_reg_outs=8,
        bbox_loss=dict(
            type='TR3DRotatedIoU3DLoss', mode='diou', reduction='none'),
        label2level={[0 for _ in class_names]}),
    train_cfg=dict(),
    test_cfg=dict(nms_pre=1000, iou_thr=0.5, score_thr=0.01))
"""
    else:
        model_block = f"""
model = dict(
    type='MinkSingleStage3DDetector',
    data_preprocessor=dict(type='Det3DDataPreprocessor'),
    backbone=dict(type='MinkResNet', in_channels=3, depth=34),
    bbox_head=dict(
        type='FCAF3DHead',
        in_channels=(64, 128, 256, 512),
        out_channels=128,
        voxel_size=sparse_voxel_size,
        pts_prune_threshold=100000,
        pts_assign_threshold=27,
        pts_center_threshold=18,
        num_classes={len(class_names)},
        num_reg_outs=8,
        center_loss=dict(type='mmdet.CrossEntropyLoss', use_sigmoid=True),
        bbox_loss=dict(type='RotatedIoU3DLoss'),
        cls_loss=dict(type='mmdet.FocalLoss')),
    train_cfg=dict(),
    test_cfg=dict(nms_pre=1000, iou_thr=0.5, score_thr=0.01))
"""

    checkpoint_hook, train_cfg_template, val_block_template, loop_block = (
        validation_config_blocks(has_validation, val_interval))
    val_block = val_block_template.format(
        box_type='Depth',
        axis_align_line='axis_align_boxes=False,\n        ')
    train_cfg_line = train_cfg_template.format(epochs=args.epochs)

    text = f"""custom_imports = dict(
    imports={py_list(imports)},
    allow_failed_imports=False)
default_scope = 'mmdet3d'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
{checkpoint_hook}
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='Det3DVisualizationHook'))
custom_hooks = [dict(type='EmptyCacheHook', after_iter=True)]
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
load_from = None
resume = False

dataset_type = 'LabelCloudDataset'
data_root = {data_root!r}
class_names = {py_list(class_names)}
metainfo = dict(classes=class_names)
input_modality = dict(use_lidar=True, use_camera=False)
backend_args = None
point_cloud_range = {pc_range}
point_features = {args.point_features!r}
load_dim = {load_dim}
use_dim = {use_dim_cfg}
sample_points = {sample_points}
sparse_voxel_size = {sparse_voxel_size!r}
voxel_size = {sparse_rgb_voxel_size(args)}
train_lr = {args.train_lr!r}
train_lr_peak_ratio = {args.train_lr_peak_ratio!r}
train_repeat_times = {train_repeat_times}
val_interval = {val_interval}
aug_mode = {args.aug_mode!r}
use_gt_database = {use_gt_database!r}
has_validation = {has_validation!r}

train_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='DEPTH', shift_height=False,
         use_color=True, load_dim=load_dim, use_dim=use_dim,
         backend_args=backend_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
{sample_transform}{aug_steps}
    dict(type='Pack3DDetInputs',
         keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='DEPTH', shift_height=False,
         use_color=True, load_dim=load_dim, use_dim=use_dim,
         backend_args=backend_args),
{test_sample_transform}
    dict(type='Pack3DDetInputs', keys=['points'])
]

train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=train_repeat_times,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file='labelcloud_infos_train.pkl',
            data_prefix=dict(pts=''),
            pipeline=train_pipeline,
            modality=input_modality,
            test_mode=False,
            filter_empty_gt=False,
            metainfo=metainfo,
            box_type_3d='Depth',
            axis_align_boxes=False,
            backend_args=backend_args)))
{val_block}
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(type='Det3DLocalVisualizer', vis_backends=vis_backends,
                  name='visualizer')

{model_block}

lr = train_lr
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, weight_decay=0.0001),
    clip_grad=dict(max_norm=10, norm_type=2))
train_cfg = dict(by_epoch=True, max_epochs={args.epochs}, val_interval=val_interval)
val_cfg = dict()
test_cfg = dict()
param_scheduler = [
    dict(type='CosineAnnealingLR', T_max={first_sched_epochs},
         eta_min=lr * train_lr_peak_ratio, begin=0, end={first_sched_epochs},
         by_epoch=True, convert_to_iter_based=True),
    dict(type='CosineAnnealingLR', T_max={second_sched_epochs},
         eta_min=lr * 1e-4, begin={first_sched_epochs},
         end={args.epochs}, by_epoch=True, convert_to_iter_based=True),
    dict(type='CosineAnnealingMomentum', T_max={first_sched_epochs},
         eta_min=0.85 / 0.95, begin=0, end={first_sched_epochs},
         by_epoch=True, convert_to_iter_based=True),
    dict(type='CosineAnnealingMomentum', T_max={second_sched_epochs},
         eta_min=1, begin={first_sched_epochs}, end={args.epochs},
         by_epoch=True, convert_to_iter_based=True)
]
auto_scale_lr = dict(enable=False, base_batch_size=16)
"""
    text = text.replace(
        f"train_cfg = dict(by_epoch=True, max_epochs={args.epochs}, val_interval=val_interval)\n"
        "val_cfg = dict()\n"
        "test_cfg = dict()",
        f"{train_cfg_line}\n{loop_block}")
    write_text_atomic(path, text)


def write_model_config(path, args, class_names, pc_range, voxel_size, stats,
                       has_validation=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.out_dir).as_posix().rstrip('/') + '/'
    db_info = data_root + 'labelcloud_dbinfos_train.pkl'
    load_dim, use_dim, raw_feature_dims = point_feature_dims(args.point_features)
    use_dim_cfg = use_dim if len(use_dim) != load_dim else load_dim
    input_channels = len(use_dim)
    train_repeat_times = getattr(
        args, 'train_repeat_times', default_train_repeat_times(args.model))
    val_interval = getattr(args, 'val_interval', DEFAULT_VAL_INTERVAL)
    use_gt_database = should_create_gt_database(args)
    use_object_sample = use_gt_database and args.aug_mode == 'full'
    first_sched_epochs, second_sched_epochs = scheduler_epochs(args.epochs)
    if args.model in SPARSE_RGB_MODELS:
        write_sparse_rgb_model_config(path, args, class_names, pc_range,
                                      load_dim, use_dim_cfg,
                                      first_sched_epochs,
                                      second_sched_epochs, has_validation)
        return
    min_points = {name: args.min_points_filter for name in class_names}
    sample_groups = {name: args.sample_group_size for name in class_names}
    anchor_sizes = [stats[name]['anchor_size'] for name in class_names]
    anchor_ranges = [[pc_range[0], pc_range[1], stats[name]['anchor_bottom'],
                      pc_range[3], pc_range[4], stats[name]['anchor_bottom']]
                     for name in class_names]
    pointpillars_voxel_size = [
        voxel_size[0], voxel_size[1], pc_range[5] - pc_range[2]
    ]
    model_voxel_size = (voxel_size if args.model == 'pv_rcnn' else
                        pointpillars_voxel_size)
    shape = output_shape(pc_range, model_voxel_size)
    sample_step = ''
    if args.sample_points > 0:
        sample_step = f"    dict(type='PointSample', num_points={args.sample_points}),\n"
    db_sampler_block = ''
    object_sample_step = ''
    if use_object_sample:
        db_sampler_block = f"""
db_sampler = dict(
    data_root=data_root,
    info_path={db_info!r},
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points={min_points}),
    classes=class_names,
    sample_groups={sample_groups},
    points_loader=dict(
            type='LoadPointsFromFile',
            coord_type='LIDAR',
            load_dim={load_dim},
            use_dim={use_dim_cfg},
            backend_args=backend_args),
    backend_args=backend_args)
"""
        object_sample_step = "    dict(type='ObjectSample', db_sampler=db_sampler),\n"
    aug_steps = ''
    if args.aug_mode == 'full':
        aug_steps = """    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(type='GlobalRotScaleTrans', rot_range=[-0.78539816, 0.78539816],
         scale_ratio_range=[0.95, 1.05]),
"""
    elif args.aug_mode == 'safe':
        aug_steps = """    dict(type='GlobalRotScaleTrans', rot_range=[0, 0],
         scale_ratio_range=[0.95, 1.05]),
"""

    if args.model == 'pointpillars':
        model_block = f"""
model = dict(
    type='VoxelNet',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            max_num_points=32,
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            max_voxels=(16000, 40000))),
    voxel_encoder=dict(
        type='PillarFeatureNet',
        in_channels={input_channels},
        feat_channels=[64],
        with_distance=False,
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range),
    middle_encoder=dict(
        type='PointPillarsScatter', in_channels=64, output_shape={shape}),
    backbone=dict(
        type='SECOND',
        in_channels=64,
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],
        out_channels=[64, 128, 256]),
    neck=dict(
        type='SECONDFPN',
        in_channels=[64, 128, 256],
        upsample_strides=[1, 2, 4],
        out_channels=[128, 128, 128]),
    bbox_head=dict(
        type='Anchor3DHead',
        num_classes={len(class_names)},
        in_channels=384,
        feat_channels=384,
        use_direction_classifier=True,
        assign_per_class=True,
        anchor_generator=dict(
            type='AlignedAnchor3DRangeGenerator',
            ranges={anchor_ranges},
            sizes={anchor_sizes},
            rotations=[0, 1.57],
            reshape_out=False),
        diff_rad_by_sin=True,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(
            type='mmdet.SmoothL1Loss', beta=1.0 / 9.0, loss_weight=2.0),
        loss_dir=dict(
            type='mmdet.CrossEntropyLoss', use_sigmoid=False,
            loss_weight=0.2)),
    train_cfg=dict(
        assigner={assigners_for_classes(class_names)},
        allowed_border=0,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        use_rotate_nms=True,
        nms_across_levels=False,
        nms_thr=0.01,
        score_thr=0.1,
        min_bbox_size=0,
        nms_pre=100,
        max_num=50))
"""
    else:
        model_block = f"""
model = dict(
    type='PointVoxelRCNN',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            max_num_points=5,
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            max_voxels=(16000, 40000))),
    voxel_encoder=dict(type='HardSimpleVFE', num_features={input_channels}),
    middle_encoder=dict(
        type='SparseEncoder',
        in_channels={input_channels},
        sparse_shape=[41, {shape[0]}, {shape[1]}],
        order=('conv', 'norm', 'act'),
        encoder_paddings=((0, 0, 0), ((1, 1, 1), 0, 0), ((1, 1, 1), 0, 0),
                          ((0, 1, 1), 0, 0)),
        return_middle_feats=True),
    points_encoder=dict(
        type='VoxelSetAbstraction',
        num_keypoints=2048,
        fused_out_channel=128,
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        voxel_sa_cfgs_list=[
            dict(type='StackedSAModuleMSG', in_channels=16, scale_factor=1,
                 radius=(0.4, 0.8), sample_nums=(16, 16),
                 mlp_channels=((16, 16), (16, 16)), use_xyz=True),
            dict(type='StackedSAModuleMSG', in_channels=32, scale_factor=2,
                 radius=(0.8, 1.2), sample_nums=(16, 32),
                 mlp_channels=((32, 32), (32, 32)), use_xyz=True),
            dict(type='StackedSAModuleMSG', in_channels=64, scale_factor=4,
                 radius=(1.2, 2.4), sample_nums=(16, 32),
                 mlp_channels=((64, 64), (64, 64)), use_xyz=True),
            dict(type='StackedSAModuleMSG', in_channels=64, scale_factor=8,
                 radius=(2.4, 4.8), sample_nums=(16, 32),
                 mlp_channels=((64, 64), (64, 64)), use_xyz=True)
        ],
        rawpoints_sa_cfgs=dict(
            type='StackedSAModuleMSG',
            in_channels={raw_feature_dims},
            radius=(0.4, 0.8),
            sample_nums=(16, 16),
            mlp_channels=((16, 16), (16, 16)),
            use_xyz=True),
        bev_feat_channel=256,
        bev_scale_factor=8),
    backbone=dict(type='SECOND', in_channels=256, layer_nums=[5, 5],
                  layer_strides=[1, 2], out_channels=[128, 256]),
    neck=dict(type='SECONDFPN', in_channels=[128, 256],
              upsample_strides=[1, 2], out_channels=[256, 256]),
    rpn_head=dict(
        type='PartA2RPNHead',
        num_classes={len(class_names)},
        in_channels=512,
        feat_channels=512,
        use_direction_classifier=True,
        dir_offset=0.78539,
        anchor_generator=dict(
            type='Anchor3DRangeGenerator',
            ranges={anchor_ranges},
            sizes={anchor_sizes},
            rotations=[0, 1.57],
            reshape_out=False),
        diff_rad_by_sin=True,
        assigner_per_size=True,
        assign_per_class=True,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
        loss_cls=dict(type='mmdet.FocalLoss', use_sigmoid=True, gamma=2.0,
                      alpha=0.25, loss_weight=1.0),
        loss_bbox=dict(type='mmdet.SmoothL1Loss', beta=1.0 / 9.0,
                       loss_weight=2.0),
        loss_dir=dict(type='mmdet.CrossEntropyLoss', use_sigmoid=False,
                      loss_weight=0.2)),
    roi_head=dict(
        type='PVRCNNRoiHead',
        num_classes={len(class_names)},
        semantic_head=dict(
            type='ForegroundSegmentationHead',
            in_channels=640,
            extra_width=0.1,
            loss_seg=dict(type='mmdet.FocalLoss', use_sigmoid=True,
                          reduction='sum', gamma=2.0, alpha=0.25,
                          activated=True, loss_weight=1.0)),
        bbox_roi_extractor=dict(
            type='Batch3DRoIGridExtractor',
            grid_size=6,
            roi_layer=dict(type='StackedSAModuleMSG', in_channels=128,
                           radius=(0.8, 1.6), sample_nums=(16, 16),
                           mlp_channels=((64, 64), (64, 64)),
                           use_xyz=True, pool_mod='max')),
        bbox_head=dict(
            type='PVRCNNBBoxHead',
            in_channels=128,
            grid_size=6,
            num_classes={len(class_names)},
            class_agnostic=True,
            shared_fc_channels=(256, 256),
            reg_channels=(256, 256),
            cls_channels=(256, 256),
            dropout_ratio=0.3,
            with_corner_loss=True,
            bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
            loss_bbox=dict(type='mmdet.SmoothL1Loss', beta=1.0 / 9.0,
                           reduction='sum', loss_weight=1.0),
            loss_cls=dict(type='mmdet.CrossEntropyLoss', use_sigmoid=True,
                          reduction='sum', loss_weight=1.0))),
    train_cfg=dict(
        rpn=dict(assigner={assigners_for_classes(class_names)},
                 allowed_border=0, pos_weight=-1, debug=False),
        rpn_proposal=dict(nms_pre=9000, nms_post=512, max_num=512,
                          nms_thr=0.8, score_thr=0, use_rotate_nms=True),
        rcnn=dict(
            assigner={assigners_3d_for_classes(class_names)},
            sampler=dict(type='IoUNegPiecewiseSampler', num=128,
                         pos_fraction=0.5, neg_piece_fractions=[0.8, 0.2],
                         neg_iou_piece_thrs=[0.55, 0.1], neg_pos_ub=-1,
                         add_gt_as_proposals=False, return_iou=True),
            cls_pos_thr=0.75,
            cls_neg_thr=0.25)),
    test_cfg=dict(
        rpn=dict(nms_pre=1024, nms_post=100, max_num=100, nms_thr=0.7,
                 score_thr=0, use_rotate_nms=True),
        rcnn=dict(use_rotate_nms=True, use_raw_score=True, nms_thr=0.1,
                  score_thr=0.1)))
"""

    checkpoint_hook, train_cfg_template, val_block_template, loop_block = (
        validation_config_blocks(has_validation, val_interval))
    val_block = val_block_template.format(
        box_type='LiDAR',
        axis_align_line='')
    train_cfg_line = train_cfg_template.format(epochs=args.epochs)

    text = f"""custom_imports = dict(
    imports=['projects.labelcloud.labelcloud_dataset',
             'projects.labelcloud.metrics'],
    allow_failed_imports=False)
default_scope = 'mmdet3d'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
{checkpoint_hook}
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='Det3DVisualizationHook'))
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
load_from = None
resume = False

dataset_type = 'LabelCloudDataset'
data_root = {data_root!r}
class_names = {py_list(class_names)}
metainfo = dict(classes=class_names)
input_modality = dict(use_lidar=True, use_camera=False)
backend_args = None
point_cloud_range = {pc_range}
voxel_size = {model_voxel_size}
point_features = {args.point_features!r}
load_dim = {load_dim}
use_dim = {use_dim_cfg}
sample_points = {args.sample_points}
train_lr = {args.train_lr!r}
train_lr_peak_ratio = {args.train_lr_peak_ratio!r}
train_repeat_times = {train_repeat_times}
val_interval = {val_interval}
aug_mode = {args.aug_mode!r}
use_gt_database = {use_gt_database!r}
has_validation = {has_validation!r}
{db_sampler_block}

train_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR',
         load_dim=load_dim, use_dim=use_dim, backend_args=backend_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
{object_sample_step}{aug_steps}
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
{sample_step}    dict(type='PointShuffle'),
    dict(type='Pack3DDetInputs',
         keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR',
         load_dim=load_dim, use_dim=use_dim, backend_args=backend_args),
    dict(type='MultiScaleFlipAug3D', img_scale=(1333, 800),
         pts_scale_ratio=1, flip=False,
         transforms=[
             dict(type='GlobalRotScaleTrans', rot_range=[0, 0],
                  scale_ratio_range=[1., 1.], translation_std=[0, 0, 0]),
             dict(type='RandomFlip3D'),
             dict(type='PointsRangeFilter',
                  point_cloud_range=point_cloud_range)
         ]),
    dict(type='Pack3DDetInputs', keys=['points'])
]

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=train_repeat_times,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file='labelcloud_infos_train.pkl',
            data_prefix=dict(pts=''),
            pipeline=train_pipeline,
            modality=input_modality,
            test_mode=False,
            metainfo=metainfo,
            box_type_3d='LiDAR',
            backend_args=backend_args)))
{val_block}
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(type='Det3DLocalVisualizer', vis_backends=vis_backends,
                  name='visualizer')

{model_block}

lr = train_lr
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, betas=(0.95, 0.85), weight_decay=0.01),
    clip_grad=dict(max_norm=35, norm_type=2))
train_cfg = dict(by_epoch=True, max_epochs={args.epochs}, val_interval=val_interval)
val_cfg = dict()
test_cfg = dict()
param_scheduler = [
    dict(type='CosineAnnealingLR', T_max={first_sched_epochs},
         eta_min=lr * train_lr_peak_ratio, begin=0, end={first_sched_epochs},
         by_epoch=True, convert_to_iter_based=True),
    dict(type='CosineAnnealingLR', T_max={second_sched_epochs},
         eta_min=lr * 1e-4, begin={first_sched_epochs},
         end={args.epochs}, by_epoch=True, convert_to_iter_based=True),
    dict(type='CosineAnnealingMomentum', T_max={first_sched_epochs},
         eta_min=0.85 / 0.95, begin=0, end={first_sched_epochs},
         by_epoch=True, convert_to_iter_based=True),
    dict(type='CosineAnnealingMomentum', T_max={second_sched_epochs},
         eta_min=1, begin={first_sched_epochs}, end={args.epochs},
         by_epoch=True, convert_to_iter_based=True)
]
auto_scale_lr = dict(enable=False, base_batch_size=16)
"""
    text = text.replace(
        f"train_cfg = dict(by_epoch=True, max_epochs={args.epochs}, val_interval=val_interval)\n"
        "val_cfg = dict()\n"
        "test_cfg = dict()",
        f"{train_cfg_line}\n{loop_block}")
    write_text_atomic(path, text)


def main():
    args = parse_args()
    if args.min_points_in_gt < 0:
        raise ValueError('--min-points-in-gt must be >= 0.')
    if args.train_lr <= 0:
        raise ValueError('--train-lr must be positive.')
    if args.train_lr_peak_ratio <= 0:
        raise ValueError('--train-lr-peak-ratio must be positive.')
    args.train_repeat_times = resolve_train_repeat_times(
        args.train_repeat_times, args.model)
    args.val_interval = resolve_val_interval(args.val_interval, args.epochs)
    out_dir = Path(args.out_dir)
    if args.overwrite:
        for name in [
                'points', 'ImageSets', 'labelcloud_infos_train.pkl',
                'labelcloud_infos_val.pkl', 'labelcloud_infos_trainval.pkl',
                'labelcloud_dbinfos_train.pkl', 'labelcloud_gt_database',
                'conversion_summary.json'
        ]:
            path = out_dir / name
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()

    class_file, pointcloud_dir, label_dir = resolve_inputs(args)
    class_names = load_classes(class_file)
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    points_out = out_dir / 'points'
    imagesets_out = out_dir / 'ImageSets'
    points_out.mkdir(parents=True, exist_ok=True)
    imagesets_out.mkdir(parents=True, exist_ok=True)

    label_files = sorted(p for p in label_dir.glob('*.json')
                         if p.name != '_classes.json')
    if not label_files:
        raise FileNotFoundError(f'No label JSON files found in {label_dir}')
    validate_label_classes(label_files, class_names)

    label_records = []
    seen_scene_ids = {}
    for label_path in label_files:
        data, rows, extents, dims_part, bottoms_part = read_label_file(
            label_path, class_names)
        pcd_name = data.get('filename') or (label_path.stem + '.pcd')
        pcd_path = pointcloud_dir / pcd_name
        if not pcd_path.exists():
            raise FileNotFoundError(f'Missing PCD for {label_path}: {pcd_path}')

        scene_id = scene_id_from_pcd_name(pcd_name)
        if scene_id in seen_scene_ids:
            raise ValueError(
                f'Duplicate scene id {scene_id!r}: {pcd_name!r} from '
                f'{label_path} conflicts with {seen_scene_ids[scene_id]}. Rename '
                'the PCD file or provide unique filenames.')
        seen_scene_ids[scene_id] = label_path
        label_records.append(
            (label_path, rows, extents, dims_part, bottoms_part, pcd_path,
             scene_id))

    extents_by_scene = {}
    dims_by_scene = {}
    bottoms_by_scene = {}
    all_dims_by_class = {name: [] for name in class_names}
    scene_ids = []
    info_by_scene = {}
    dropped_boxes_by_reason = {}
    dropped_boxes_by_class = {}

    print(f'Converting {len(label_records)} scenes with classes: {class_names}')
    for label_path, rows, extents, dims_part, bottoms_part, pcd_path, scene_id in label_records:
        scene_ids.append(scene_id)
        bin_path = points_out / f'{scene_id}.bin'
        point_count = num_points_in_bin(bin_path)
        converted_points = False
        points = None
        if point_count is None or output_is_stale(bin_path, [pcd_path]):
            points = read_pcd_as_xyzrgb(pcd_path)
            write_bin_atomic(bin_path, points)
            point_count = points.shape[0]
            converted_points = True
        elif args.min_points_in_gt > 0:
            points = read_bin_points(bin_path)

        if args.min_box_dim > 0 or args.min_points_in_gt > 0:
            points_xyz = None if points is None else points[:, :3]
            (
                rows, extents, dims_part, bottoms_part, scene_dropped_reason,
                scene_dropped_class
            ) = filter_label_rows(rows, class_names, points_xyz,
                                  args.min_box_dim, args.min_points_in_gt)
            for reason, count in scene_dropped_reason.items():
                dropped_boxes_by_reason[reason] = (
                    dropped_boxes_by_reason.get(reason, 0) + count)
            for name, count in scene_dropped_class.items():
                dropped_boxes_by_class[name] = (
                    dropped_boxes_by_class.get(name, 0) + count)

        source_mtime_ns = max(pcd_path.stat().st_mtime_ns,
                              label_path.stat().st_mtime_ns,
                              class_file.stat().st_mtime_ns)
        info_by_scene[scene_id] = build_data_info(
            scene_id, 6, rows, class_to_idx, source_mtime_ns)
        extents_by_scene[scene_id] = extents
        dims_by_scene[scene_id] = dims_part
        bottoms_by_scene[scene_id] = bottoms_part
        for name in class_names:
            all_dims_by_class[name].extend(dims_part[name])

        status = 'points' if converted_points else 'up-to-date'
        print(f'  {scene_id}: points={point_count} boxes={len(rows)} {status}')

    splits = split_scenes(scene_ids, args.train_ratio, args.val_ratio, args.seed)
    for split_name, ids in splits.items():
        write_text_atomic(imagesets_out / f'{split_name}.txt',
                          ''.join(f'{x}\n' for x in ids))

    train_infos = [info_by_scene[x] for x in splits['train']]
    val_infos = [info_by_scene[x] for x in splits['val']]
    train_extents = []
    train_dims_by_class = {name: [] for name in class_names}
    train_bottoms_by_class = {name: [] for name in class_names}
    for scene_id in splits['train']:
        train_extents.extend(extents_by_scene[scene_id])
        for name in class_names:
            train_dims_by_class[name].extend(dims_by_scene[scene_id][name])
            train_bottoms_by_class[name].extend(
                bottoms_by_scene[scene_id][name])
    total_boxes = total_boxes_by_class(all_dims_by_class)
    train_boxes = total_boxes_by_class(train_dims_by_class)
    split_counts = {k: len(v) for k, v in splits.items()}
    if total_boxes == 0:
        raise ValueError(
            'No valid GT boxes remain after conversion filters. '
            f'dropped_boxes_by_reason={dropped_boxes_by_reason}, '
            f'dropped_boxes_by_class={dropped_boxes_by_class}. '
            'Lower MIN_BOX_DIM/MIN_POINTS_IN_GT or inspect the labelCloud labels.')
    if train_boxes == 0:
        raise ValueError(
            'Train split has no valid GT boxes after conversion filters. '
            f'splits={split_counts}, '
            f'dropped_boxes_by_reason={dropped_boxes_by_reason}, '
            f'dropped_boxes_by_class={dropped_boxes_by_class}. '
            'Adjust TRAIN_RATIO/VAL_RATIO/SPLIT_SEED or lower MIN_BOX_DIM/MIN_POINTS_IN_GT.')

    save_info(out_dir / 'labelcloud_infos_train.pkl', class_names, train_infos)
    save_info(out_dir / 'labelcloud_infos_val.pkl', class_names, val_infos)
    save_info(out_dir / 'labelcloud_infos_trainval.pkl', class_names,
              train_infos + val_infos)

    db_counts = {}
    create_database = should_create_gt_database(args)
    if create_database:
        db_counts = create_gt_database(
            out_dir,
            train_infos,
            class_names,
            rebuild=args.overwrite or args.rebuild_gt_database,
            target_elements=args.gt_database_target_elements,
            min_chunk_size=args.gt_database_min_chunk_size,
            max_chunk_size=args.gt_database_max_chunk_size,
            max_points=args.gt_database_max_points)
    pc_range, voxel_size = aligned_point_cloud_range(
        train_extents, args.range_margin, args.voxel_xy)
    stats = class_stats(class_names, train_dims_by_class,
                        train_bottoms_by_class)
    model_cfg = args.model_cfg or (out_dir / 'cfgs' /
                                   f'{tagged_name(args.model, args.extra_tag)}.py')
    has_validation = bool(val_infos)
    write_model_config(
        model_cfg, args, class_names, pc_range, voxel_size, stats,
        has_validation=has_validation)
    if args.model in SPARSE_RGB_MODELS:
        model_voxel_size = sparse_rgb_voxel_size(args)
    elif args.model == 'pv_rcnn':
        model_voxel_size = voxel_size
    else:
        model_voxel_size = [
            voxel_size[0], voxel_size[1], pc_range[5] - pc_range[2]
        ]

    summary = {
        'classes': class_names,
        'num_scenes': len(scene_ids),
        'splits': {k: len(v) for k, v in splits.items()},
        'num_boxes_by_class': {
            k: len(v)
            for k, v in all_dims_by_class.items()
        },
        'train_num_boxes_by_class': {
            k: len(v)
            for k, v in train_dims_by_class.items()
        },
        'gt_database_counts': db_counts,
        'gt_database_created': create_database,
        'dropped_boxes_by_reason': dropped_boxes_by_reason,
        'dropped_boxes_by_class': dropped_boxes_by_class,
        'min_box_dim': args.min_box_dim,
        'min_points_in_gt': args.min_points_in_gt,
        'point_cloud_range': pc_range,
        'voxel_size': voxel_size,
        'model_voxel_size': model_voxel_size,
        'stats_source': 'train',
        'aug_mode': args.aug_mode,
        'gt_database': args.gt_database,
        'skip_gt_database': args.skip_gt_database,
        'point_features': args.point_features,
        'load_dim': point_feature_dims(args.point_features)[0],
        'extra_tag': args.extra_tag,
        'sparse_voxel_size': args.sparse_voxel_size,
        'train_lr': args.train_lr,
        'train_lr_peak_ratio': args.train_lr_peak_ratio,
        'train_repeat_times': args.train_repeat_times,
        'val_interval': args.val_interval,
        'use_dim': point_feature_dims(args.point_features)[1],
        'has_validation': has_validation,
        'model': args.model,
        'model_cfg': Path(model_cfg).as_posix(),
        'data_root': out_dir.as_posix(),
    }
    write_json_atomic(out_dir / 'conversion_summary.json', summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
