import ast
import json
import math
import pickle
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tools.convert_labelcloud_to_custom as converter
import tools.infer_labelcloud as infer_labelcloud
import tools.train_labelcloud_pipeline as train_pipeline
from projects.labelcloud.utils import (
    class_stats, labelcloud_detection_metrics, lidar_box_iou_3d,
    load_classes, points_in_rotated_box, read_label_file, safe_name)
from tools.convert_labelcloud_to_custom import (
    build_data_info, create_gt_database, epoch_cfg_options, point_feature_dims,
    label_class_counts, resolve_train_repeat_times, resolve_val_interval,
    scheduler_epochs, should_create_gt_database, validate_label_classes,
    write_model_config)
from tools.infer_labelcloud import (
    config_use_dim, default_infer_cfg_file, default_infer_ckpt, parse_args,
    prediction_payload)
from tools.train_labelcloud_pipeline import (
    ensure_train_device, generated_config_has_validation,
    generated_config_metadata, resolve_batch_size, skip_convert_config_mismatches)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def labelcloud_payload(filename='scene.pcd'):
    return {
        'folder': 'pointClouds',
        'filename': filename,
        'path': f'pointClouds\\{filename}',
        'objects': [{
            'name': 'cargo',
            'centroid': {
                'x': 1.0,
                'y': 2.0,
                'z': 3.0,
            },
            'dimensions': {
                'length': 4.0,
                'width': 2.0,
                'height': 1.0,
            },
            'rotations': {
                'x': 0,
                'y': 0,
                'z': 450,
            },
        }],
    }


def test_read_label_file_matches_labelcloud_centroid_abs(tmp_path):
    label_path = tmp_path / 'labels' / 'scene.json'
    write_json(label_path, labelcloud_payload())

    data, rows, extents, dims_by_class, bottoms_by_class = read_label_file(
        label_path, ['cargo'])

    assert data['filename'] == 'scene.pcd'
    assert len(rows) == 1
    x, y, z, dx, dy, dz, heading, name = rows[0]
    assert (x, y, z, dx, dy, dz, name) == (1.0, 2.0, 3.0, 4.0, 2.0, 1.0,
                                           'cargo')
    assert math.isclose(heading, math.pi / 2, rel_tol=1e-6)
    assert dims_by_class['cargo'] == [(4.0, 2.0, 1.0)]
    assert bottoms_by_class['cargo'] == [2.5]
    assert extents[0][2] == 2.5
    assert extents[1][2] == 3.5


def test_labelcloud_literal_unicode_escape_class_names_are_decoded(tmp_path):
    encoded_name = 'U\\u578b\\u6846\\u67b6-4*\\u6c34\\u5e73\\u7ba1\\u9053'
    decoded_name = 'U型框架-4*水平管道'
    class_path = tmp_path / '_classes.json'
    label_path = tmp_path / 'labels' / 'scene.json'
    write_json(class_path, {'classes': [{'name': encoded_name}]})
    payload = labelcloud_payload()
    payload['objects'][0]['name'] = encoded_name
    write_json(label_path, payload)

    class_names = load_classes(class_path)
    assert class_names == [decoded_name]

    _, rows, _, dims_by_class, bottoms_by_class = read_label_file(
        label_path, class_names)
    assert rows[0][7] == decoded_name
    assert dims_by_class[decoded_name] == [(4.0, 2.0, 1.0)]
    assert bottoms_by_class[decoded_name] == [2.5]

    counts = label_class_counts([label_path])
    assert counts[decoded_name] == 1
    assert encoded_name not in counts


def test_labelcloud_unicode_escape_decode_preserves_existing_chinese(tmp_path):
    class_path = tmp_path / '_classes.json'
    write_json(class_path, {'classes': [{'name': '中文\\u578b'}]})

    assert load_classes(class_path) == ['中文型']


def test_read_label_file_rejects_non_yaw_boxes(tmp_path):
    payload = labelcloud_payload()
    payload['objects'][0]['rotations']['x'] = 0.01
    label_path = tmp_path / 'labels' / 'scene.json'
    write_json(label_path, payload)

    with pytest.raises(ValueError, match='Only z-axis rotation'):
        read_label_file(label_path, ['cargo'])


def test_read_label_file_rejects_non_positive_dimensions(tmp_path):
    payload = labelcloud_payload()
    payload['objects'][0]['dimensions']['height'] = 0.0
    label_path = tmp_path / 'labels' / 'scene.json'
    write_json(label_path, payload)

    with pytest.raises(ValueError, match='Box dimensions must be positive'):
        read_label_file(label_path, ['cargo'])


def test_validate_label_classes_reports_all_unknowns(tmp_path):
    label_dir = tmp_path / 'labels'
    write_json(label_dir / 'a.json', {
        'objects': [
            {'name': 'cargo'},
            {'name': 'crate'},
            {'name': 'crate'},
        ]
    })
    write_json(label_dir / 'b.json', {'objects': [{'name': 'pipe'}]})

    with pytest.raises(ValueError) as exc:
        validate_label_classes(sorted(label_dir.glob('*.json')), ['cargo'])

    message = str(exc.value)
    assert '3 boxes across 2 classes' in message
    assert 'crate=2' in message
    assert 'pipe=1' in message


def test_points_in_rotated_box_uses_geometric_center():
    box = np.array([0.0, 0.0, 0.0, 4.0, 2.0, 2.0, math.pi / 2],
                   dtype=np.float64)
    points = np.array([
        [0.0, 0.0, 0.0],
        [0.9, 1.9, 0.9],
        [1.1, 0.0, 0.0],
        [0.0, 2.1, 0.0],
        [0.0, 0.0, 1.1],
    ],
                      dtype=np.float64)

    assert points_in_rotated_box(points, box).tolist() == [
        True, True, False, False, False
    ]


def test_lidar_box_iou_3d_handles_bottom_center_rotated_boxes():
    boxes_a = np.array([[0.0, 0.0, 0.0, 4.0, 2.0, 2.0, math.pi / 2]],
                       dtype=np.float64)
    boxes_b = np.array([
        [0.0, 0.0, 0.0, 4.0, 2.0, 2.0, math.pi / 2],
        [0.0, 0.0, 1.0, 4.0, 2.0, 2.0, math.pi / 2],
        [10.0, 0.0, 0.0, 4.0, 2.0, 2.0, math.pi / 2],
    ],
                       dtype=np.float64)

    ious = lidar_box_iou_3d(boxes_a, boxes_b)

    assert ious.shape == (1, 3)
    assert math.isclose(ious[0, 0], 1.0, rel_tol=1e-6)
    assert math.isclose(ious[0, 1], 1.0 / 3.0, rel_tol=1e-6)
    assert ious[0, 2] == 0.0

    boxes_with_velocity = np.pad(boxes_a, ((0, 0), (0, 2)), constant_values=3.0)
    assert math.isclose(
        lidar_box_iou_3d(boxes_with_velocity, boxes_b)[0, 0],
        1.0,
        rel_tol=1e-6)


def test_labelcloud_detection_metrics_match_by_class_and_score():
    results = [{
        'pred_boxes':
        np.array([
            [0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0],
            [0.1, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0],
            [5.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0],
        ],
                 dtype=np.float64),
        'pred_scores':
        np.array([0.9, 0.8, 0.7], dtype=np.float64),
        'pred_labels':
        np.array([0, 0, 1], dtype=np.int64),
        'gt_boxes':
        np.array([
            [0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0],
            [5.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0],
        ],
                 dtype=np.float64),
        'gt_labels':
        np.array([0, 1], dtype=np.int64),
    }]

    metrics = labelcloud_detection_metrics(
        results, iou_thresholds=[0.5], classes=['cargo', 'crate'])

    assert metrics['labelcloud/num_scenes'] == 1.0
    assert metrics['labelcloud/num_predictions'] == 3.0
    assert metrics['labelcloud/num_ground_truths'] == 2.0
    assert math.isclose(metrics['labelcloud/precision@0.5'], 2.0 / 3.0)
    assert metrics['labelcloud/recall@0.5'] == 1.0
    assert metrics['labelcloud/cargo_AP@0.5'] == 1.0
    assert metrics['labelcloud/crate_AP@0.5'] == 1.0
    assert metrics['labelcloud/mAP@0.5'] == 1.0

    filtered = labelcloud_detection_metrics(
        results,
        iou_thresholds=[0.5],
        classes=['cargo', 'crate'],
        score_thr=0.75)

    assert filtered['labelcloud/num_predictions'] == 2.0
    assert math.isclose(filtered['labelcloud/precision@0.5'], 0.5)
    assert math.isclose(filtered['labelcloud/recall@0.5'], 0.5)
    assert math.isclose(filtered['labelcloud/mAP@0.5'], 0.5)


def test_safe_name_preserves_ascii_names():
    assert safe_name('cargo') == 'cargo'
    assert safe_name('cargo_01.v2') == 'cargo_01.v2'


def test_safe_name_handles_non_ascii_without_collisions():
    first = safe_name('垂直圆筒带排水口')
    second = safe_name('水平圆筒带排水口')

    assert first.startswith('垂直圆筒带排水口')
    assert second.startswith('水平圆筒带排水口')
    assert first != second
    assert '垂直圆筒带排水口' in first
    assert '水平圆筒带排水口' in second


def test_safe_name_limits_utf8_bytes_for_long_non_ascii_names():
    value = '超长中文分类名' * 20
    slug = safe_name(value)

    assert len(slug.encode('utf-8')) <= 80
    assert slug.startswith('超长中文分类名')


def test_safe_name_hashes_unsafe_ascii_collisions():
    first = safe_name('cargo type')
    second = safe_name('cargo/type')

    assert first.startswith('cargo_type_')
    assert second.startswith('cargo_type_')
    assert first != second


def test_build_data_info_keeps_labelcloud_geometric_centers():
    rows = [(1.0, 2.0, 3.0, 4.0, 2.0, 1.0, 0.25, 'cargo')]

    info = build_data_info('scene', 6, rows, {'cargo': 0}, 123)

    assert info['lidar_points'] == {
        'num_pts_feats': 6,
        'lidar_path': 'points/scene.bin',
    }
    assert info['instances'] == [{
        'bbox_3d': [1.0, 2.0, 3.0, 4.0, 2.0, 1.0, 0.25],
        'bbox_label_3d': 0,
    }]
    assert info['source_mtime_ns'] == 123


def test_create_gt_database_writes_mmdet3d_bottom_center_boxes(tmp_path,
                                                              monkeypatch):
    def dump(obj, path):
        with open(path, 'wb') as f:
            pickle.dump(obj, f)

    monkeypatch.setitem(sys.modules, 'mmengine',
                        types.SimpleNamespace(dump=dump))

    out_dir = tmp_path
    points_dir = out_dir / 'points'
    points_dir.mkdir()
    points = np.array([
        [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
        [0.4, 0.0, 1.2, 0.0, 1.0, 0.0],
        [2.0, 0.0, 1.0, 0.0, 0.0, 1.0],
    ],
                      dtype=np.float32)
    points.tofile(points_dir / 'scene.bin')
    info = {
        'sample_idx': 'scene',
        'lidar_points': {
            'lidar_path': 'points/scene.bin',
        },
        'instances': [{
            'bbox_3d': [0.0, 0.0, 1.0, 2.0, 2.0, 2.0, 0.0],
            'bbox_label_3d': 0,
        }],
        'source_mtime_ns': 0,
    }

    counts = create_gt_database(
        out_dir,
        [info],
        ['cargo'],
        rebuild=True,
        min_chunk_size=1,
        max_chunk_size=2)

    assert counts == {'cargo': 1}
    with open(out_dir / 'labelcloud_dbinfos_train.pkl', 'rb') as f:
        db_infos = pickle.load(f)
    db_info = db_infos['cargo'][0]
    assert db_info['box3d_lidar'].tolist() == [
        0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0
    ]

    db_points = np.fromfile(out_dir / db_info['path'],
                            dtype=np.float32).reshape(-1, 6)
    assert db_points.shape == (2, 6)
    assert np.allclose(db_points[:, :3],
                       np.array([[0.0, 0.0, 1.0], [0.4, 0.0, 1.2]],
                                dtype=np.float32))


def test_create_gt_database_rebuilds_when_generation_params_change(
        tmp_path, monkeypatch):

    def dump(obj, path):
        with open(path, 'wb') as f:
            pickle.dump(obj, f)

    monkeypatch.setitem(sys.modules, 'mmengine',
                        types.SimpleNamespace(dump=dump))

    out_dir = tmp_path
    points_dir = out_dir / 'points'
    points_dir.mkdir()
    points = np.array([
        [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
        [0.4, 0.0, 1.2, 0.0, 1.0, 0.0],
        [-0.4, 0.0, 0.8, 0.0, 0.0, 1.0],
    ],
                      dtype=np.float32)
    points.tofile(points_dir / 'scene.bin')
    info = {
        'sample_idx': 'scene',
        'lidar_points': {
            'lidar_path': 'points/scene.bin',
        },
        'instances': [{
            'bbox_3d': [0.0, 0.0, 1.0, 2.0, 2.0, 2.0, 0.0],
            'bbox_label_3d': 0,
        }],
        'source_mtime_ns': 0,
    }

    create_gt_database(
        out_dir,
        [info],
        ['cargo'],
        rebuild=True,
        min_chunk_size=1,
        max_chunk_size=2,
        max_points=0)
    create_gt_database(
        out_dir,
        [info],
        ['cargo'],
        rebuild=False,
        min_chunk_size=1,
        max_chunk_size=2,
        max_points=1)

    with open(out_dir / 'labelcloud_dbinfos_train.pkl', 'rb') as f:
        db_infos = pickle.load(f)

    db_info = db_infos['cargo'][0]
    assert db_info['num_points_in_gt'] == 1
    db_points = np.fromfile(out_dir / db_info['path'],
                            dtype=np.float32).reshape(-1, 6)
    assert db_points.shape == (1, 6)
    manifest = json.loads(
        (out_dir / 'labelcloud_gt_database_manifest.json').read_text())
    assert manifest['max_points'] == 1


def test_point_feature_dims_are_explicit():
    assert point_feature_dims('xyzrgb') == (6, [0, 1, 2, 3, 4, 5], 3)
    assert point_feature_dims('xyzi') == (6, [0, 1, 2, 3], 1)
    with pytest.raises(ValueError):
        point_feature_dims('xyz')


def test_config_use_dim_reads_config_object(monkeypatch):
    cfg = types.SimpleNamespace(use_dim=[0, 1, 2, 5])
    monkeypatch.setattr(infer_labelcloud, '_load_mmengine_config',
                        lambda cfg_file: cfg)

    assert config_use_dim('labelcloud_pointpillars.py') == [0, 1, 2, 5]


def test_config_use_dim_reads_test_pipeline(monkeypatch):
    cfg = types.SimpleNamespace(
        use_dim=None,
        test_pipeline=[dict(type='LoadPointsFromFile', load_dim=6, use_dim=4)])
    monkeypatch.setattr(infer_labelcloud, '_load_mmengine_config',
                        lambda cfg_file: cfg)

    assert config_use_dim('labelcloud_pv_rcnn.py') == [0, 1, 2, 3]


def test_config_use_dim_requires_explicit_config(monkeypatch):
    monkeypatch.setattr(infer_labelcloud, '_load_mmengine_config',
                        lambda cfg_file: types.SimpleNamespace())

    with pytest.raises(ValueError, match='use_dim'):
        config_use_dim('labelcloud_tr3d.py')


def test_default_infer_paths_use_only_infer_env(monkeypatch):
    monkeypatch.setenv('CFG_FILE', 'data/labelcloud/cfgs/labelcloud_tr3d.py')
    monkeypatch.delenv('INFER_CFG_FILE', raising=False)
    monkeypatch.delenv('INFER_CKPT', raising=False)

    assert default_infer_cfg_file() is None
    assert default_infer_ckpt() is None

    monkeypatch.setenv('INFER_CFG_FILE', '/workspace/checkpoints/deploy/config.py')
    monkeypatch.setenv('INFER_CKPT', '/workspace/checkpoints/deploy/model.pth')

    assert default_infer_cfg_file() == '/workspace/checkpoints/deploy/config.py'
    assert default_infer_ckpt() == '/workspace/checkpoints/deploy/model.pth'


def test_infer_parser_reads_input_dir_env(monkeypatch):
    monkeypatch.setenv('INFER_CFG_FILE', '/workspace/checkpoints/deploy/config.py')
    monkeypatch.setenv('INFER_CKPT', '/workspace/checkpoints/deploy/model.pth')
    monkeypatch.setenv('INPUT_DIR', '/workspace/infer')
    monkeypatch.setenv('INFER_DIR', './host-infer')
    monkeypatch.setattr(sys, 'argv', ['infer_labelcloud.py'])

    args = parse_args()

    assert args.cfg_file == '/workspace/checkpoints/deploy/config.py'
    assert args.ckpt == '/workspace/checkpoints/deploy/model.pth'
    assert args.input_dir == '/workspace/infer'


def test_write_model_config_outputs_parseable_safe_config(tmp_path):
    cfg_path = tmp_path / 'cfgs' / 'labelcloud_pointpillars.py'
    args = types.SimpleNamespace(
        out_dir=tmp_path / 'labelcloud',
        point_features='xyzrgb',
        gt_database='auto',
        skip_gt_database=False,
        aug_mode='safe',
        min_points_filter=5,
        sample_group_size=15,
        sample_points=-1,
        epochs=80,
        train_lr=0.0003,
        train_lr_peak_ratio=1.0,
        model='pointpillars')
    stats = {'cargo': dict(anchor_size=[4.0, 2.0, 1.0], anchor_bottom=2.5)}

    write_model_config(
        cfg_path,
        args,
        ['cargo'],
        [-1.0, -2.0, 0.0, 5.0, 6.0, 4.0],
        [0.1, 0.1, 0.1],
        stats)
    text = cfg_path.read_text()

    ast.parse(text)
    assert "imports=['projects.labelcloud.labelcloud_dataset'" in text
    assert "save_best='labelcloud/mAP@0.5'" in text
    assert "type='LabelCloudMetric'" in text
    assert "sample_points = -1" in text
    assert "use_gt_database = False" in text
    assert 'max_epochs=80' in text
    assert 'T_max=30' in text
    assert 'T_max=50' in text
    assert 'end=80' in text
    assert 'train_repeat_times = 2' in text
    assert 'val_interval = 5' in text
    assert 'times=train_repeat_times' in text
    assert 'val_interval=5' in text
    assert "ObjectSample" not in text


def test_write_model_config_outputs_tr3d_depth_rgb_config(tmp_path):
    cfg_path = tmp_path / 'cfgs' / 'labelcloud_tr3d.py'
    args = types.SimpleNamespace(
        out_dir=tmp_path / 'labelcloud',
        point_features='xyzrgb',
        gt_database='auto',
        skip_gt_database=False,
        aug_mode='full',
        min_points_filter=5,
        sample_group_size=15,
        sample_points=-1,
        epochs=80,
        sparse_voxel_size=0.015,
        train_lr=0.0003,
        train_lr_peak_ratio=1.0,
        model='tr3d')
    stats = {'cargo': dict(anchor_size=[4.0, 2.0, 1.0], anchor_bottom=2.5)}

    write_model_config(
        cfg_path,
        args,
        ['cargo'],
        [-1.0, -2.0, 0.0, 5.0, 6.0, 4.0],
        [0.1, 0.1, 0.1],
        stats)
    text = cfg_path.read_text()

    ast.parse(text)
    assert "'projects.TR3D.tr3d'" in text
    assert "type='MinkSingleStage3DDetector'" in text
    assert "type='TR3DHead'" in text
    assert "label2level=[0]" in text
    assert "coord_type='DEPTH'" in text
    assert "use_color=True" in text
    assert "box_type_3d='Depth'" in text
    assert "sample_points = -1" in text
    assert "type='TR3DPointSample'" not in text
    assert "use_gt_database = False" in text
    assert "NormalizePointsColor" not in text
    assert "ObjectSample" not in text
    assert 'train_repeat_times = 5' in text
    assert 'val_interval = 5' in text
    assert 'times=train_repeat_times' in text


def test_write_model_config_outputs_fcaf3d_depth_rgb_config(tmp_path):
    cfg_path = tmp_path / 'cfgs' / 'labelcloud_fcaf3d.py'
    args = types.SimpleNamespace(
        out_dir=tmp_path / 'labelcloud',
        point_features='xyzrgb',
        gt_database='auto',
        skip_gt_database=False,
        aug_mode='safe',
        min_points_filter=5,
        sample_group_size=15,
        sample_points=100000,
        epochs=80,
        sparse_voxel_size=0.015,
        train_lr=0.0003,
        train_lr_peak_ratio=1.0,
        model='fcaf3d')
    stats = {'cargo': dict(anchor_size=[4.0, 2.0, 1.0], anchor_bottom=2.5)}

    write_model_config(
        cfg_path,
        args,
        ['cargo'],
        [-1.0, -2.0, 0.0, 5.0, 6.0, 4.0],
        [0.1, 0.1, 0.1],
        stats)
    text = cfg_path.read_text()

    ast.parse(text)
    assert "type='FCAF3DHead'" in text
    assert "num_classes=1" in text
    assert "coord_type='DEPTH'" in text
    assert "box_type_3d='Depth'" in text
    assert "sample_points = 100000" in text
    assert "type='PointSample'" in text
    assert "use_gt_database = False" in text
    assert "NormalizePointsColor" not in text


def test_sparse_rgb_models_require_xyzrgb_and_disable_gt_database(tmp_path):
    args = types.SimpleNamespace(
        out_dir=tmp_path / 'labelcloud',
        point_features='xyzi',
        gt_database='auto',
        skip_gt_database=False,
        aug_mode='full',
        min_points_filter=5,
        sample_group_size=15,
        sample_points=-1,
        epochs=80,
        sparse_voxel_size=0.015,
        train_lr=0.0003,
        train_lr_peak_ratio=1.0,
        model='tr3d')
    stats = {'cargo': dict(anchor_size=[4.0, 2.0, 1.0], anchor_bottom=2.5)}

    with pytest.raises(ValueError, match='require --point-features xyzrgb'):
        write_model_config(
            tmp_path / 'cfgs' / 'bad.py',
            args,
            ['cargo'],
            [-1.0, -2.0, 0.0, 5.0, 6.0, 4.0],
            [0.1, 0.1, 0.1],
            stats)

    args.point_features = 'xyzrgb'
    assert should_create_gt_database(args) is False
    args.gt_database = 'on'
    with pytest.raises(ValueError, match='not supported for TR3D/FCAF3D'):
        should_create_gt_database(args)


def test_scheduler_epochs_keep_two_phase_ratio_and_validate():
    assert scheduler_epochs(40) == (15, 25)
    assert scheduler_epochs(80) == (30, 50)
    assert scheduler_epochs(2) == (1, 1)
    with pytest.raises(ValueError, match='at least 2'):
        scheduler_epochs(1)


def test_epoch_cfg_options_override_scheduler_with_list_indexes():
    assert epoch_cfg_options(80) == [
        'train_cfg.max_epochs=80',
        'param_scheduler.0.T_max=30',
        'param_scheduler.0.end=30',
        'param_scheduler.1.T_max=50',
        'param_scheduler.1.begin=30',
        'param_scheduler.1.end=80',
        'param_scheduler.2.T_max=30',
        'param_scheduler.2.end=30',
        'param_scheduler.3.T_max=50',
        'param_scheduler.3.begin=30',
        'param_scheduler.3.end=80',
    ]


def test_training_cadence_defaults_and_validation_clamp():
    assert resolve_train_repeat_times('auto', 'tr3d') == 5
    assert resolve_train_repeat_times('auto', 'fcaf3d') == 5
    assert resolve_train_repeat_times('auto', 'pointpillars') == 2
    assert resolve_train_repeat_times('7', 'tr3d') == 7
    assert resolve_val_interval('5', 80) == 5
    assert resolve_val_interval('5', 3) == 3
    with pytest.raises(ValueError, match='positive integer'):
        resolve_train_repeat_times('0', 'tr3d')
    with pytest.raises(ValueError, match='positive integer'):
        resolve_val_interval('0', 80)


def test_convert_rejects_duplicate_scene_ids(tmp_path, monkeypatch):
    labelcloud_root = tmp_path / 'labelCloud'
    pointcloud_dir = labelcloud_root / 'pointClouds'
    label_dir = labelcloud_root / 'labels'
    pointcloud_dir.mkdir(parents=True)
    label_dir.mkdir()
    write_json(labelcloud_root / '_classes.json',
               {'classes': [{
                   'name': 'cargo'
               }]})
    (pointcloud_dir / 'scene.pcd').write_text('')
    write_json(label_dir / 'a.json', labelcloud_payload('scene.pcd'))
    write_json(label_dir / 'b.json', labelcloud_payload('scene.pcd'))
    monkeypatch.setattr(converter, 'read_pcd_as_xyzrgb',
                        lambda path: np.zeros((1, 6), dtype=np.float32))
    monkeypatch.setattr(sys, 'argv', [
        'convert_labelcloud_to_custom.py',
        '--labelcloud-root',
        str(labelcloud_root),
        '--out-dir',
        str(tmp_path / 'out'),
        '--skip-gt-database',
    ])

    with pytest.raises(ValueError, match='Duplicate scene id'):
        converter.main()


def test_convert_rejects_empty_gt_after_filters(tmp_path, monkeypatch):
    labelcloud_root = tmp_path / 'labelCloud'
    pointcloud_dir = labelcloud_root / 'pointClouds'
    label_dir = labelcloud_root / 'labels'
    pointcloud_dir.mkdir(parents=True)
    label_dir.mkdir()
    write_json(labelcloud_root / '_classes.json',
               {'classes': [{
                   'name': 'cargo'
               }]})
    (pointcloud_dir / 'scene.pcd').write_text('')
    write_json(label_dir / 'scene.json', labelcloud_payload('scene.pcd'))
    monkeypatch.setattr(converter, 'read_pcd_as_xyzrgb',
                        lambda path: np.zeros((1, 6), dtype=np.float32))
    monkeypatch.setattr(sys, 'argv', [
        'convert_labelcloud_to_custom.py',
        '--labelcloud-root',
        str(labelcloud_root),
        '--out-dir',
        str(tmp_path / 'out'),
        '--model',
        'tr3d',
        '--min-points-in-gt',
        '2',
    ])

    with pytest.raises(ValueError, match='No valid GT boxes remain'):
        converter.main()


def test_class_stats_uses_one_anchor_per_class_for_mmdet3d_assigner():
    stats = class_stats(
        ['cargo'],
        {'cargo': [(2.0, 1.0, 1.0), (4.0, 3.0, 2.0), (100.0, 50.0, 10.0)]},
        {'cargo': [0.0, 1.0, 50.0]})

    assert stats['cargo']['anchor_size'] == [4.0, 3.0, 2.0]
    assert stats['cargo']['anchor_bottom'] == 1.0


def test_train_device_guard_allows_dry_run_but_rejects_real_cpu_training():
    ensure_train_device(0, dry_run=True)
    with pytest.raises(RuntimeError, match='No CUDA GPU is visible'):
        ensure_train_device(0, dry_run=False)


def test_auto_batch_size_defaults_to_two_per_gpu():
    assert resolve_batch_size('auto', 1) == 2
    assert resolve_batch_size('auto', 2) == 4


def test_skip_convert_warns_when_generation_options_do_not_match_config(tmp_path):
    cfg_path = tmp_path / 'labelcloud_pv_rcnn.py'
    cfg_path.write_text("""
point_features = 'xyzrgb'
sample_points = 100000
aug_mode = 'full'
use_gt_database = True
""")
    args = types.SimpleNamespace(
        skip_convert=True,
        model_cfg=cfg_path,
        aug_mode='safe',
        point_features='xyzi',
        sample_points=-1,
        sparse_voxel_size=0.02,
        train_lr=0.0001,
        train_lr_peak_ratio=1.0,
        train_repeat_times=2,
        val_interval=5,
        gt_database='auto',
        skip_gt_database=False,
        model='pv_rcnn')

    assert generated_config_metadata(cfg_path) == {
        'point_features': 'xyzrgb',
        'sample_points': 100000,
        'aug_mode': 'full',
        'use_gt_database': True,
    }
    assert skip_convert_config_mismatches(args) == [
        "aug_mode: config='full', current='safe'",
        "point_features: config='xyzrgb', current='xyzi'",
        'sample_points: config=100000, current=-1',
        'use_gt_database: config=True, current=False',
    ]


def test_skip_convert_ignores_generation_options_when_config_matches(tmp_path):
    cfg_path = tmp_path / 'labelcloud_pv_rcnn.py'
    cfg_path.write_text("""
point_features = 'xyzrgb'
aug_mode = 'full'
""")
    args = types.SimpleNamespace(
        skip_convert=True,
        model_cfg=cfg_path,
        aug_mode='full',
        point_features='xyzrgb',
        sample_points=-1,
        sparse_voxel_size=0.015,
        train_lr=0.0003,
        train_lr_peak_ratio=1.0,
        train_repeat_times=2,
        val_interval=5,
        gt_database='auto',
        skip_gt_database=False,
        model='pv_rcnn')

    assert skip_convert_config_mismatches(args) == []


def test_generated_config_can_be_loaded_by_mmengine_when_available(tmp_path):
    Config = pytest.importorskip('mmengine.config').Config
    pytest.importorskip('mmcv')
    pytest.importorskip('mmdet')
    pytest.importorskip('mmdet3d')

    cfg_path = tmp_path / 'cfgs' / 'labelcloud_pointpillars.py'
    args = types.SimpleNamespace(
        out_dir=tmp_path / 'labelcloud',
        point_features='xyzrgb',
        gt_database='auto',
        skip_gt_database=False,
        aug_mode='safe',
        min_points_filter=5,
        sample_group_size=15,
        sample_points=-1,
        epochs=80,
        train_lr=0.0003,
        train_lr_peak_ratio=1.0,
        model='pointpillars')
    stats = {'cargo': dict(anchor_size=[4.0, 2.0, 1.0], anchor_bottom=2.5)}

    write_model_config(
        cfg_path,
        args,
        ['cargo'],
        [-1.0, -2.0, 0.0, 5.0, 6.0, 4.0],
        [0.1, 0.1, 0.1],
        stats)

    cfg = Config.fromfile(cfg_path)

    assert cfg.dataset_type == 'LabelCloudDataset'
    assert cfg.val_evaluator.type == 'LabelCloudMetric'
    assert cfg.default_hooks.checkpoint.save_best == 'labelcloud/mAP@0.5'


def test_generated_config_can_disable_empty_validation_split(tmp_path):
    cfg_path = tmp_path / 'cfgs' / 'labelcloud_tr3d.py'
    args = types.SimpleNamespace(
        out_dir=tmp_path / 'labelcloud',
        point_features='xyzrgb',
        gt_database='auto',
        skip_gt_database=True,
        aug_mode='safe',
        min_points_filter=5,
        sample_group_size=15,
        sample_points=300000,
        epochs=80,
        sparse_voxel_size=0.015,
        train_lr=0.0003,
        train_lr_peak_ratio=1.0,
        model='tr3d')

    write_model_config(
        cfg_path,
        args,
        ['cargo'],
        [-1.0, -2.0, 0.0, 5.0, 6.0, 4.0],
        [0.1, 0.1, 0.1],
        {},
        has_validation=False)

    namespace = {}
    exec(cfg_path.read_text(), namespace)

    assert namespace['has_validation'] is False
    assert namespace['val_dataloader'] is None
    assert namespace['val_evaluator'] is None
    assert namespace['test_dataloader']['dataset']['ann_file'] == (
        'labelcloud_infos_train.pkl')
    assert namespace['test_evaluator'] is None
    assert namespace['val_cfg'] is None
    assert namespace['test_cfg'] == {}
    assert 'save_best' not in namespace['default_hooks']['checkpoint']
    assert generated_config_has_validation(cfg_path) is False


def test_prediction_payload_exports_geometric_center():
    class ArrayLike:

        def __init__(self, values):
            self.values = np.asarray(values)

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.values

    class Boxes:

        def __init__(self):
            self.tensor = ArrayLike([[1.0, 2.0, 0.5, 4.0, 2.0, 1.0, 0.25]])

    class Pred:

        def __init__(self):
            self.bboxes_3d = Boxes()
            self.scores_3d = ArrayLike([0.8])
            self.labels_3d = ArrayLike([0])

    payload = prediction_payload(
        'scene',
        Pred(),
        ['U\\u578b\\u6846\\u67b6-4*\\u6c34\\u5e73\\u7ba1\\u9053'],
        0.3)

    assert payload['objects'][0]['name'] == 'U型框架-4*水平管道'
    assert payload['objects'][0]['box'] == {
        'x': 1.0,
        'y': 2.0,
        'z': 1.0,
        'dx': 4.0,
        'dy': 2.0,
        'dz': 1.0,
        'heading': 0.25,
    }


def test_infer_help_does_not_import_mmdet3d_runtime():
    result = subprocess.run(
        [sys.executable, 'tools/infer_labelcloud.py', '--help'],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True)

    assert 'Run MMDet3D inference for labelCloud point clouds.' in result.stdout


def test_convert_parser_reads_env_defaults(monkeypatch):
    monkeypatch.setenv('LABELCLOUD_DIR', './lc')
    monkeypatch.setenv('OUT_DIR', 'data/out')
    monkeypatch.setenv('MODEL', 'tr3d')
    monkeypatch.setenv('SAMPLE_POINTS', '123456')
    monkeypatch.setenv('POINT_FEATURES', 'xyzrgb')
    monkeypatch.setenv('AUG_MODE', 'safe')
    monkeypatch.setenv('GT_DATABASE', 'off')
    monkeypatch.setenv('TRAIN_REPEAT_TIMES', '7')
    monkeypatch.setenv('VAL_INTERVAL', '3')
    monkeypatch.setattr(sys, 'argv', ['convert_labelcloud_to_custom.py'])

    args = converter.parse_args()

    assert args.labelcloud_root == './lc'
    assert args.out_dir == 'data/out'
    assert args.model == 'tr3d'
    assert args.sample_points == 123456
    assert args.aug_mode == 'safe'
    assert args.gt_database == 'off'
    assert args.train_repeat_times == '7'
    assert args.val_interval == '3'


def test_train_parser_reads_env_defaults(monkeypatch):
    monkeypatch.setenv('MODEL', 'fcaf3d')
    monkeypatch.setenv('BATCH_SIZE', '1')
    monkeypatch.setenv('MASTER_PORT', '18888')
    monkeypatch.setenv('EPOCHS', '40')
    monkeypatch.setenv('GT_DATABASE', 'off')
    monkeypatch.setenv('GT_DATABASE_TARGET_ELEMENTS', '123')
    monkeypatch.setenv('GT_DATABASE_MIN_CHUNK_SIZE', '4')
    monkeypatch.setenv('GT_DATABASE_MAX_CHUNK_SIZE', '5')
    monkeypatch.setenv('GT_DATABASE_MAX_POINTS', '6')
    monkeypatch.setenv('TRAIN_REPEAT_TIMES', '7')
    monkeypatch.setenv('VAL_INTERVAL', '3')
    monkeypatch.setattr(sys, 'argv', ['train_labelcloud_pipeline.py'])

    args = train_pipeline.parse_args()

    assert args.model == 'fcaf3d'
    assert args.batch_size == '1'
    assert args.master_port == 18888
    assert args.epochs == 40
    assert args.gt_database == 'off'
    assert args.gt_database_target_elements == 123
    assert args.gt_database_min_chunk_size == 4
    assert args.gt_database_max_chunk_size == 5
    assert args.gt_database_max_points == 6
    assert args.train_repeat_times == '7'
    assert args.val_interval == '3'
