#!/usr/bin/env python3
import argparse
import ast
import os
import re
import shutil
import socket
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.convert_labelcloud_to_custom import (
    DEFAULT_VAL_INTERVAL, MODEL_CHOICES, epoch_cfg_options, load_env_file,
    resolve_train_repeat_times, resolve_val_interval,
    should_create_gt_database)


load_env_file()


AUG_MODE_CHOICES = ('full', 'safe', 'none')
GT_DATABASE_CHOICES = ('auto', 'on', 'off')
POINT_FEATURE_CHOICES = ('xyzrgb', 'xyzi')
AMP_DTYPE_CHOICES = ('fp16', 'bf16')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert labelCloud data and train an MMDet3D detector.')
    parser.add_argument('--labelcloud-root', default=os.environ.get('LABELCLOUD_DIR'))
    parser.add_argument('--class-file', default=None)
    parser.add_argument('--pointcloud-dir', default=None)
    parser.add_argument('--label-dir', default=None)
    parser.add_argument('--out-dir', default=os.environ.get('OUT_DIR', 'data/labelcloud'))
    parser.add_argument('--model', choices=MODEL_CHOICES, default=os.environ.get('MODEL', 'tr3d'))
    parser.add_argument('--model-cfg', default=os.environ.get('CFG_FILE'))
    parser.add_argument('--extra-tag', default=os.environ.get('EXTRA_TAG'))
    parser.add_argument('--train-ratio', type=float, default=float(os.environ.get('TRAIN_RATIO', '0.8')))
    parser.add_argument('--val-ratio', type=float, default=float(os.environ.get('VAL_RATIO', '0.2')))
    parser.add_argument('--seed', type=int, default=int(os.environ.get('SPLIT_SEED', '42')))
    parser.add_argument('--batch-size', default=os.environ.get('BATCH_SIZE', 'auto'))
    parser.add_argument('--workers', type=int, default=int(os.environ.get('WORKERS', '2')))
    parser.add_argument('--num-gpus', default=os.environ.get('NUM_GPUS', 'auto'))
    parser.add_argument(
        '--amp',
        dest='amp',
        action='store_true',
        default=env_bool('TRAIN_AMP', False),
        help='Enable MMEngine AMP training.')
    parser.add_argument(
        '--no-amp',
        dest='amp',
        action='store_false',
        help='Disable MMEngine AMP training.')
    parser.add_argument(
        '--amp-dtype',
        choices=AMP_DTYPE_CHOICES,
        default=os.environ.get('TRAIN_AMP_DTYPE', 'fp16').strip().lower(),
        help='AMP autocast dtype for training.')
    parser.add_argument(
        '--master-port',
        type=int,
        default=(int(os.environ['MASTER_PORT'])
                 if os.environ.get('MASTER_PORT') else None))
    parser.add_argument(
        '--epochs',
        type=int,
        default=(int(os.environ['EPOCHS']) if os.environ.get('EPOCHS') else None))
    parser.add_argument(
        '--train-repeat-times',
        default=os.environ.get('TRAIN_REPEAT_TIMES', 'auto'))
    parser.add_argument(
        '--val-interval',
        default=os.environ.get('VAL_INTERVAL', str(DEFAULT_VAL_INTERVAL)))
    parser.add_argument('--work-dir', default=os.environ.get('WORK_DIR'))
    parser.add_argument('--pretrained-model', default=os.environ.get('PRETRAINED_MODEL'))
    parser.set_defaults(resume_ckpt=os.environ.get('RESUME_CKPT'))
    parser.add_argument('--resume-ckpt', dest='resume_ckpt')
    parser.add_argument('--sample-points', type=int, default=int(os.environ.get('SAMPLE_POINTS', '500000')))
    parser.add_argument(
        '--sparse-voxel-size',
        type=float,
        default=float(os.environ.get('SPARSE_VOXEL_SIZE', '0.015')))
    parser.add_argument(
        '--point-features',
        choices=POINT_FEATURE_CHOICES,
        default=os.environ.get('POINT_FEATURES', 'xyzrgb'))
    parser.add_argument(
        '--aug-mode',
        choices=AUG_MODE_CHOICES,
        default=os.environ.get('AUG_MODE', 'full'))
    parser.add_argument(
        '--gt-database',
        choices=GT_DATABASE_CHOICES,
        default=os.environ.get('GT_DATABASE', 'auto'))
    parser.add_argument('--skip-gt-database', action='store_true')
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
        default=float(os.environ.get('MIN_BOX_DIM', '0.0001')))
    parser.add_argument(
        '--min-points-in-gt',
        type=int,
        default=int(os.environ.get('MIN_POINTS_IN_GT', '1')))
    parser.add_argument(
        '--train-lr',
        type=float,
        default=float(os.environ.get('TRAIN_LR', '0.0003')))
    parser.add_argument(
        '--train-lr-peak-ratio',
        type=float,
        default=float(os.environ.get('TRAIN_LR_PEAK_RATIO', '1.0')))
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--skip-convert', action='store_true')
    parser.add_argument('--skip-train', action='store_true')
    parser.add_argument('--reset-output', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args()


def none_if_blank(value):
    return None if value == '' else value


def env_bool(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ('1', 'true', 'yes', 'on')


def is_relative_to(path, parent):
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def normalize_resume_ckpt(args):
    args.resume_ckpt = none_if_blank(args.resume_ckpt)
    if args.resume_ckpt is None:
        return

    if args.resume_ckpt == 'auto' and not args.skip_convert:
        print(
            'WARNING: 完整转换+训练流程会忽略 RESUME_CKPT=auto；'
            '自动续训只在 --skip-convert 训练入口启用。',
            file=sys.stderr)
        args.resume_ckpt = None
        return

    if not args.reset_output:
        return

    if args.resume_ckpt == 'auto':
        print(
            'WARNING: --reset-output 会先清理 work_dir，已禁用 RESUME_CKPT=auto 自动续训。',
            file=sys.stderr)
        args.resume_ckpt = None
        return

    ckpt_path = Path(args.resume_ckpt)
    if is_relative_to(ckpt_path, Path(args.work_dir)):
        raise ValueError(
            f'--reset-output 会删除 work_dir 内的 RESUME_CKPT: {args.resume_ckpt}。'
            '请把 checkpoint 移到 WORK_DIR 外，或清空 RESUME_CKPT。')


def run(cmd, cwd, dry_run=False):
    print('+ ' + ' '.join(str(x) for x in cmd))
    if not dry_run:
        subprocess.run(cmd, cwd=cwd, check=True)


def visible_gpu_count():
    try:
        import torch
    except ImportError:
        return 0
    return torch.cuda.device_count()


def resolve_num_gpus(value):
    if value == 'auto':
        return visible_gpu_count()
    num_gpus = int(value)
    if num_gpus < 0:
        raise ValueError('--num-gpus must be "auto" or a non-negative integer.')
    return num_gpus


def resolve_batch_size(value, num_gpus):
    if value == 'auto':
        return max(1, num_gpus) * 2
    batch_size = int(value)
    if batch_size <= 0:
        raise ValueError('--batch-size must be "auto" or a positive integer.')
    if num_gpus > 1 and batch_size % num_gpus != 0:
        raise ValueError('--batch-size must be divisible by GPU count.')
    return batch_size


def per_device_batch_size(total_batch_size, num_gpus):
    if num_gpus <= 1:
        return total_batch_size
    return total_batch_size // num_gpus


def ensure_train_device(num_gpus, dry_run=False):
    if num_gpus < 1 and not dry_run:
        raise RuntimeError(
            'No CUDA GPU is visible, but labelCloud training requires MMDet3D '
            'CUDA ops. Use --skip-train for conversion-only runs or --dry-run '
            'to inspect the generated command.')


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def tagged_name(model, extra_tag=None):
    name = f'labelcloud_{model}'
    if extra_tag:
        tag = re.sub(r'[^A-Za-z0-9_.-]+', '_', extra_tag).strip('._-')
        if tag:
            name = f'{name}_{tag}'
    return name


def default_model_cfg(out_dir, model, extra_tag=None):
    return (Path(out_dir) / 'cfgs' / f'{tagged_name(model, extra_tag)}.py').as_posix()


def default_work_dir(out_dir, model, extra_tag=None):
    return (Path(out_dir) / 'work_dirs' / tagged_name(model, extra_tag)).as_posix()


def generated_config_metadata(model_cfg):
    path = Path(model_cfg)
    try:
        tree = ast.parse(path.read_text())
    except (OSError, SyntaxError):
        return {}

    metadata = {}
    for stmt in tree.body:
        if not isinstance(stmt, ast.Assign):
            continue
        for target in stmt.targets:
            if not isinstance(target, ast.Name):
                continue
            if target.id not in ('aug_mode', 'point_features', 'sample_points',
                                 'sparse_voxel_size', 'train_lr',
                                 'train_lr_peak_ratio', 'train_repeat_times',
                                 'val_interval', 'use_gt_database',
                                 'has_validation'):
                continue
            try:
                metadata[target.id] = ast.literal_eval(stmt.value)
            except (ValueError, SyntaxError):
                pass
    return metadata


def generated_config_has_validation(model_cfg, default=True):
    metadata = generated_config_metadata(model_cfg)
    has_validation = metadata.get('has_validation')
    return default if has_validation is None else bool(has_validation)


def skip_convert_config_mismatches(args):
    if not args.skip_convert:
        return []
    metadata = generated_config_metadata(args.model_cfg)
    checks = [
        ('aug_mode', args.aug_mode),
        ('point_features', args.point_features),
        ('sample_points', args.sample_points),
        ('sparse_voxel_size', args.sparse_voxel_size),
        ('train_lr', args.train_lr),
        ('train_lr_peak_ratio', args.train_lr_peak_ratio),
        ('use_gt_database', should_create_gt_database(args)),
    ]
    mismatches = []
    for key, current_value in checks:
        config_value = metadata.get(key)
        if config_value is not None and config_value != current_value:
            mismatches.append(
                f'{key}: config={config_value!r}, current={current_value!r}')
    return mismatches


def main():
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    args.labelcloud_root = none_if_blank(args.labelcloud_root)
    args.extra_tag = none_if_blank(args.extra_tag)
    args.model_cfg = none_if_blank(args.model_cfg) or default_model_cfg(
        args.out_dir, args.model, args.extra_tag)
    args.work_dir = none_if_blank(args.work_dir) or default_work_dir(
        args.out_dir, args.model, args.extra_tag)
    args.pretrained_model = none_if_blank(args.pretrained_model)
    args.train_repeat_times = resolve_train_repeat_times(
        args.train_repeat_times, args.model)
    args.val_interval = resolve_val_interval(args.val_interval, args.epochs or 80)
    normalize_resume_ckpt(args)

    mismatches = skip_convert_config_mismatches(args)
    if mismatches:
        print(
            'WARNING: --skip-convert reuses the existing generated config; '
            + '; '.join(mismatches) +
            '. Run pipeline, convert, or reconvert to regenerate the config.',
            file=sys.stderr)

    if not args.skip_convert:
        convert_cmd = [
            sys.executable, 'tools/convert_labelcloud_to_custom.py',
            '--out-dir', args.out_dir,
            '--model', args.model,
            '--model-cfg', args.model_cfg,
            '--extra-tag', args.extra_tag or '',
            '--train-ratio', str(args.train_ratio),
            '--val-ratio', str(args.val_ratio),
            '--seed', str(args.seed),
            '--epochs', str(args.epochs or 80),
            '--train-repeat-times', str(args.train_repeat_times),
            '--val-interval', str(args.val_interval),
            '--sample-points', str(args.sample_points),
            '--sparse-voxel-size', str(args.sparse_voxel_size),
            '--point-features', args.point_features,
            '--aug-mode', args.aug_mode,
            '--gt-database', args.gt_database,
            '--gt-database-target-elements', str(args.gt_database_target_elements),
            '--gt-database-min-chunk-size', str(args.gt_database_min_chunk_size),
            '--gt-database-max-chunk-size', str(args.gt_database_max_chunk_size),
            '--gt-database-max-points', str(args.gt_database_max_points),
            '--min-box-dim', str(args.min_box_dim),
            '--min-points-in-gt', str(args.min_points_in_gt),
            '--train-lr', str(args.train_lr),
            '--train-lr-peak-ratio', str(args.train_lr_peak_ratio),
        ]
        if args.labelcloud_root:
            convert_cmd.extend(['--labelcloud-root', args.labelcloud_root])
        else:
            missing = [
                name for name, value in [
                    ('--class-file', args.class_file),
                    ('--pointcloud-dir', args.pointcloud_dir),
                    ('--label-dir', args.label_dir),
                ] if not value
            ]
            if missing:
                raise ValueError('Missing conversion inputs: ' + ', '.join(missing))
            convert_cmd.extend([
                '--class-file', args.class_file,
                '--pointcloud-dir', args.pointcloud_dir,
                '--label-dir', args.label_dir,
            ])
        if args.overwrite:
            convert_cmd.append('--overwrite')
        if args.skip_gt_database:
            convert_cmd.append('--skip-gt-database')
        if args.rebuild_gt_database:
            convert_cmd.append('--rebuild-gt-database')
        run(convert_cmd, cwd=root, dry_run=args.dry_run)

    if args.skip_train:
        return

    num_gpus = resolve_num_gpus(args.num_gpus)
    ensure_train_device(num_gpus, dry_run=args.dry_run)
    batch_size = resolve_batch_size(args.batch_size, num_gpus)
    device_batch_size = per_device_batch_size(batch_size, num_gpus)
    if args.reset_output and Path(args.work_dir).exists():
        print('+ rm -rf ' + args.work_dir)
        if not args.dry_run:
            shutil.rmtree(args.work_dir)

    has_validation = generated_config_has_validation(
        args.model_cfg, default=args.val_ratio > 0)
    cfg_options = [
        f'train_dataloader.batch_size={device_batch_size}',
        f'train_dataloader.num_workers={args.workers}',
        f'train_dataloader.persistent_workers={args.workers > 0}',
        f'train_dataloader.dataset.times={args.train_repeat_times}',
    ]
    if has_validation:
        cfg_options.extend([
            f'train_cfg.val_interval={args.val_interval}',
            f'val_dataloader.num_workers={max(0, min(args.workers, 2))}',
            f'val_dataloader.persistent_workers={args.workers > 0}',
            f'test_dataloader.num_workers={max(0, min(args.workers, 2))}',
            f'test_dataloader.persistent_workers={args.workers > 0}',
        ])
    else:
        cfg_options.extend([
            'val_cfg=None',
            'val_dataloader=None',
            'val_evaluator=None',
            'test_cfg=None',
            'test_dataloader=None',
            'test_evaluator=None',
            'default_hooks.checkpoint.save_best=None',
        ])
    if args.epochs is not None:
        cfg_options.extend(epoch_cfg_options(args.epochs))
    if args.pretrained_model is not None:
        cfg_options.append(f'load_from={args.pretrained_model!r}')

    train_args = [
        'tools/train.py',
        args.model_cfg,
        '--work-dir', args.work_dir,
        '--cfg-options', *cfg_options,
    ]
    if args.amp:
        train_args.extend(['--amp', '--amp-dtype', args.amp_dtype])
    if args.resume_ckpt is not None:
        train_args.extend(['--resume', args.resume_ckpt])

    if num_gpus > 1:
        master_port = args.master_port if args.master_port is not None else find_free_port()
        cmd = [
            sys.executable, '-m', 'torch.distributed.run',
            '--nproc_per_node', str(num_gpus),
            '--master_port', str(master_port),
            *train_args,
            '--launcher', 'pytorch',
        ]
    else:
        cmd = [sys.executable, *train_args]
    run(cmd, cwd=root, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
