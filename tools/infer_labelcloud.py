#!/usr/bin/env python3
import argparse
from contextlib import nullcontext
import gc
import json
import os
import shutil
import sys
from pathlib import Path, PurePosixPath

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.labelcloud.utils import (
    decode_label_name, points_in_rotated_box, read_pcd_as_xyzrgb,
    write_json_atomic)
from tools.crop_labelcloud_predictions import box_from_object, crop_filename, load_points, save_pcd
from tools.convert_labelcloud_to_custom import load_env_file


load_env_file()


MODEL_CHOICES = ('pv_rcnn', 'pointpillars', 'tr3d', 'fcaf3d')


class InferenceAmpConfigError(ValueError):
    pass


def default_infer_ckpt():
    return none_if_blank(os.environ.get('INFER_CKPT'))


def default_infer_cfg_file():
    return none_if_blank(os.environ.get('INFER_CFG_FILE'))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run MMDet3D inference for labelCloud point clouds.')
    parser.add_argument('--model', choices=MODEL_CHOICES, default=os.environ.get('MODEL', 'tr3d'))
    parser.add_argument('--cfg-file', default=default_infer_cfg_file())
    parser.add_argument('--ckpt', default=default_infer_ckpt())
    parser.add_argument(
        '--input-dir',
        default=none_if_blank(os.environ.get('INPUT_DIR')) or 'infer')
    parser.add_argument('--output-dir', default=os.environ.get('PRED_OUT_DIR'))
    parser.add_argument('--score-thresh', type=float, default=float(os.environ.get('SCORE_THRESH', '0.3')))
    parser.add_argument('--device', default=os.environ.get('DEVICE', 'cuda:0'))
    parser.add_argument(
        '--amp',
        dest='amp',
        action='store_true',
        default=env_bool('INFER_AMP', False),
        help='Enable CUDA autocast during inference.')
    parser.add_argument(
        '--no-amp',
        dest='amp',
        action='store_false',
        help='Disable CUDA autocast during inference.')
    parser.add_argument(
        '--amp-dtype',
        choices=('fp16', 'bf16'),
        default=os.environ.get('INFER_AMP_DTYPE', 'fp16').strip().lower(),
        help='Autocast dtype used when inference AMP is enabled.')
    parser.add_argument(
        '--crop',
        action='store_true',
        help='Write object crop PCDs and crop_path fields. Disabled by default.')
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        default=os.environ.get('SKIP_EXISTING', '').lower() in
        ('1', 'true', 'yes', 'on'),
        help='Skip scenes with an existing predictions.json in output-dir.')
    return parser.parse_args()


def none_if_blank(value):
    return None if value == '' else value


def env_bool(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ('1', 'true', 'yes', 'on')


def default_output_dir(cfg_file):
    return (Path('output') / 'predictions' / Path(cfg_file).stem).as_posix()


def normalize_use_dim(use_dim):
    if use_dim is None:
        return None
    if isinstance(use_dim, int):
        return list(range(use_dim))
    if isinstance(use_dim, (list, tuple)):
        return [int(x) for x in use_dim]
    raise TypeError(f'Unsupported use_dim value: {use_dim!r}')


def _get_cfg_value(cfg, key):
    if cfg is None:
        return None
    if hasattr(cfg, 'get'):
        return cfg.get(key, None)
    return getattr(cfg, key, None)


def _pipeline_use_dim(pipeline):
    if pipeline is None:
        return None
    for step in pipeline:
        if not hasattr(step, 'get'):
            continue
        step_type = step.get('type')
        if step_type in ('LoadPointsFromFile', 'LoadPointsFromDict'):
            if 'use_dim' in step:
                return step['use_dim']
        nested = step.get('transforms')
        nested_use_dim = _pipeline_use_dim(nested)
        if nested_use_dim is not None:
            return nested_use_dim
    return None


def _dataset_pipeline_use_dim(cfg):
    dataloader = _get_cfg_value(cfg, 'test_dataloader')
    if dataloader is None:
        return None
    dataset = _get_cfg_value(dataloader, 'dataset')
    if _get_cfg_value(dataset, 'type') == 'RepeatDataset':
        dataset = _get_cfg_value(dataset, 'dataset')
    pipeline = _get_cfg_value(dataset, 'pipeline')
    return _pipeline_use_dim(pipeline)


def _load_mmengine_config(cfg_file):
    from mmengine.config import Config
    return Config.fromfile(cfg_file)


def config_use_dim(cfg_file):
    cfg = _load_mmengine_config(cfg_file)
    for candidate in (
            _get_cfg_value(cfg, 'use_dim'),
            _pipeline_use_dim(_get_cfg_value(cfg, 'test_pipeline')),
            _dataset_pipeline_use_dim(cfg),
    ):
        if candidate is not None:
            return normalize_use_dim(candidate)
    raise ValueError(f'配置文件缺少推理点特征 use_dim: {cfg_file}')


def input_files(path):
    path = Path(path)
    if path.is_dir():
        files = sorted(path.glob('*.pcd'))
        if not files:
            raise FileNotFoundError(f'No .pcd point clouds found in {path}')
        return files
    return [path]


def prediction_payload(frame_id, pred, class_names, score_thresh):
    bboxes = pred.bboxes_3d.tensor.detach().cpu().numpy()
    scores = pred.scores_3d.detach().cpu().numpy()
    labels = pred.labels_3d.detach().cpu().numpy()
    objects = []
    for idx, (box, score, label) in enumerate(zip(bboxes, scores, labels)):
        if float(score) < score_thresh:
            continue
        label = int(label)
        name = (
            decode_label_name(class_names[label])
            if 0 <= label < len(class_names) else str(label))
        center_box = box.astype(np.float64).copy()
        center_box[2] += center_box[5] / 2.0
        objects.append({
            'index': len(objects),
            'name': name,
            'label': label,
            'score': float(score),
            'box': {
                'x': float(center_box[0]),
                'y': float(center_box[1]),
                'z': float(center_box[2]),
                'dx': float(center_box[3]),
                'dy': float(center_box[4]),
                'dz': float(center_box[5]),
                'heading': float(center_box[6]),
            },
        })
    return {'frame_id': frame_id, 'objects': objects}


def write_prediction_crops(output_dir, scene_id, input_path, payload):
    crop_dir = output_dir / 'crops' / scene_id
    shutil.rmtree(crop_dir, ignore_errors=True)
    xyz, colors = load_points(input_path)
    for obj in payload.get('objects', []):
        mask = points_in_rotated_box(xyz, box_from_object(obj))
        crop_points = xyz[mask]
        crop_colors = colors[mask] if colors is not None else None
        crop_path = crop_dir / crop_filename(obj)
        save_pcd(crop_path, crop_points, crop_colors)
        relative_path = f'crops/{scene_id}/{crop_path.name}'
        obj['crop_path'] = relative_path
    return f'crops/{scene_id}'


def remove_prediction_crops(output_dir, scene_id, payload):
    shutil.rmtree(output_dir / 'crops' / scene_id, ignore_errors=True)
    changed = False
    for obj in payload.get('objects', []):
        if 'crop_path' in obj:
            obj.pop('crop_path', None)
            changed = True
    return changed


def existing_crop_file(output_dir, crop_path):
    if not isinstance(crop_path, str) or not crop_path:
        return None
    relative = PurePosixPath(crop_path)
    if relative.is_absolute() or '..' in relative.parts:
        return None
    candidate = (output_dir / Path(*relative.parts)).resolve()
    root = output_dir.resolve()
    if root != candidate and root not in candidate.parents:
        return None
    return candidate if candidate.is_file() else None


def validate_amp_dtype(dtype_name):
    if dtype_name not in ('fp16', 'bf16'):
        raise InferenceAmpConfigError(f'不支持的推理 AMP dtype: {dtype_name}')


def infer_autocast_context(torch, device_name, enabled, dtype_name):
    if not enabled:
        return nullcontext()
    validate_amp_dtype(dtype_name)
    if torch is None:
        raise InferenceAmpConfigError('推理 AMP 需要 PyTorch。')

    device = torch.device(device_name)
    if device.type != 'cuda':
        raise InferenceAmpConfigError('推理 AMP 只支持 CUDA device。')
    if not torch.cuda.is_available():
        raise InferenceAmpConfigError('推理 AMP 已启用，但当前环境没有可用 CUDA。')

    if dtype_name == 'fp16':
        dtype = torch.float16
    elif dtype_name == 'bf16':
        if not hasattr(torch, 'bfloat16'):
            raise InferenceAmpConfigError('当前 PyTorch 不支持 bfloat16。')
        if hasattr(torch.cuda, 'is_bf16_supported') and not torch.cuda.is_bf16_supported():
            raise InferenceAmpConfigError('当前 CUDA/GPU 不支持 bfloat16 autocast。')
        dtype = torch.bfloat16
    return torch.cuda.amp.autocast(dtype=dtype)


def inference_detector_with_amp(inference_detector, model, points, *, amp, amp_dtype, device):
    try:
        import torch
    except ImportError:
        torch = None

    with infer_autocast_context(torch, device, amp, amp_dtype):
        return inference_detector(model, points)


def main():
    args = parse_args()
    if args.amp:
        validate_amp_dtype(args.amp_dtype)
    cfg_file = none_if_blank(args.cfg_file)
    if cfg_file is None:
        raise ValueError('推理必须通过 --cfg-file 或 INFER_CFG_FILE 指定明确配置文件。')
    ckpt = none_if_blank(args.ckpt)
    if ckpt is None:
        raise ValueError('推理必须通过 --ckpt 或 INFER_CKPT 指定明确 checkpoint 路径。')
    if ckpt == 'auto':
        raise ValueError('推理不支持 --ckpt auto；请使用明确 checkpoint 路径。')
    ckpt_path = Path(ckpt)
    output_dir = Path(none_if_blank(args.output_dir) or default_output_dir(cfg_file))
    use_dim = config_use_dim(cfg_file)

    from mmdet3d.apis import inference_detector, init_model
    try:
        import torch
    except ImportError:
        torch = None

    model = init_model(cfg_file, ckpt_path.as_posix(), device=args.device)
    class_names = [
        decode_label_name(name)
        for name in model.dataset_meta.get('classes', [])
    ]
    files = input_files(args.input_dir)

    summary = []
    for idx, sample_file in enumerate(files):
        scene_dir = output_dir / sample_file.stem
        pred_file = scene_dir / 'predictions.json'
        if args.skip_existing and pred_file.exists():
            print(f'{idx + 1}/{len(files)} {sample_file.stem}: skipped')
            payload = json.loads(pred_file.read_text(encoding='utf-8'))
            objects = payload.get('objects', [])
            crop_dir = None
            if args.crop:
                crop_dir = f'crops/{sample_file.stem}'
                if any(existing_crop_file(output_dir, obj.get('crop_path')) is None
                       for obj in objects):
                    crop_dir = write_prediction_crops(output_dir, sample_file.stem, sample_file, payload)
                    write_json_atomic(pred_file, payload)
            elif remove_prediction_crops(output_dir, sample_file.stem, payload):
                write_json_atomic(pred_file, payload)
            summary.append({
                'frame_id': sample_file.stem,
                'input_file': sample_file.as_posix(),
                'num_predictions': len(objects),
                'scene_dir': scene_dir.as_posix(),
                'predictions_json': pred_file.relative_to(output_dir).as_posix(),
                'crop_dir': crop_dir,
                'skipped': True,
            })
            continue
        points = read_pcd_as_xyzrgb(sample_file)[:, use_dim]
        result, _ = inference_detector_with_amp(
            inference_detector,
            model,
            points,
            amp=args.amp,
            amp_dtype=args.amp_dtype,
            device=args.device,
        )
        payload = prediction_payload(
            sample_file.stem, result.pred_instances_3d, class_names,
            args.score_thresh)
        crop_dir = None
        if args.crop:
            crop_dir = write_prediction_crops(output_dir, sample_file.stem, sample_file, payload)
        else:
            remove_prediction_crops(output_dir, sample_file.stem, payload)
        write_json_atomic(pred_file, payload)
        print(f'{idx + 1}/{len(files)} {sample_file.stem}: boxes={len(payload["objects"])}')
        summary.append({
            'frame_id': sample_file.stem,
            'input_file': sample_file.as_posix(),
            'num_predictions': len(payload['objects']),
            'scene_dir': scene_dir.as_posix(),
            'predictions_json': pred_file.relative_to(output_dir).as_posix(),
            'crop_dir': crop_dir,
        })
        del points, result, payload
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_json_atomic(output_dir / 'summary.json', {
        'model': args.model,
        'cfg_file': cfg_file,
        'ckpt': ckpt_path.as_posix(),
        'use_dim': use_dim,
        'class_names': list(class_names),
        'input': Path(args.input_dir).as_posix(),
        'score_thresh': args.score_thresh,
        'amp': bool(args.amp),
        'amp_dtype': args.amp_dtype if args.amp else None,
        'scenes': summary,
    })


if __name__ == '__main__':
    main()
