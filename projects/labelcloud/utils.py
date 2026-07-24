import hashlib
import json
import math
import os
import random
import re
from pathlib import Path

import numpy as np


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in ('1', 'true', 'yes', 'y', 'on'):
        return True
    if value in ('0', 'false', 'no', 'n', 'off'):
        return False
    raise ValueError(f'Expected boolean value, got {value!r}')


def none_if_blank(value):
    return None if value == '' else value


_UNICODE_ESCAPE_PATTERN = re.compile(
    r'\\u[0-9a-fA-F]{4}|\\U[0-9a-fA-F]{8}|\\x[0-9a-fA-F]{2}')


def decode_label_name(value):
    if not isinstance(value, str):
        return value
    if not _UNICODE_ESCAPE_PATTERN.search(value):
        return value

    def decode_match(match):
        token = match.group(0)
        try:
            if token.startswith('\\x'):
                return chr(int(token[2:], 16))
            if token.startswith('\\u'):
                return chr(int(token[2:], 16))
            return chr(int(token[2:], 16))
        except ValueError:
            return token

    return _UNICODE_ESCAPE_PATTERN.sub(decode_match, value)


def write_text_atomic(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + '.tmp')
    tmp_path.write_text(text, encoding='utf-8')
    os.replace(tmp_path, path)


def write_json_atomic(path, payload):
    write_text_atomic(path, json.dumps(payload, indent=2, ensure_ascii=False))


def write_bin_atomic(path, array):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + '.tmp')
    array.astype(np.float32, copy=False).tofile(tmp_path)
    os.replace(tmp_path, path)


def output_is_stale(output_path, source_paths):
    output_path = Path(output_path)
    if not output_path.exists():
        return True
    output_mtime = output_path.stat().st_mtime_ns
    return any(Path(source).stat().st_mtime_ns > output_mtime
               for source in source_paths)


def load_classes(class_file):
    data = json.loads(Path(class_file).read_text())
    classes = [decode_label_name(item['name']) for item in data['classes']]
    if not classes:
        raise ValueError(f'No classes found in {class_file}')
    if len(set(classes)) != len(classes):
        raise ValueError(f'Duplicate class names found in {class_file}: {classes}')
    return classes


def normalize_heading_degrees(angle_deg):
    angle = math.radians(float(angle_deg))
    return (angle + math.pi) % (2 * math.pi) - math.pi


def require_finite_number(value, desc, label_path):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f'Expected finite {desc} in {label_path}, got {value!r}')
    return number


def read_label_file(label_path, class_names):
    data = json.loads(Path(label_path).read_text())
    rows = []
    extents = []
    dims_by_class = {name: [] for name in class_names}
    bottoms_by_class = {name: [] for name in class_names}

    for obj in data.get('objects', []):
        name = decode_label_name(obj['name'])
        if name not in class_names:
            raise ValueError(f'Unknown class {name!r} in {label_path}')

        c = obj['centroid']
        d = obj['dimensions']
        r = obj['rotations']
        rot_x = require_finite_number(r.get('x', 0.0), 'rotations.x', label_path)
        rot_y = require_finite_number(r.get('y', 0.0), 'rotations.y', label_path)
        if abs(rot_x) > 1e-4 or abs(rot_y) > 1e-4:
            raise ValueError(
                f'Only z-axis rotation is supported; got non-zero x/y rotation in {label_path}')

        x = require_finite_number(c['x'], 'centroid.x', label_path)
        y = require_finite_number(c['y'], 'centroid.y', label_path)
        z = require_finite_number(c['z'], 'centroid.z', label_path)
        dx = require_finite_number(d['length'], 'dimensions.length', label_path)
        dy = require_finite_number(d['width'], 'dimensions.width', label_path)
        dz = require_finite_number(d['height'], 'dimensions.height', label_path)
        if dx <= 0.0 or dy <= 0.0 or dz <= 0.0:
            raise ValueError(
                f'Box dimensions must be positive in {label_path}; '
                f'got length={dx}, width={dy}, height={dz}')
        heading = normalize_heading_degrees(
            require_finite_number(r['z'], 'rotations.z', label_path))
        rows.append((x, y, z, dx, dy, dz, heading, name))
        dims_by_class[name].append((dx, dy, dz))
        bottoms_by_class[name].append(z - dz / 2.0)

        radius = math.sqrt(dx * dx + dy * dy) / 2.0
        extents.append((x - radius, y - radius, z - dz / 2.0))
        extents.append((x + radius, y + radius, z + dz / 2.0))

    return data, rows, extents, dims_by_class, bottoms_by_class


def read_pcd_as_xyzrgb(path):
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError('open3d is required to read PCD files.') from exc

    pcd = o3d.io.read_point_cloud(
        str(path), remove_nan_points=True, remove_infinite_points=True)
    xyz = np.asarray(pcd.points, dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f'Failed to read XYZ points from {path}')
    colors = np.asarray(pcd.colors, dtype=np.float32)
    if colors.shape == xyz.shape and colors.size > 0:
        rgb = np.clip(colors, 0.0, 1.0).astype(np.float32)
    else:
        rgb = np.zeros_like(xyz, dtype=np.float32)
    return np.concatenate([xyz, rgb], axis=1).astype(np.float32, copy=False)


def split_scenes(scene_ids, train_ratio, val_ratio, seed):
    total = train_ratio + val_ratio
    if total <= 0:
        raise ValueError('At least one split ratio must be positive.')
    train_ratio, val_ratio = [x / total for x in (train_ratio, val_ratio)]

    scene_ids = list(scene_ids)
    random.Random(seed).shuffle(scene_ids)
    n = len(scene_ids)
    n_val = int(round(n * val_ratio))
    n_train = n - n_val

    if train_ratio > 0 and n_train == 0 and n > 0:
        n_train, n_val = 1, max(0, n_val - 1)
    if val_ratio > 0 and n_val == 0 and n > 1:
        n_train, n_val = n - 1, 1
    return {'train': scene_ids[:n_train], 'val': scene_ids[n_train:n_train + n_val]}


def aligned_point_cloud_range(extents, margin, voxel_xy):
    if not extents:
        return [-80.0, -80.0, -5.0, 80.0, 80.0, 7.0], [voxel_xy, voxel_xy, 0.3]

    arr = np.asarray(extents, dtype=np.float32)
    mins = arr.min(axis=0) - margin
    maxs = arr.max(axis=0) + margin

    def align_axis(lo, hi, voxel):
        lo = math.floor(float(lo) / voxel) * voxel
        hi = math.ceil(float(hi) / voxel) * voxel
        cells = int(round((hi - lo) / voxel))
        rem = cells % 16
        if rem:
            hi += (16 - rem) * voxel
        return lo, hi

    x_min, x_max = align_axis(mins[0], maxs[0], voxel_xy)
    y_min, y_max = align_axis(mins[1], maxs[1], voxel_xy)
    z_min = math.floor(float(mins[2]) * 10.0) / 10.0
    z_max = math.ceil(float(maxs[2]) * 10.0) / 10.0
    z_span = max(z_max - z_min, 4.0)
    voxel_z = round(z_span / 40.0, 4)
    z_max = z_min + voxel_z * 40
    return [x_min, y_min, z_min, x_max, y_max, z_max], [voxel_xy, voxel_xy, voxel_z]


def class_stats(class_names, dims_by_class, bottoms_by_class):
    stats = {}
    for name in class_names:
        dims = np.asarray(dims_by_class[name], dtype=np.float32)
        bottoms = np.asarray(bottoms_by_class[name], dtype=np.float32)
        if dims.size == 0:
            stats[name] = dict(anchor_size=[1.0, 1.0, 1.0], anchor_bottom=0.0)
            continue
        # Anchor3DHead with assign_per_class expects one anchor size per class.
        stats[name] = dict(
            anchor_size=np.maximum(np.median(dims, axis=0), 0.05).tolist(),
            anchor_bottom=float(np.median(bottoms)))
    return stats


def points_in_rotated_box(points_xyz, box):
    center = box[:3]
    dims = box[3:6]
    heading = box[6]
    shifted = points_xyz - center
    cos_h = np.cos(heading)
    sin_h = np.sin(heading)
    local_x = cos_h * shifted[:, 0] + sin_h * shifted[:, 1]
    local_y = -sin_h * shifted[:, 0] + cos_h * shifted[:, 1]
    local_z = shifted[:, 2]
    eps = 1e-6
    return (
        (np.abs(local_x) <= dims[0] / 2.0 + eps) &
        (np.abs(local_y) <= dims[1] / 2.0 + eps) &
        (np.abs(local_z) <= dims[2] / 2.0 + eps))


def rotated_box_corners_xy(box):
    x, y, _, dx, dy, _, heading = [float(v) for v in box[:7]]
    local = np.array([
        [dx / 2.0, dy / 2.0],
        [-dx / 2.0, dy / 2.0],
        [-dx / 2.0, -dy / 2.0],
        [dx / 2.0, -dy / 2.0],
    ],
                     dtype=np.float64)
    cos_h = math.cos(heading)
    sin_h = math.sin(heading)
    rot = np.array([[cos_h, -sin_h], [sin_h, cos_h]], dtype=np.float64)
    return local @ rot.T + np.array([x, y], dtype=np.float64)


def polygon_area(points):
    if len(points) < 3:
        return 0.0
    pts = np.asarray(points, dtype=np.float64)
    x = pts[:, 0]
    y = pts[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) -
                     np.dot(y, np.roll(x, -1))) * 0.5)


def _line_intersection(p1, p2, q1, q2):
    p = np.asarray(p1, dtype=np.float64)
    r = np.asarray(p2, dtype=np.float64) - p
    q = np.asarray(q1, dtype=np.float64)
    s = np.asarray(q2, dtype=np.float64) - q
    denom = r[0] * s[1] - r[1] * s[0]
    if abs(denom) < 1e-12:
        return p2
    qmp = q - p
    t = (qmp[0] * s[1] - qmp[1] * s[0]) / denom
    return p + t * r


def _clip_polygon(subject, clipper):
    output = [np.asarray(point, dtype=np.float64) for point in subject]
    clipper = [np.asarray(point, dtype=np.float64) for point in clipper]
    if len(output) < 3 or len(clipper) < 3:
        return []

    def inside(point, edge_start, edge_end):
        edge = edge_end - edge_start
        rel = point - edge_start
        return edge[0] * rel[1] - edge[1] * rel[0] >= -1e-9

    for idx, edge_start in enumerate(clipper):
        edge_end = clipper[(idx + 1) % len(clipper)]
        input_points = output
        output = []
        if not input_points:
            break
        prev = input_points[-1]
        prev_inside = inside(prev, edge_start, edge_end)
        for current in input_points:
            current_inside = inside(current, edge_start, edge_end)
            if current_inside:
                if not prev_inside:
                    output.append(
                        _line_intersection(prev, current, edge_start,
                                           edge_end))
                output.append(current)
            elif prev_inside:
                output.append(
                    _line_intersection(prev, current, edge_start, edge_end))
            prev = current
            prev_inside = current_inside
    return output


def rotated_iou_2d(box_a, box_b):
    corners_a = rotated_box_corners_xy(box_a)
    corners_b = rotated_box_corners_xy(box_b)
    area_a = polygon_area(corners_a)
    area_b = polygon_area(corners_b)
    if area_a <= 0.0 or area_b <= 0.0:
        return 0.0
    inter_poly = _clip_polygon(corners_a, corners_b)
    inter_area = polygon_area(inter_poly)
    union = area_a + area_b - inter_area
    return 0.0 if union <= 0.0 else float(inter_area / union)


def lidar_box_iou_3d(boxes_a, boxes_b):
    boxes_a = _as_lidar_boxes(boxes_a)
    boxes_b = _as_lidar_boxes(boxes_b)
    ious = np.zeros((boxes_a.shape[0], boxes_b.shape[0]), dtype=np.float64)
    if boxes_a.size == 0 or boxes_b.size == 0:
        return ious

    volumes_a = boxes_a[:, 3] * boxes_a[:, 4] * boxes_a[:, 5]
    volumes_b = boxes_b[:, 3] * boxes_b[:, 4] * boxes_b[:, 5]
    for i, box_a in enumerate(boxes_a):
        if volumes_a[i] <= 0.0:
            continue
        z_min_a = box_a[2]
        z_max_a = box_a[2] + box_a[5]
        area_a = box_a[3] * box_a[4]
        for j, box_b in enumerate(boxes_b):
            if volumes_b[j] <= 0.0:
                continue
            z_overlap = min(z_max_a, box_b[2] + box_b[5]) - max(
                z_min_a, box_b[2])
            if z_overlap <= 0.0:
                continue
            bev_iou = rotated_iou_2d(box_a, box_b)
            if bev_iou <= 0.0:
                continue
            area_b = box_b[3] * box_b[4]
            inter_area = bev_iou * (area_a + area_b) / (1.0 + bev_iou)
            inter_volume = inter_area * z_overlap
            union = volumes_a[i] + volumes_b[j] - inter_volume
            if union > 0.0:
                ious[i, j] = inter_volume / union
    return ious


def _as_lidar_boxes(value):
    boxes = np.asarray(value, dtype=np.float64)
    if boxes.size == 0:
        return np.zeros((0, 7), dtype=np.float64)
    boxes = boxes.reshape(1, -1) if boxes.ndim == 1 else boxes.reshape(
        -1, boxes.shape[-1])
    if boxes.shape[1] < 7:
        raise ValueError(f'Expected boxes with at least 7 dims, got {boxes.shape}')
    return boxes[:, :7]


def average_precision(tp_flags, fp_flags, scores, num_gts):
    if num_gts <= 0:
        return None
    if len(scores) == 0:
        return 0.0
    order = np.argsort(-np.asarray(scores, dtype=np.float64))
    tp = np.cumsum(np.asarray(tp_flags, dtype=np.float64)[order])
    fp = np.cumsum(np.asarray(fp_flags, dtype=np.float64)[order])
    recalls = tp / max(float(num_gts), 1.0)
    precisions = tp / np.maximum(tp + fp, 1e-12)
    recalls = np.concatenate([[0.0], recalls, [1.0]])
    precisions = np.concatenate([[0.0], precisions, [0.0]])
    for idx in range(precisions.size - 1, 0, -1):
        precisions[idx - 1] = max(precisions[idx - 1], precisions[idx])
    change = np.where(recalls[1:] != recalls[:-1])[0]
    return float(np.sum((recalls[change + 1] - recalls[change]) *
                        precisions[change + 1]))


def labelcloud_detection_metrics(results,
                                 iou_thresholds=(0.25, 0.5),
                                 classes=None,
                                 score_thr=0.0):
    total_scenes = len(results)
    total_gts = sum(len(item['gt_labels']) for item in results)
    max_label = -1
    score_thr = float(score_thr)
    total_preds = 0
    for item in results:
        pred_scores = np.asarray(item['pred_scores'], dtype=np.float64).reshape(-1)
        pred_labels = np.asarray(item['pred_labels'], dtype=np.int64).reshape(-1)
        keep = pred_scores >= score_thr
        total_preds += int(np.count_nonzero(keep))
        if np.any(keep):
            max_label = max(max_label, int(np.max(pred_labels[keep])))
        if len(item['gt_labels']):
            max_label = max(max_label, int(np.max(item['gt_labels'])))
    num_classes = len(classes) if classes is not None else max_label + 1

    metrics = {
        'labelcloud/num_scenes': float(total_scenes),
        'labelcloud/num_predictions': float(total_preds),
        'labelcloud/num_ground_truths': float(total_gts),
    }
    if total_scenes == 0 or num_classes <= 0:
        for thr in iou_thresholds:
            suffix = f'@{thr:g}'
            metrics[f'labelcloud/precision{suffix}'] = 0.0
            metrics[f'labelcloud/recall{suffix}'] = 0.0
            metrics[f'labelcloud/f1{suffix}'] = 0.0
            metrics[f'labelcloud/mAP{suffix}'] = 0.0
        return metrics

    for thr in iou_thresholds:
        class_records = [
            dict(scores=[], tp=[], fp=[], num_gts=0) for _ in range(num_classes)
        ]
        for item in results:
            pred_boxes = _as_lidar_boxes(item['pred_boxes'])
            pred_scores = np.asarray(
                item['pred_scores'], dtype=np.float64).reshape(-1)
            pred_labels = np.asarray(
                item['pred_labels'], dtype=np.int64).reshape(-1)
            gt_boxes = _as_lidar_boxes(item['gt_boxes'])
            gt_labels = np.asarray(item['gt_labels'], dtype=np.int64).reshape(-1)
            keep = pred_scores >= score_thr
            pred_boxes = pred_boxes[keep]
            pred_scores = pred_scores[keep]
            pred_labels = pred_labels[keep]

            for class_idx in range(num_classes):
                pred_inds = np.where(pred_labels == class_idx)[0]
                gt_inds = np.where(gt_labels == class_idx)[0]
                record = class_records[class_idx]
                record['num_gts'] += int(len(gt_inds))
                if len(pred_inds) == 0:
                    continue
                order = pred_inds[np.argsort(-pred_scores[pred_inds])]
                matched = np.zeros(len(gt_inds), dtype=bool)
                ious = lidar_box_iou_3d(pred_boxes[order], gt_boxes[gt_inds])
                for local_pred_idx, pred_idx in enumerate(order):
                    record['scores'].append(float(pred_scores[pred_idx]))
                    if len(gt_inds) == 0:
                        record['tp'].append(0)
                        record['fp'].append(1)
                        continue
                    candidate_ious = ious[local_pred_idx].copy()
                    candidate_ious[matched] = -1.0
                    best_gt = int(np.argmax(candidate_ious))
                    if candidate_ious[best_gt] >= float(thr):
                        matched[best_gt] = True
                        record['tp'].append(1)
                        record['fp'].append(0)
                    else:
                        record['tp'].append(0)
                        record['fp'].append(1)

        total_tp = sum(sum(record['tp']) for record in class_records)
        total_fp = sum(sum(record['fp']) for record in class_records)
        total_gt_for_thr = sum(record['num_gts'] for record in class_records)
        precision = total_tp / max(total_tp + total_fp, 1)
        recall = total_tp / max(total_gt_for_thr, 1)
        f1 = 0.0 if precision + recall == 0 else (
            2.0 * precision * recall / (precision + recall))
        aps = [
            average_precision(record['tp'], record['fp'], record['scores'],
                              record['num_gts']) for record in class_records
        ]
        valid_aps = [ap for ap in aps if ap is not None]
        suffix = f'@{thr:g}'
        metrics[f'labelcloud/precision{suffix}'] = float(precision)
        metrics[f'labelcloud/recall{suffix}'] = float(recall)
        metrics[f'labelcloud/f1{suffix}'] = float(f1)
        metrics[f'labelcloud/mAP{suffix}'] = (
            float(np.mean(valid_aps)) if valid_aps else 0.0)
        if classes is not None:
            for class_idx, class_name in enumerate(classes):
                ap = aps[class_idx]
                if ap is not None:
                    safe_class = safe_name(class_name)
                    metrics[f'labelcloud/{safe_class}_AP{suffix}'] = float(ap)
    return metrics


def safe_name(value, max_len=80):
    raw = str(value)
    chars = []
    for char in raw:
        chars.append(char if char.isalnum() or char in '_.-' else '_')
    slug = re.sub(r'_+', '_', ''.join(chars)).strip('._-')
    if slug == raw and 0 < len(slug.encode('utf-8')) <= max_len:
        return slug

    digest = hashlib.sha1(raw.encode('utf-8')).hexdigest()[:10]
    if not slug:
        return f'label_{digest}'

    suffix = f'_{digest}'
    budget = max(1, max_len - len(suffix.encode('utf-8')))
    truncated = []
    used = 0
    for char in slug:
        size = len(char.encode('utf-8'))
        if used + size > budget:
            break
        truncated.append(char)
        used += size
    slug = ''.join(truncated).rstrip('._-')
    return f'{slug or "label"}{suffix}'
