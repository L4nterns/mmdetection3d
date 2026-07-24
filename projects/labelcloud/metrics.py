from typing import Dict, Optional, Sequence

import numpy as np
from mmengine.evaluator import BaseMetric

from mmdet3d.registry import METRICS
from projects.labelcloud.utils import labelcloud_detection_metrics


@METRICS.register_module()
class LabelCloudMetric(BaseMetric):
    """Class-aware 3D IoU metric for labelCloud experiments.

    This intentionally avoids pretending labelCloud data has KITTI-specific
    fields. It evaluates converted LiDAR boxes directly with class-aware 3D IoU.
    """

    def __init__(self,
                 classes: Optional[Sequence[str]] = None,
                 iou_thresholds: Sequence[float] = (0.25, 0.5),
                 score_thr: float = 0.0,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(prefix=prefix, collect_device=collect_device)
        self.classes = tuple(classes) if classes is not None else None
        self.iou_thresholds = tuple(float(x) for x in iou_thresholds)
        self.score_thr = float(score_thr)

    @staticmethod
    def _get(container, key, default=None):
        if container is None:
            return default
        if hasattr(container, 'get'):
            return container.get(key, default)
        return getattr(container, key, default)

    @staticmethod
    def _to_numpy(value, dtype=None):
        if value is None:
            return np.zeros((0, ), dtype=dtype or np.float32)
        if hasattr(value, 'tensor'):
            value = value.tensor
        if hasattr(value, 'detach'):
            value = value.detach()
        if hasattr(value, 'cpu'):
            value = value.cpu()
        if hasattr(value, 'numpy'):
            value = value.numpy()
        return np.asarray(value, dtype=dtype)

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        for data_sample in data_samples:
            pred = self._get(data_sample, 'pred_instances_3d', {})
            eval_ann_info = self._get(data_sample, 'eval_ann_info', {})
            pred_boxes = self._to_numpy(
                self._get(pred, 'bboxes_3d'), dtype=np.float64)
            pred_scores = self._to_numpy(
                self._get(pred, 'scores_3d'), dtype=np.float64).reshape(-1)
            pred_labels = self._to_numpy(
                self._get(pred, 'labels_3d'), dtype=np.int64).reshape(-1)
            gt_boxes = self._to_numpy(
                self._get(eval_ann_info, 'gt_bboxes_3d'),
                dtype=np.float64)
            gt_labels = self._to_numpy(
                self._get(eval_ann_info, 'gt_labels_3d'),
                dtype=np.int64).reshape(-1)
            self.results.append(
                dict(
                    pred_boxes=pred_boxes,
                    pred_scores=pred_scores,
                    pred_labels=pred_labels,
                    gt_boxes=gt_boxes,
                    gt_labels=gt_labels))

    def compute_metrics(self, results: list) -> Dict[str, float]:
        return labelcloud_detection_metrics(
            results,
            iou_thresholds=self.iou_thresholds,
            classes=self.classes,
            score_thr=self.score_thr)
