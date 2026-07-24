from typing import Callable, List, Optional, Union

import numpy as np

from mmdet3d.datasets import Det3DDataset
from mmdet3d.registry import DATASETS
from mmdet3d.structures import DepthInstance3DBoxes, LiDARInstance3DBoxes


@DATASETS.register_module()
class LabelCloudDataset(Det3DDataset):
    """LiDAR 3D detection dataset for converted labelCloud projects.

    The converter writes labelCloud boxes as geometric centers. MMDet3D boxes
    use bottom centers internally, so this dataset wraps boxes with
    ``origin=(0.5, 0.5, 0.5)`` during annotation parsing.
    """

    METAINFO = {'classes': (), 'palette': []}

    def __init__(self,
                 data_root: Optional[str] = None,
                 ann_file: str = '',
                 metainfo: Optional[dict] = None,
                 data_prefix: Optional[dict] = None,
                 pipeline: Optional[List[Union[dict, Callable]]] = None,
                 modality: Optional[dict] = None,
                 box_type_3d: str = 'LiDAR',
                 filter_empty_gt: bool = True,
                 axis_align_boxes: bool = False,
                 test_mode: bool = False,
                 load_eval_anns: bool = True,
                 backend_args: Optional[dict] = None,
                 **kwargs) -> None:
        if metainfo is None or 'classes' not in metainfo:
            raise ValueError('LabelCloudDataset requires metainfo["classes"].')

        classes = tuple(metainfo['classes'])
        self.METAINFO = {
            'classes': classes,
            'palette': self._palette(len(classes)),
        }
        if data_prefix is None:
            data_prefix = dict(pts='')
        if pipeline is None:
            pipeline = []
        if modality is None:
            modality = dict(use_lidar=True, use_camera=False)
        self.axis_align_boxes = axis_align_boxes

        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            metainfo=metainfo,
            data_prefix=data_prefix,
            pipeline=pipeline,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            load_eval_anns=load_eval_anns,
            backend_args=backend_args,
            **kwargs)

    @staticmethod
    def _palette(num_classes: int) -> List[tuple]:
        base = [
            (106, 0, 228), (119, 11, 32), (0, 180, 255), (0, 220, 120),
            (255, 145, 0), (220, 20, 60), (255, 215, 0), (138, 43, 226),
        ]
        if num_classes <= len(base):
            return base[:num_classes]
        return [base[i % len(base)] for i in range(num_classes)]

    def parse_ann_info(self, info: dict) -> dict:
        ann_info = super().parse_ann_info(info)
        if ann_info is None:
            ann_info = dict(
                gt_bboxes_3d=np.zeros((0, 7), dtype=np.float32),
                gt_labels_3d=np.zeros(0, dtype=np.int64),
                instances=[])

        ann_info = self._remove_dontcare(ann_info)
        if self.axis_align_boxes:
            ann_info['gt_bboxes_3d'] = ann_info['gt_bboxes_3d'][..., :6]
        box_cls = {
            'LiDARInstance3DBoxes': LiDARInstance3DBoxes,
            'DepthInstance3DBoxes': DepthInstance3DBoxes,
        }.get(self.box_type_3d.__name__)
        if box_cls is None:
            raise ValueError(
                f'Unsupported labelCloud box type: {self.box_type_3d.__name__}')
        ann_info['gt_bboxes_3d'] = box_cls(
            ann_info['gt_bboxes_3d'],
            box_dim=ann_info['gt_bboxes_3d'].shape[-1],
            origin=(0.5, 0.5, 0.5))
        return ann_info
