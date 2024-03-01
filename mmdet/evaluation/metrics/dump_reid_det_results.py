# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence

from mmengine.evaluator import DumpResults
from mmengine.evaluator.metric import _to_cpu

from mmdet.registry import METRICS


@METRICS.register_module()
class DumpReIDDetResults(DumpResults):
    """Dump model predictions to a pickle file for offline evaluation.

    Different from `DumpResults` in MMEngine, it compresses instance
    segmentation masks into RLE format.

    This is a simplified version of `DumpDetResults`, without segmentation
    code.

    Args:
        out_file_path (str): Path of the dumped file. Must end with '.pkl'
            or '.pickle'.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
    """

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Transfer tensors in predictions to CPU."""
        data_samples = _to_cpu(data_samples)
        for data_sample in data_samples:
            # remove gt
            data_sample.pop('gt_instances', None)
            data_sample.pop('ignored_instances', None)
        self.results.extend(data_samples)
