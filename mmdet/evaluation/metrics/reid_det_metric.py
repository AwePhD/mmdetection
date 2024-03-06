import re
from pathlib import Path
from typing import NamedTuple

import pandas as pd
import torch
from mmcv.ops import bbox_overlaps as compute_ious
from mmengine.evaluator import BaseMetric
from torch import Tensor
from torch.nn import functional as F

from mmdet.evaluation import compute_average_precision
from mmdet.registry import METRICS

# Default metric options arguments for ReIDDetMetric
DEFAULT_GALLERY_SIZE = 100
DEFAULT_GALLERY_THRESHOLD = .25

BBOXE_COLUMNS = ["bbox_x", "bbox_y", "bbox_w", "bbox_h"]

DTYPES_DATASET = ({
    "person_id": "UInt16",
    "frame_id": "UInt16",
    "split": "category",
    "bbox_x": "Int32",
    "bbox_y": "Int32",
    "bbox_w": "Int32",
    "bbox_h": "Int32"
})


class FramePrediction(NamedTuple):
    # confidence score of the detector for a single frame.
    # (num_pred)
    scores: Tensor
    # (x1, y1, x2, y2) bboxes for a single frame.
    # (num_pred, 4)
    bboxes: Tensor
    # ReID features, length depends of the ReID model.
    # (num_pred, n_dim_reid)
    reid_features: Tensor


def get_bbox_from_annotations(annotations: pd.Series) -> Tensor:
    if annotations.bbox_x is pd.NA:
        return torch.zeros(4, dtype=torch.int32)

    return torch.tensor([
        annotations.bbox_x, annotations.bbox_y, annotations.bbox_x +
        annotations.bbox_w, annotations.bbox_y + annotations.bbox_h
    ],
                        dtype=torch.float32)


# TODO: Add top-k metric
@METRICS.register_module()
class ReIDDetMetric(BaseMetric):
    allowed_metrics = ['mAP']

    def check_annotations_integrity(self, annotations: pd.DataFrame) -> None:
        assert len(annotations.person_id.unique()) == self.n_samples

        for _, sample in annotations.groupby("person_id"):
            assert len(sample) == self.gallery_size + 1
            assert len(sample.query("split == 'query'")) == 1

    def __init__(
            self,
            ann_file: str,
            split_column: str,
            metric: str = "mAP",
            metric_options: dict = {
                "n_samples": 641,  # Modified SYSU default value
                "gallery_size": DEFAULT_GALLERY_SIZE,
                "gallery_threshold": DEFAULT_GALLERY_THRESHOLD,
            },
            collect_device: str = "cpu",
            prefix: str | None = "det_reid") -> None:
        super().__init__(collect_device, prefix)

        if metric not in self.allowed_metrics:
            raise KeyError(f'metric {metric} is not supported.')

        self.metrics: list[str] = [metric]
        self.n_samples: int = metric_options["n_samples"]
        self.gallery_size: int = metric_options["gallery_size"]
        self.gallery_threshold: float = metric_options["gallery_threshold"]

        annotations_file = Path(ann_file)
        assert annotations_file.exists(), f"{annotations_file} not found"
        annotations = pd.read_csv(annotations_file)
        annotations.rename(columns={split_column: "split"}, inplace=True)
        self.check_annotations_integrity(annotations)
        self.annotations: pd.DataFrame = annotations.astype(DTYPES_DATASET)

        self.frame_id_to_prediction: dict[int, FramePrediction] = {}

        self.frame_id_expression = re.compile(r"\d+")

        # Will stay empty, see frame_id_to_prediction
        self.results: list = self.annotations.frame_id.unique().tolist()

    # =========================================================================
    # ========================= PREPROCESS METHODS ============================
    # =========================================================================

    def get_frame_id_from_img_path(self, img_path: str) -> int:
        img_name = Path(img_path).stem
        return int(self.frame_id_expression.findall(img_name)[0])

    def _process_single_frame(self, pred_instances: dict,
                              img_path: str) -> None:
        prediction = FramePrediction(
            scores=pred_instances["scores"].cpu(),
            bboxes=pred_instances["bboxes"].cpu(),
            reid_features=pred_instances["reid_features"].cpu(),
        )

        frame_id = self.get_frame_id_from_img_path(img_path)
        self.frame_id_to_prediction[frame_id] = prediction

    def process(self, data_batch: dict, data_samples: list[dict]) -> None:
        # Fill self.result with batch annotations and predictions.
        for data_sample in data_samples:
            pred_instances = data_sample["pred_instances"]
            img_path = data_sample["img_path"]

            self._process_single_frame(pred_instances, img_path)

    def _get_query_features(self, query_annotation: pd.DataFrame) -> Tensor:
        frame_id = query_annotation.frame_id.iloc[0]
        prediction = self.frame_id_to_prediction[frame_id]
        gt_bbox = get_bbox_from_annotations(query_annotation.iloc[0])

        ious = compute_ious(
            gt_bbox.unsqueeze(0),  # (1, 4)
            prediction.bboxes,  # (num_queries, 4)
        ).squeeze(0)  # (1, num_queries) -> (num_queries)
        i_best_bbox = ious.argmax()

        query_features = prediction.reid_features[i_best_bbox]
        query_features_normalized = F.normalize(query_features, dim=0)

        return query_features_normalized

    def _compute_AP_sample(self, sample: pd.DataFrame) -> float:
        query_features = self._get_query_features(
            sample.query("split == 'query'"))

        all_similarities: list[Tensor] = []
        all_are_queries: list[Tensor] = []
        count_positives = 0
        count_true_positives = 0

        for annotations in sample.query("split == 'gallery'").itertuples():
            gt_bbox = get_bbox_from_annotations(annotations)  # type: ignore
            prediction = self.frame_id_to_prediction[
                annotations.frame_id]  # type: ignore

            is_distractor = (gt_bbox == 0).all()
            count_positives += not is_distractor

            are_confident_enough = prediction.scores > self.gallery_threshold
            if are_confident_enough.sum() == 0:
                continue

            # n_ok = number of bboxes / features kept after thresholding
            features = prediction.reid_features[are_confident_enough]
            bboxes = prediction.bboxes[are_confident_enough]
            similarities = torch.einsum("d,nd->n", query_features, features)

            are_queries = torch.zeros(len(similarities), dtype=torch.bool)

            # End of the gallery frame processing
            if is_distractor:
                all_are_queries.extend(are_queries.unbind())
                all_similarities.extend(similarities.unbind())
                continue

            gt_width: int = annotations.bbox_w  # type: ignore
            gt_height: int = annotations.bbox_h  # type: ignore
            iou_threshold = min(0.5, (gt_width * gt_height) /
                                ((gt_width + 10) * (gt_height + 10)))

            i_sorted_by_similarites = similarities.argsort(descending=True)
            ious = compute_ious(
                gt_bbox.unsqueeze(0),  # (1, 4)
                bboxes[i_sorted_by_similarites]  # (n_ok, 4)
            ).squeeze(0)  # (1, n_ok) -> (n_ok)
            for i in i_sorted_by_similarites:
                if ious[i] >= iou_threshold:
                    are_queries[i] = True
                    count_true_positives += 1
                    break

            all_are_queries.extend(are_queries.unbind())
            all_similarities.extend(similarities.unbind())

        assert count_true_positives <= count_positives
        if count_true_positives == 0:
            return 0

        all_similarities_np = torch.stack(all_similarities).numpy()
        all_are_queries_np = torch.stack(all_are_queries).numpy()

        average_precision = compute_average_precision(all_are_queries_np,
                                                      all_similarities_np)
        recall_rate = count_true_positives / count_positives

        return average_precision * recall_rate

    def compute_metrics(self, results: list) -> dict:
        n_processed_predicion = len(self.frame_id_to_prediction)
        assert n_processed_predicion == len(self.annotations.frame_id.unique())

        annotations_by_person_id = self.annotations.groupby("person_id")
        sum_average_precision = sum(
            self._compute_AP_sample(sample)
            for _, sample in annotations_by_person_id)

        n_samples = len(annotations_by_person_id.groups)
        return {"mAP": sum_average_precision / n_samples}
