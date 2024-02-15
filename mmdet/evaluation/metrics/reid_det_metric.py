from dataclasses import dataclass, field
from typing import NamedTuple

import torch
from mmcv.ops import bbox_overlaps as compute_ious
from mmengine.evaluator import BaseMetric
from torch import Tensor
from torch.nn import functional as F

from mmdet.evaluation import compute_average_precision
from mmdet.registry import METRICS
from mmdet.structures.reid_det_data_sample import EvalType

# Default metric options arguments for ReIDDetMetric
DEFAULT_GALLERY_SIZE = 100
DEFAULT_GALLERY_THRESHOLD = .25


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


def get_thresholded_view(frame_prediction: FramePrediction,
                         gallery_threshold: float) -> FramePrediction:
    prediction_is_okay = frame_prediction.scores > gallery_threshold
    return FramePrediction(
        bboxes=frame_prediction.bboxes[prediction_is_okay],
        scores=frame_prediction.scores[prediction_is_okay],
        reid_features=frame_prediction.reid_features[prediction_is_okay],
    )


@dataclass
class EvalData:
    # (x1, y1, x2, y2) annotation, might have NO_DETECTION value, see
    # mmdet.datasets.transform.loading.py:NO_DETECTION.
    gt_bbox: Tensor
    # prediction of the model for the frame.
    prediction: FramePrediction = field(init=False)
    # flag for no detection annotation. It is not mandatory, just an ease.
    is_distractor: bool = field(init=False)

    def __post_init__(self):
        # See mmdet.datasets.transforms.loading:NO_DETECTION
        self.is_distractor = (self.gt_bbox == 0).all()


class Sample:

    def __init__(self, person_id: int, gallery_size: int,
                 gallery_threshold: float):
        self.person_id: int = person_id
        self.gallery_size: int = gallery_size
        self.gallery_threshold: float = gallery_threshold

        self.query: EvalData = None
        self.gallery: list[EvalData] = []

    def add_data(self, data: EvalData, eval_type: EvalType) -> None:
        """
        Add the data based on its eval_type (query or gallery). We assert that
        the query does not exist already or that the gallery size does not
        exceeds its max size.

        Args:
            data (EvalData): The data to insert in self.results
            eval_type (EvalType): query (= True), gallery (= False)
        """
        if eval_type:  # QUERY CASE
            assert not self.query, (f"Attempt to define the query twice for"
                                    f" {self.person_id = }")
            self.query = data
            return

        # GALLERY CASE
        assert len(self.gallery) < self.gallery_size, (
            "Attempt to define more than the gallery "
            f"size ({self.gallery_size}) for"
            f" {self.person_id = }.")
        self.gallery.append(data)

    def assert_complete(self) -> None:
        assert bool(self.query)
        assert sum([bool(gallery_frame)
                    for gallery_frame in self.gallery]) == self.gallery_size


# TODO: Add top-k metric
@METRICS.register_module()
class ReIDDetMetric(BaseMetric):
    allowed_metrics = ['mAP']

    def __init__(
            self,
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

        # Output of model predict.
        self.results: list[Sample] = []

    # =========================================================================
    # ========================= PREPROCESS METHODS ============================
    # =========================================================================

    def _feed_results(self, eval_data: EvalData, reid_label: int,
                      eval_type: EvalType) -> None:
        """
        Feed self.results with the evaluation data provided by the annotations.
        We need its reid_label to locate its sample in self.results and its
        EvalType (query = True, gallery = False) to feed it accordingly.

        Args:
            eval_data (EvalData): The evaluation data to insert in self.results
            reid_label (int): Person ID
            eval_type (EvalType): query (= True) and gallery (= False)
        """
        for sample in self.results:
            if sample.person_id == reid_label:
                sample.add_data(eval_data, eval_type)
                return

        # If there is no sample created for the person_id of the annotation,
        # make a new one.
        sample = Sample(
            person_id=reid_label,
            gallery_size=self.gallery_size,
            gallery_threshold=self.gallery_threshold)
        self.results.append(sample)

        sample.add_data(eval_data, eval_type)

    def _process_single_frame(self, gt_instances: dict,
                              pred_instances: dict) -> None:
        """
        We feed the Sample elements inside self.results with the prediction
        and the annotation.
        1. We construct our FramePrediction
        2. We iterate over the annotations instance (gt_bbox, reid_label =
        person_id, reid_label)
            1. We create a new EvalData from the annotation
            2. We add it to self.results
            3. We add the FramePrediction to our EvalData, and threshold it if
            necessary.
        """
        frame_prediction = FramePrediction(
            bboxes=pred_instances["bboxes"].cpu(),
            scores=pred_instances["scores"].cpu(),
            reid_features=pred_instances["reid_features"].cpu(),
        )

        for gt_bbox, reid_label, eval_type in zip(gt_instances["bboxes"],
                                                  gt_instances["reid_labels"],
                                                  gt_instances["eval_types"]):
            gt_bbox = gt_bbox.cpu()
            reid_label = reid_label.cpu()
            eval_type = eval_type.cpu()

            eval_data = EvalData(gt_bbox.cpu())
            self._feed_results(eval_data, reid_label, eval_type)

            # If the annotation is query annotation -> its prediction is the
            # original prediction.
            # Else, create a new FramePrediction with its values as a view of
            # the original one with threshold applied.
            if eval_type:
                eval_data.prediction = frame_prediction
            else:
                eval_data.prediction = get_thresholded_view(
                    frame_prediction, self.gallery_threshold)

    def process(self, data_batch: dict, data_samples: list[dict]) -> None:
        # Fill self.result with batch annotations and predictions.
        for data_sample in data_samples:
            self._process_single_frame(data_sample["gt_instances"],
                                       data_sample["pred_instances"])

    # =========================================================================
    # ========================= EVALUATION METHODS ============================
    # =========================================================================

    @staticmethod
    def _get_query_features(query: EvalData) -> Tensor:
        ious = compute_ious(  # (num_queries)
            query.gt_bbox.unsqueeze(0),  # (1, 4)
            query.prediction.bboxes,  # (num_queries, 4)
        )
        i_best_bbox = ious.argmax()

        query_features = query.prediction.reid_features[i_best_bbox]
        query_features_normalized = F.normalize(query_features, dim=0)

        return query_features_normalized

    def _compute_metrics_single_sample(self, result: Sample) -> float:
        result.assert_complete()

        query_features = self._get_query_features(result.query)

        similarities: list[Tensor] = []
        is_query: list[Tensor] = []
        count_positives = 0
        count_true_positives = 0

        # For each instance in the gallery instances (make the gallery)
        for gallery_frame in result.gallery:
            gt_bbox = gallery_frame.gt_bbox
            frame_bboxes = gallery_frame.prediction.bboxes
            frame_reid_features = gallery_frame.prediction.reid_features

            # count one positive if the gallery frame has a gt bbox
            # (= not a distractor)
            count_positives += not gallery_frame.is_distractor

            if len(frame_bboxes) == 0:
                continue
            # Filter by score -> Done in _process_single_frame
            # Compute similariy
            similarities_frame = torch.einsum('d,nd->n', query_features,
                                              frame_reid_features)

            is_query_frame = torch.zeros_like(similarities_frame)

            # If there is no annotations -> useless to search the gt.
            if gallery_frame.is_distractor:
                # Append similarities scores and true positive detections
                is_query.extend(is_query_frame.unbind(0))
                similarities.extend(similarities_frame.unbind(0))
                continue

            gt_width = gt_bbox[2] - gt_bbox[0]
            gt_height = gt_bbox[3] - gt_bbox[1]
            iou_threshold = min(0.5, (gt_width * gt_height) /
                                ((gt_width + 10) * (gt_height + 10)))

            # best first
            i_sorted_by_similarities = similarities_frame.argsort()
            ious = compute_ious(
                gt_bbox.unsqueeze(0),  # (1, 4)
                frame_bboxes[i_sorted_by_similarities],  # (n, 4)
            ).squeeze(0)  # (1, n) -> (n)
            for i in i_sorted_by_similarities:
                if ious[i] >= iou_threshold:
                    is_query_frame[i] = True
                    # count positive that has been correctly detected
                    count_true_positives += 1
                    break

            # Append similarities scores and true positive detections
            similarities.extend(similarities_frame)
            is_query.extend(is_query_frame)

        assert count_true_positives <= count_positives
        if count_true_positives == 0:
            return 0

        similarities_np = torch.stack(similarities).numpy()
        is_query_np = torch.stack(is_query).numpy().astype(bool)

        average_precision = compute_average_precision(is_query_np,
                                                      similarities_np)

        # penalize AP with the detection recall rate.
        recall_rate = count_true_positives / count_positives
        return average_precision * recall_rate

    def compute_metrics(self, results: list[Sample]) -> dict:
        assert len(results) == self.n_samples

        sum_average_precision = sum(
            self._compute_metrics_single_sample(result)  # compute AP
            for result in results)

        return {"mAP": sum_average_precision / len(results)}
