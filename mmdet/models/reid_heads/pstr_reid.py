import torch
import torch.nn as nn
from mmengine.model import BaseModule
from torch import Tensor
from torch.nn import functional as F

from mmdet.models.dense_heads import (LabeledMatchingLayerQ,
                                      UnlabeledMatchingLayer)
from mmdet.models.layers.transformer import PartAttentionDecoder
from mmdet.registry import MODELS
from mmdet.structures.reid_det_data_sample import ReIDDetSampleList
from mmdet.utils import ConfigType, OptConfigType

N_PSTR_DECODER_LAYERS = 3
LOSS_DICT_KEY_TEMPLATE_OIM = "d{}.loss_oim_s{}"
LOSS_DICT_KEY_TEMPLATE_TRIPLET = "d{}.loss_tri_s{}"


@MODELS.register_module()
class PSTRHeadReID(BaseModule):
    """
    PSTR Head ReID, computes the ReID features from PSTR detections output
    and frame's features maps.

    NOTE: default value are the same as original PSTR.

    Args:
        decoder (:obj:`ConfigType`): Config dict for ReID decoder.
        num_person (int): Number of person IDs in the dataset.
        queue_size (int): Size of unlabeled queue for loss computation.
            Default to False.
        temperature (int): Temperature of labeled detection
            Default to 15.
        unlabeled_weight (int): Temperature of unlabeled detection
            Default to 10.
        oim_weight (float): Weight for OIM loss. Default to 0.5
        triplet_loss (:obj:`ConfigType`): Config dict for a triplet loss.
            Defaults to `TripletLoss`.
    """

    def __init__(
        self,
        decoder: ConfigType,
        num_person: int,
        queue_size: int = 5000,
        unlabeled_weight: int = 10,
        temperature: int = 15,
        oim_weight: float = 0.5,
        triplet_loss: OptConfigType = dict(
            type="TripletLoss",
            margin=.3,  # Default
            loss_weight=0.5,
            hard_mining=True  # Default
        )  # TripletLoss
    ):
        super().__init__()

        self.decoder = PartAttentionDecoder(**decoder)

        self.num_person = num_person
        self.queue_size = queue_size

        self.unlabeled_weight = unlabeled_weight
        self.temperature = temperature
        self.oim_weight = oim_weight

        self.triplet_loss = MODELS.build(
            triplet_loss) if triplet_loss else None

        self._init_layers()

    def _init_layers(self):
        self.n_dim_reid = 256

        num_reid_decoder = 3

        self.labeled_matching_layers = nn.ModuleList([
            LabeledMatchingLayerQ(self.num_person, self.n_dim_reid)
            for _ in range(num_reid_decoder)
        ])

        self.unlabeled_matching_layers = nn.ModuleList([
            UnlabeledMatchingLayer(self.queue_size, self.n_dim_reid)
            for _ in range(num_reid_decoder)
        ])

    def _forward_decoder(
        self,
        detection_decoder_states: Tensor,
        references: Tensor,
        spatial_shapes: Tensor,
        valid_ratios: Tensor,
        multi_scale_features_maps_flattened: list[Tensor],
    ) -> Tensor:
        """
        Compute raw ReID features from detector and features maps.
        n_decoder_layers=1 in inference and n_decoder_layers=3 in training.

        Args:
            detection_decoder_states (Tensor): The outputs of detector which
                have one in inference and 3 (number of decoder layers) in
                training. (n_decoder_layers, bs, n_queries, n_dim_det)
            references (Tensor): References point from deformable attention.
                (n_decoder_layers, bs, n_queries, 4)
            spatial_shapes (Tensor): The dimension of features maps used in
                detection, before flattening. (n_used_detection, 2)
            valid_ratios (Tensor): Valid ratios of detections, in case some are
                annotatated to be ignored. (bs, ratio_dim, 2).
            multi_scale_features_maps (list[Tensor]): The features maps
                from the backbone/neck. A list of n_scales, each element
                is (bs, , x_dim*y_dim, n_dim_neck).

        Returns:
            Tensor: Result of the raw ReID features
            (n_scales, n_decoder_layers, bs, n_queries, n_dim_reid)
        """
        n_scales = len(multi_scale_features_maps_flattened)

        # Inference ReID: do not input from all 3 layers of the decoder.
        if not self.training:
            assert detection_decoder_states.shape[0] == 1
            last_state = detection_decoder_states.squeeze(0)

            assert references.shape[0] == 1
            reference = references.squeeze(0)

            # (n_scales) [(bs, num_queries, n_dim_reid)]
            inter_reid_states = [
                self.decoder(
                    query=last_state,
                    value=features_maps_flattened,
                    reference_points=reference,
                    spatial_shapes=spatial_shapes,
                    valid_ratios=valid_ratios,
                ) for features_maps_flattened in
                multi_scale_features_maps_flattened
            ]

            # (n_scales, n_decoder_layers=1, bs, num_queries, n_dim_reid)
            inter_reid_states = torch.stack(inter_reid_states).unsqueeze(1)
            return inter_reid_states

        number_decoder_layers = references.shape[0]
        assert number_decoder_layers == N_PSTR_DECODER_LAYERS, \
            f"PSTR needs exactly {N_PSTR_DECODER_LAYERS} layers from decoder"

        inter_reid_states = [
            self.decoder(
                query=detection_decoder_states[i_layer],
                value=multi_scale_features_maps_flattened[i_scale],
                reference_points=references[i_layer],
                spatial_shapes=spatial_shapes,
                valid_ratios=valid_ratios,
            )
            # Per layer of decoder in detector
            for i_layer in range(number_decoder_layers)
            # Per level of scale in backbone/neck
            for i_scale in range(n_scales)
        ]

        inter_reid_states_stacked = torch.stack(inter_reid_states)
        inter_reid_states_reshaped = inter_reid_states_stacked.view(
            n_scales, number_decoder_layers,
            *inter_reid_states_stacked.shape[1:]
            # Delete deformable detr dimension, only use one in Deformable
            # DETR in PSTR.
        )

        return inter_reid_states_reshaped

    def forward(
        self,
        detection_decoder_states: Tensor,
        references: Tensor,
        spatial_shapes: Tensor,
        valid_ratios: Tensor,
        multi_scale_features_maps: tuple[Tensor],
    ) -> list[Tensor]:
        """
        Compute ReID features from detector and features maps.

        Args:
            detection_decoder_states (Tensor): The outputs of detector
                (n_decoder_layers, bs, n_queries, n_dim_det).
            references (Tensor): References point from deformable attention.
                (n_decoder_layers, bs, n_queries, 4)
            spatial_shapes (Tensor): The dimension of features maps used in
                detection, before flattening. (n_used_detection, 2)
            valid_ratios (Tensor): Valid ratios of detections, in case some are
                annotatated to be ignored. (bs, ratio_dim, 2).
            multi_scale_features_maps (tuple[Tensor]): The features maps
                from the backbone/neck. A n_scales-tuple, each element
                is (bs, n_dim_neck, x_dim, y_dim).

        Returns:
            list[Tensor]: The list of ReID features by scale, n_scale = 3.
                Each scale is (n_decoder_layers, bs, n_queries, n_dim_reid).
                n_decoder_layers (in the return) depends if the model perform
                inference or loss computation, see _forward_decoder.
        """
        assert detection_decoder_states.shape[0] == references.shape[0]
        assert len(multi_scale_features_maps) == 3

        # Last dimensions (W, H) are flattened
        # Permute spatial dims (num_value) with features dim (n_dim)
        # (num_scales) [(bs, num_value, n_dim)]
        multi_scale_features_maps_flattened = [
            feature_map.flatten(2).permute(0, 2, 1)
            for feature_map in list(multi_scale_features_maps)
        ]

        # Inference mode needs output from last layer of the detector.
        # Other layers are only use during training to make it easier for the
        # model to learn.
        if not self.training:
            # (1, bs, n_queries, n_dim_det)
            detection_decoder_states = detection_decoder_states[-1:]
            references = references[-1:]  # (1, bs, n_queries, 4)

        # NOTE: n_decoder_layers=3 if self.training else n_decoder_layers=1
        # (scales, n_decoder_layers, batch_size, num_queries, n_features)
        inter_reid_states = self._forward_decoder(
            detection_decoder_states,
            references,
            spatial_shapes,
            valid_ratios,
            multi_scale_features_maps_flattened,
        )

        # Center the reid features, average over the num_queries.
        # (n_scales) [n_decoder_layers, bs, num_queries, n_features]
        reid_outputs: list[Tensor] = [
            inter_reid_state -
            torch.mean(inter_reid_state, dim=2, keepdim=True)
            for inter_reid_state in inter_reid_states
        ]

        return reid_outputs

    def loss_single_layer_scale(
        self,
        batch_reid_features: Tensor,
        batch_assigned_person_ids: Tensor,
        data_samples: ReIDDetSampleList,
        i_layer: int,
        i_scale: int,
    ) -> dict[str, Tensor]:
        """
        Compute the ReID losses (OIM and Triplet, if the former if set).
        For a single decoder layer and a single scale.

        Args:
            batch_reid_features (Tensor): ReID features by and sample.
                (bs, n_queries, n_dim_reid)
            batch_assigned_person_ids (Tensor): Result of the
                the matching from assigner, 0 is not assigned and other value
                the person ID assigned to it (-1 is for detection kept but
                no ReID annotations) (bs, n_queries)
            data_samples (ReIDDetSampleList): List of annotations, length
                equals to batch_size.
            i_layer (int): index of decoder layer.
            i_scale (int): index of features map scale.

        Returns:
            dict[str, Tensor]: The keys are the loss based on the input
            (scales) and value is the loss value.
        """
        batch_size = len(data_samples)
        assert batch_size == batch_reid_features.shape[
            0] == batch_assigned_person_ids.shape[0]

        assert batch_reid_features.shape[-1] == self.n_dim_reid

        triplet_loss_key = LOSS_DICT_KEY_TEMPLATE_TRIPLET.format(
            i_layer, i_scale)
        oim_loss_key = LOSS_DICT_KEY_TEMPLATE_OIM.format(i_layer, i_scale)
        losses = {}

        # Flatten inputs. We can compute the loss on the batch directly
        # (bs, num_queries, n_dim_reid) -> (bs * num_queries, n_dim_reid)
        batch_reid_features_flatten = batch_reid_features.reshape(
            -1, self.n_dim_reid)
        # (bs, num_queries) -> (bs * num_queries)
        batch_assigned_person_ids_flatten = batch_assigned_person_ids.flatten()
        detection_is_assigned = batch_assigned_person_ids_flatten != 0
        # (n_keep, n_dim_reid)
        assigned_reid_features = F.normalize(
            batch_reid_features_flatten[detection_is_assigned])
        # (n_keep)
        only_assigned_person_ids = batch_assigned_person_ids_flatten[
            detection_is_assigned]

        # Perform cosine similarity with the labeled memory
        labeled_outputs = self.labeled_matching_layers[i_layer](
            assigned_reid_features,
            only_assigned_person_ids,
        )

        (
            labeled_logits,  # (n_assigned, self.num_person)
            labeled_reid_features,  # (n_positive, n_dim_reid)
            labeled_person_ids,  # (n_positive)
        ) = labeled_outputs
        labeled_logits *= self.temperature

        # (n_assigned, self.queue_size)
        unlabeled_logits = self.unlabeled_matching_layers[i_layer](
            assigned_reid_features, only_assigned_person_ids)
        unlabeled_logits *= self.unlabeled_weight

        # (n_keep, self.num_person + self.queue_size)
        matching_scores = torch.cat((labeled_logits, unlabeled_logits), dim=1)

        # (n_keep, self.num_person + self.queue_size)
        probabilities = F.softmax(matching_scores, dim=1)
        # (n_keep, self.num_person + self.queue_size)
        focal_probabilities = ((1 - probabilities + 1e-12)**2 *
                               (probabilities + 1e-12).log())

        # NOTE: original PSTR average counting ignore_index value
        # while reduction does not take them into account.
        loss_oim = F.nll_loss(
            focal_probabilities,
            only_assigned_person_ids,
            reduction="none",
            ignore_index=-1).mean()
        loss_oim *= self.oim_weight
        losses[oim_loss_key] = loss_oim

        if not self.triplet_loss:
            return {oim_loss_key: loss_oim}

        # In PSTR version, the triplet loss module is designed to remove
        # unlabeled detection. In the current, it's not designed for this.
        # So we manage here:
        # (1) we need more 2 different reid_labels, else equals 0
        # (2) we need input positive example only.
        num_positives = len(
            set(person_id
                for person_id in only_assigned_person_ids.cpu().tolist()
                if person_id != -1))
        if num_positives < 2:
            losses[triplet_loss_key] = assigned_reid_features.new_zeros(1)[0]
            return losses

        is_labeled = only_assigned_person_ids > 0

        # (2 * n_positive, n_dim_reid)
        # We use the current batch features and the ones in the matching
        # layer's lookup table.
        positive_reid_features = torch.cat(
            (assigned_reid_features[is_labeled], labeled_reid_features))
        # (2 * n_positive)
        positive_person_ids = torch.cat(
            (only_assigned_person_ids[is_labeled], labeled_person_ids))

        loss_triplet = self.triplet_loss(positive_reid_features,
                                         positive_person_ids)

        losses[triplet_loss_key] = loss_triplet
        return losses

    def loss_single_layer(
        self,
        all_scales_batch_reid_features: Tensor,
        batch_assigned_person_ids: Tensor,
        data_samples: ReIDDetSampleList,
        i_layer: int,
    ) -> dict[str, Tensor]:
        """
        Compute the ReID losses (OIM and Triplet, the former if set).
        For a single decoder layer

        Args:
            all_scales_batch_reid_features (Tensor): ReID features by scale
                and sample. (n_scales, bs, n_queries, n_dim_reid)
            batch_assigned_person_ids (Tensor): Result of the
                the matching from assigner, 0 is not assigned and other value
                the person ID assigned to it (-1 is for detection kept but
                no ReID annotations) (bs, n_queries)
            data_samples (ReIDDetSampleList): List of annotations, length
                equals to batch_size.
            i_layer (int): index of decoder layer.

        Returns:
            dict[str, Tensor]: The keys are the loss based on the input
            (scales) and value is the loss value.
        """
        num_scales = all_scales_batch_reid_features.shape[0]

        all_scales_reid_loss: dict[str, Tensor] = dict()
        for i_scale in reversed(range(num_scales)):
            all_scales_reid_loss |= self.loss_single_layer_scale(
                all_scales_batch_reid_features[i_scale],
                batch_assigned_person_ids,
                data_samples,
                i_layer,
                i_scale,
            )

        return all_scales_reid_loss

    def loss(
        self,
        detection_decoder_states: Tensor,
        references: Tensor,
        spatial_shapes: Tensor,
        valid_ratios: Tensor,
        multi_scale_features_maps: tuple[Tensor],
        all_layers_batch_assigned_person_ids: Tensor,
        data_samples: ReIDDetSampleList,
    ) -> dict[str, Tensor]:
        """
        Compute the ReID losses (OIM and Triplet, the former if set).

        Args:
            detection_decoder_states (Tensor): The outputs of detector which
                have one in inference and 3 (number of decoder layers) in
                training. (n_decoder_layers, bs, n_queries, n_dim_det)
            references (Tensor): References point from deformable attention.
                (n_decoder_layers, bs, n_queries, 4)
            spatial_shapes (Tensor): The dimension of features maps used in
                detection, before flattening. (n_used_detection, 2)
            valid_ratios (Tensor): Valid ratios of detections, in case some are
                annotatated to be ignored. (bs, ratio_dim, 2).
            multi_scale_features_maps (tuple[Tensor]): The features maps
                from the backbone/neck. A n_scales-tuple, each element
                is (bs, n_dim_neck, x_dim, y_dim).
            all_layers_batch_assigned_person_ids (Tensor): Result of the
                the matching from assigner, 0 is not assigned and other value
                the person ID assigned to it (-1 is for detection kept but no
                ReID annotations) (n_decoder_layers, bs, n_queries)
            data_samples (ReIDDetSampleList): List of annotations, length
                equals to batch_size.

        Returns:
            dict[str, Tensor]: The keys are the loss based on the input
            (scales and decoder outputs) and value is the loss value.
        """

        # (num_layers, num_scales, batch_size, num_queries, n_features)
        all_layers_all_scales_batch_reid_features = torch.stack(
            self.forward(
                detection_decoder_states,
                references,
                spatial_shapes,
                valid_ratios,
                multi_scale_features_maps,
            )).permute((1, 0, 2, 3, 4))

        all_layers_all_scales_batchreid_loss: dict[str, Tensor] = dict()

        num_layers = all_layers_all_scales_batch_reid_features.shape[0]
        for i_layer in reversed(range(num_layers)):
            all_layers_all_scales_batchreid_loss |= self.loss_single_layer(
                all_layers_all_scales_batch_reid_features[i_layer],
                all_layers_batch_assigned_person_ids[i_layer],
                data_samples,
                i_layer,
            )

        return all_layers_all_scales_batchreid_loss
