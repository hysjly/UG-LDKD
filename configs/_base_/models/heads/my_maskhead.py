# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_conv_layer, build_upsample_layer
from mmcv.ops.carafe import CARAFEPack
from mmengine.config import ConfigDict
from mmengine.model import BaseModule, ModuleList
from mmengine.structures import InstanceData
from torch import Tensor
from torch.nn.modules.utils import _pair

from mmdet.models.task_modules.samplers import SamplingResult
from mmdet.models.utils import empty_instances
from mmdet.registry import MODELS
from mmdet.structures.mask import mask_target
from mmdet.utils import ConfigType, InstanceList, OptConfigType, OptMultiConfig

BYTES_PER_FLOAT = 4
GPU_MEM_LIMIT = 1024**3  # 1 GB memory limit

@MODELS.register_module()
class FCNMaskHeadv1(BaseModule):

    def __init__(self,
                 num_convs: int = 4,
                 roi_feat_size: int = 14,
                 in_channels: int = 256,
                 conv_kernel_size: int = 3,
                 conv_out_channels: int = 256,
                 num_classes: int = 80,
                 class_agnostic: int = False,
                 upsample_cfg: ConfigType = dict(
                     type='deconv', scale_factor=2),
                 conv_cfg: OptConfigType = None,
                 norm_cfg: OptConfigType = None,
                 predictor_cfg: ConfigType = dict(type='Conv'),
                 loss_mask: ConfigType = dict(
                     type='CrossEntropyLoss', use_mask=True, loss_weight=1.0),
                 init_cfg: OptMultiConfig = None) -> None:
        assert init_cfg is None, 'To prevent abnormal initialization ' \
                                 'behavior, init_cfg is not allowed to be set'
        super().__init__(init_cfg=init_cfg)
        self.upsample_cfg = upsample_cfg.copy()
        if self.upsample_cfg['type'] not in [
                None, 'deconv', 'nearest', 'bilinear', 'carafe'
        ]:
            raise ValueError(
                f'Invalid upsample method {self.upsample_cfg["type"]}, '
                'accepted methods are "deconv", "nearest", "bilinear", '
                '"carafe"')
        self.num_convs = num_convs
        # WARN: roi_feat_size is reserved and not used
        self.roi_feat_size = _pair(roi_feat_size)
        self.in_channels = in_channels
        self.conv_kernel_size = conv_kernel_size
        self.conv_out_channels = conv_out_channels
        self.upsample_method = self.upsample_cfg.get('type')
        self.scale_factor = self.upsample_cfg.pop('scale_factor', None)
        self.num_classes = num_classes
        self.class_agnostic = class_agnostic
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.predictor_cfg = predictor_cfg
        self.loss_mask = MODELS.build(loss_mask)

        self.convs = ModuleList()
        for i in range(self.num_convs):
            in_channels = (
                self.in_channels if i == 0 else self.conv_out_channels)
            padding = (self.conv_kernel_size - 1) // 2
            self.convs.append(
                ConvModule(
                    in_channels,
                    self.conv_out_channels,
                    self.conv_kernel_size,
                    padding=padding,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg))
        upsample_in_channels = (
            self.conv_out_channels if self.num_convs > 0 else in_channels)
        upsample_cfg_ = self.upsample_cfg.copy()
        if self.upsample_method is None:
            self.upsample = None
        elif self.upsample_method == 'deconv':
            upsample_cfg_.update(
                in_channels=upsample_in_channels,
                out_channels=self.conv_out_channels,
                kernel_size=self.scale_factor,
                stride=self.scale_factor)
            self.upsample = build_upsample_layer(upsample_cfg_)
        elif self.upsample_method == 'carafe':
            upsample_cfg_.update(
                channels=upsample_in_channels, scale_factor=self.scale_factor)
            self.upsample = build_upsample_layer(upsample_cfg_)
        else:
            # suppress warnings
            align_corners = (None
                             if self.upsample_method == 'nearest' else False)
            upsample_cfg_.update(
                scale_factor=self.scale_factor,
                mode=self.upsample_method,
                align_corners=align_corners)
            self.upsample = build_upsample_layer(upsample_cfg_)

        out_channels = 1 if self.class_agnostic else self.num_classes
        logits_in_channel = (
            self.conv_out_channels
            if self.upsample_method == 'deconv' else upsample_in_channels)
        self.conv_logits = build_conv_layer(self.predictor_cfg,
                                            logits_in_channel, out_channels, 1)
        self.relu = nn.ReLU(inplace=True)
        self.debug_imgs = None

    def init_weights(self) -> None:
        """Initialize the weights."""
        super().init_weights()
        for m in [self.upsample, self.conv_logits]:
            if m is None:
                continue
            elif isinstance(m, CARAFEPack):
                m.init_weights()
            elif hasattr(m, 'weight') and hasattr(m, 'bias'):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    # 修改点：增加rois可选参数，当rois不为空时返回(mask_preds, rois)，否则返回mask_preds
    def forward(self, x: Tensor, rois: Optional[Tensor] = None):
        """Forward features from the upstream network.

        Args:
            x (Tensor): Extract mask RoI features.
            rois (Tensor, optional): The RoIs associated with x. Defaults to None.

        Returns:
            Tensor or (Tensor, Tensor): Predicted foreground masks,
            and optionally the rois if provided.
        """
        for conv in self.convs:
            x = conv(x)
        if self.upsample is not None:
            x = self.upsample(x)
            if self.upsample_method == 'deconv':
                x = self.relu(x)
        mask_preds = self.conv_logits(x)

        # 如果rois不为None，则返回 (mask_preds, rois)，否则只返回mask_preds
        if rois is not None:
            return mask_preds, rois
        else:
            return mask_preds

    def get_targets(self, sampling_results: List[SamplingResult],
                    batch_gt_instances: InstanceList,
                    rcnn_train_cfg: ConfigDict) -> Tensor:
        pos_proposals = [res.pos_priors for res in sampling_results]
        pos_assigned_gt_inds = [
            res.pos_assigned_gt_inds for res in sampling_results
        ]
        gt_masks = [res.masks for res in batch_gt_instances]
        mask_targets = mask_target(pos_proposals, pos_assigned_gt_inds,
                                   gt_masks, rcnn_train_cfg)
        return mask_targets

    def loss_and_target(self, mask_preds: Tensor,
                        sampling_results: List[SamplingResult],
                        batch_gt_instances: InstanceList,
                        rcnn_train_cfg: ConfigDict) -> dict:
        mask_targets = self.get_targets(
            sampling_results=sampling_results,
            batch_gt_instances=batch_gt_instances,
            rcnn_train_cfg=rcnn_train_cfg)

        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])

        loss = dict()
        if mask_preds.size(0) == 0:
            loss_mask = mask_preds.sum()
        else:
            if self.class_agnostic:
                loss_mask = self.loss_mask(mask_preds, mask_targets,
                                           torch.zeros_like(pos_labels))
            else:
                loss_mask = self.loss_mask(mask_preds, mask_targets,
                                           pos_labels)
        loss['loss_mask'] = loss_mask
        return dict(loss_mask=loss, mask_targets=mask_targets)

    def predict_by_feat(self,
                        mask_preds: Tuple[Tensor],
                        results_list: List[InstanceData],
                        batch_img_metas: List[dict],
                        rcnn_test_cfg: ConfigDict,
                        rescale: bool = False,
                        activate_map: bool = False) -> InstanceList:
        assert len(mask_preds) == len(results_list) == len(batch_img_metas)

        for img_id in range(len(batch_img_metas)):
            img_meta = batch_img_metas[img_id]
            results = results_list[img_id]
            bboxes = results.bboxes
            if bboxes.shape[0] == 0:
                results_list[img_id] = empty_instances(
                    [img_meta],
                    bboxes.device,
                    task_type='mask',
                    instance_results=[results],
                    mask_thr_binary=rcnn_test_cfg.mask_thr_binary)[0]
            else:
                im_mask = self._predict_by_feat_single(
                    mask_preds=mask_preds[img_id],
                    bboxes=bboxes,
                    labels=results.labels,
                    img_meta=img_meta,
                    rcnn_test_cfg=rcnn_test_cfg,
                    rescale=rescale,
                    activate_map=activate_map)
                results.masks = im_mask
        return results_list

    def _predict_by_feat_single(self,
                                mask_preds: Tensor,
                                bboxes: Tensor,
                                labels: Tensor,
                                img_meta: dict,
                                rcnn_test_cfg: ConfigDict,
                                rescale: bool = False,
                                activate_map: bool = False) -> Tensor:
        scale_factor = bboxes.new_tensor(img_meta['scale_factor']).repeat(
            (1, 2))
        img_h, img_w = img_meta['ori_shape'][:2]
        device = bboxes.device

        if not activate_map:
            mask_preds = mask_preds.sigmoid()
        else:
            # In AugTest, has been activated before
            mask_preds = bboxes.new_tensor(mask_preds)

        if rescale:
            bboxes /= scale_factor
        else:
            w_scale, h_scale = scale_factor[0, 0], scale_factor[0, 1]
            img_h = np.round(img_h * h_scale.item()).astype(np.int32)
            img_w = np.round(img_w * w_scale.item()).astype(np.int32)

        N = len(mask_preds)
        if device.type == 'cpu':
            num_chunks = N
        else:
            num_chunks = int(
                np.ceil(N * int(img_h) * int(img_w) * BYTES_PER_FLOAT /
                        GPU_MEM_LIMIT))
            assert (num_chunks <= N), \
                'Default GPU_MEM_LIMIT is too small; try increasing it'
        chunks = torch.chunk(torch.arange(N, device=device), num_chunks)

        threshold = rcnn_test_cfg.mask_thr_binary
        im_mask = torch.zeros(
            N,
            img_h,
            img_w,
            device=device,
            dtype=torch.bool if threshold >= 0 else torch.uint8)

        if not self.class_agnostic:
            mask_preds = mask_preds[range(N), labels][:, None]

        for inds in chunks:
            masks_chunk, spatial_inds = _do_paste_mask(
                mask_preds[inds],
                bboxes[inds],
                img_h,
                img_w,
                skip_empty=device.type == 'cpu')

            if threshold >= 0:
                masks_chunk = (masks_chunk >= threshold).to(dtype=torch.bool)
            else:
                masks_chunk = (masks_chunk * 255).to(dtype=torch.uint8)

            im_mask[(inds, ) + spatial_inds] = masks_chunk
        return im_mask

def _do_paste_mask(masks: Tensor,
                   boxes: Tensor,
                   img_h: int,
                   img_w: int,
                   skip_empty: bool = True) -> tuple:
    device = masks.device
    if skip_empty:
        x0_int, y0_int = torch.clamp(
            boxes.min(dim=0).values.floor()[:2] - 1,
            min=0).to(dtype=torch.int32)
        x1_int = torch.clamp(
            boxes[:, 2].max().ceil() + 1, max=img_w).to(dtype=torch.int32)
        y1_int = torch.clamp(
            boxes[:, 3].max().ceil() + 1, max=img_h).to(dtype=torch.int32)
    else:
        x0_int, y0_int = 0, 0
        x1_int, y1_int = img_w, img_h
    x0, y0, x1, y1 = torch.split(boxes, 1, dim=1)  # each is Nx1

    N = masks.shape[0]

    img_y = torch.arange(y0_int, y1_int, device=device).to(torch.float32) + 0.5
    img_x = torch.arange(x0_int, x1_int, device=device).to(torch.float32) + 0.5
    img_y = (img_y - y0) / (y1 - y0) * 2 - 1
    img_x = (img_x - x0) / (x1 - x0) * 2 - 1

    if not torch.onnx.is_in_onnx_export():
        if torch.isinf(img_x).any():
            inds = torch.where(torch.isinf(img_x))
            img_x[inds] = 0
        if torch.isinf(img_y).any():
            inds = torch.where(torch.isinf(img_y))
            img_y[inds] = 0

    # gx = img_x[:, None, :].expand(N, img_y.size(0), img_x.size(0))
    # gy = img_y[:, :, None].expand(N, img_y.size(0), img_x.size(0))
    # grid = torch.stack([gx, gy], dim=3)
    #
    # img_masks = F.grid_sample(
    #     masks.to(dtype=torch.float32), grid, align_corners=False)

    img_y_coord = torch.arange(y0_int, y1_int, device=device, dtype=torch.float32) + 0.5
    img_x_coord = torch.arange(x0_int, x1_int, device=device, dtype=torch.float32) + 0.5

    gy, gx = torch.meshgrid(img_y_coord, img_x_coord, indexing='ij')
    # gy.shape: [H, W], gx.shape: [H, W]

    # 将gy/gx映射到[-1,1]范围
    # gy = (gy - y0) / (y1 - y0) * 2 - 1
    # gx = (gx - x0) / (x1 - x0) * 2 - 1

    # 为了适配N个masks，需要在前面增加batch维度N
    gy = gy.unsqueeze(0).expand(N, -1, -1)
    gx = gx.unsqueeze(0).expand(N, -1, -1)

    grid = torch.stack([gx, gy], dim=3)  # (N, H, W, 2)
    img_masks = F.grid_sample(masks.float(), grid, align_corners=False)

    if skip_empty:
        return img_masks[:, 0], (slice(y0_int, y1_int), slice(x0_int, x1_int))
    else:
        return img_masks[:, 0], ()

