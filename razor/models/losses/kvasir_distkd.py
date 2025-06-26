# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ...registry import MODELS
from .utils import weighted_loss
from .MyRegEncoder.RegEncoderv2 import GenerateYPv1
from .MyRegEncoder.Model import Encoder
from .MyRegEncoder.Re import MaskResizer

def align(s, t):
    if s.shape[0] > t.shape[0]:
        s = s[:t.shape[0]]
    if s.shape[0] < t.shape[0]:
        padding=s.mean(dim=0,keepdim=True).repeat(t.shape[0]-s.shape[0],1).to(s.device)
        s = torch.cat([s, padding], dim=0)
    return s


def cosine_similarity(a, b, eps=1e-8):
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + eps)


def pearson_correlation(a, b, eps=1e-8):
    return cosine_similarity(a - a.mean(1, keepdim=True),
                             b - b.mean(1, keepdim=True), eps)


def inter_class_relation(y_s, y_t):
    return 1 - pearson_correlation(y_s, y_t).mean()


def intra_class_relation(y_s, y_t):
    return inter_class_relation(y_s.transpose(0, 1), y_t.transpose(0, 1))


class DISTLoss(nn.Module):

    def __init__(
            self,
            inter_loss_weight=1.0,
            intra_loss_weight=1.0,
            tau=1.0,
            loss_weight: float = 1.0,
            teacher_detach: bool = True,
    ):
        super(DISTLoss, self).__init__()
        self.inter_loss_weight = inter_loss_weight
        self.intra_loss_weight = intra_loss_weight
        self.tau = tau

        self.loss_weight = loss_weight
        self.teacher_detach = teacher_detach
        self.yp_generator = GenerateYPv1(class_num=1, img_height=256, img_width=256, device='cuda:0',
                                       model_path='/home/jz207/workspace/liull/MMDetection/Encoder_Kvasir_SEG_PreTrain_last.pth')
        self.mask_resizer = MaskResizer(image_size=(256, 256))

    def forward(self,
                s_feat: Tensor,
                t_feat: Tensor,
                ):
        s_input, s_rois = s_feat
        t_input, t_rois = t_feat

        s_input = self.mask_resizer.convert_to_full_size_mask(s_input, s_rois)
        t_input = self.mask_resizer.convert_to_full_size_mask(t_input, t_rois)

        x = align(s_input, t_input)
        N, C, H, W = t_input.shape

        preds_S=s_input
        preds_T=t_input

        roi = min(preds_S.shape[0],preds_T.shape[0])
        logits_S=preds_S[:roi]
        logits_T = preds_T[:roi]
        if self.teacher_detach:
            logits_T = logits_T.detach()
        y_s = (logits_S / self.tau).softmax(dim=1)
        y_t = (logits_T / self.tau).softmax(dim=1)
        inter_loss = self.tau ** 2 * inter_class_relation(y_s, y_t)
        intra_loss = self.tau ** 2 * intra_class_relation(y_s, y_t)
        kd_loss = self.inter_loss_weight * inter_loss + self.intra_loss_weight * intra_loss  # noqa
        return kd_loss * self.loss_weight
