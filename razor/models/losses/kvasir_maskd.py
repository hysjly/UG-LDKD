import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.runner.checkpoint import _load_checkpoint
from torch import Tensor
# from mmcv.runner.checkpoint import _load_checkpoint

# from ..builder import LOSSES

from ...registry import MODELS
from .utils import weighted_loss
from .MyRegEncoder.RegEncoderv2 import GenerateYPv1
from .MyRegEncoder.Model import Encoder
from .MyRegEncoder.Re import MaskResizer

@weighted_loss
def align(s: Tensor, t: Tensor) -> Tensor:
    """Align student feature map to match teacher feature map.
    Adjust channels and spatial size.

    Args:
        s (Tensor): Student mask tensor.
        t (Tensor): Teacher mask tensor.

    Returns:
        Tensor: Aligned student mask tensor.
    """
    s_channels = s.shape[1]
    t_channels = t.shape[1]
    t_height = t.shape[2]
    t_width = t.shape[3]

    if s_channels != t_channels or s.shape[2] != t_height or s.shape[3] != t_width:
        adjust_channels = nn.Sequential(
            nn.Conv2d(s_channels, t_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(t_channels),
            nn.ReLU(inplace=True)
        ).to(s.device)  # Ensure device consistency

        # Adjust channels
        s = adjust_channels(s)

        # Adjust spatial size if necessary
        if s.shape[2] != t_height or s.shape[3] != t_width:
            s = F.interpolate(s, size=(t_height, t_width), mode='bilinear', align_corners=True)

    return s
def dice_coeff(inputs):
    # inputs: [B, T, H*W]
    pred = inputs[:, None, :, :]
    target = inputs[:, :, None, :]

    mask = pred.new_ones(pred.size(0), target.size(1), pred.size(2))
    mask[:, torch.arange(mask.size(1)), torch.arange(mask.size(2))] = 0

    a = torch.sum(pred * target, -1)
    b = torch.sum(pred * pred, -1) + 1e-12
    c = torch.sum(target * target, -1) + 1e-12
    d = (2 * a) / (b + c)
    d = (d * mask).sum() / mask.sum()
    return d


class MaskModule(nn.Module):

    def __init__(self, channels, num_tokens=8, weight_mask=False):
        super().__init__()
        self.weight_mask = weight_mask
        self.mask_token = nn.Parameter(torch.randn(num_tokens, channels).normal_(0, 0.01))
        if self.weight_mask:
            self.prob = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, num_tokens, kernel_size=1)
            )
            self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels  # fan-out
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward_mask(self, x):
        N, C, H, W = x.shape
        mask_token = self.mask_token.expand(N, -1, -1)  # [N, T, C]
        k = x.view(N, -1, H * W)
        attn = mask_token @ k  # [N, T, H * W]
        attn = attn.sigmoid()
        attn = attn.view(N, -1, H, W)
        return attn

    def forward_prob(self, x):
        mask_probs = self.prob(x)  # [N, T, 1, 1]
        mask_probs = mask_probs.softmax(1).unsqueeze(2)  # [N, T, 1, 1, 1]
        return mask_probs

    def forward_train(self, x):
        mask = self.forward_mask(x)
        out = x.unsqueeze(1) * mask.unsqueeze(2)  # [N, T, C, H, W]
        # probs
        if self.weight_mask:
            mask_probs = self.forward_prob(x)
            #print(mask_probs.detach().flatten(1))
            out = out * mask_probs
        out = out.sum(1)
        # loss
        mask_loss = dice_coeff(mask.flatten(2))
        return out, mask_loss

    def forward(self, x):
        return self.forward_train(x)

#
# @LOSSES.register_module()
class MasKDLoss(nn.Module):

    def __init__(self, channels, num_tokens=6, weight_mask=True, custom_mask=True, custom_mask_warmup=1000, pretrained=None, loss_weight=1.):
        super().__init__()
        self.loss_weight = loss_weight
        self.weight_mask = weight_mask
        self.custom_mask = custom_mask
        self.custom_mask_warmup = custom_mask_warmup
        self.yp_generator = GenerateYPv1(class_num=1, img_height=256, img_width=256, device='cuda:0',
                                       model_path='/home/jz207/workspace/liull/MMDetection/Encoder_Kvasir_SEG_PreTrain_last.pth')
        self.mask_resizer = MaskResizer(image_size=(256, 256))

        self.mask_modules = nn.ModuleList([
            MaskModule(channels=c, num_tokens=num_tokens, weight_mask=weight_mask) for c in channels]
        )

        self.init_weights(pretrained)
        self._iter = 0

    def init_weights(self, pretrained=None):
        if pretrained is None:
            return
        ckpt = _load_checkpoint(pretrained, map_location='cpu')
        state_dict = {}
        for k, v in ckpt['state_dict'].items():
            if 'mask_modules' in k:
                state_dict[k[k.find('mask_modules'):]] = v
        self.load_state_dict(state_dict, strict=True)


    def forward(self,
                s_feat: Tensor,
                t_feat: Tensor,
                ) -> torch.Tensor:

        s_input, s_rois = s_feat
        t_input, t_rois = t_feat

        s_input = self.mask_resizer.convert_to_full_size_mask(s_input, s_rois)
        t_input = self.mask_resizer.convert_to_full_size_mask(t_input, t_rois)

        x = align(s_input, t_input)
        N, C, H, W = t_input.shape

        y_s_list=s_input
        y_t_list=t_input

        if not isinstance(y_s_list, (tuple, list)):
            y_s_list = (y_s_list, )
            y_t_list = (y_t_list, )
        assert len(y_s_list) == len(y_t_list) == len(self.mask_modules)

        losses = []
        for y_s, y_t, mask_module in zip(y_s_list, y_t_list, self.mask_modules):
            # predict the masks
            mask = mask_module.forward_mask(y_t)
            if self.custom_mask and self._iter >= self.custom_mask_warmup:
                if self._iter == self.custom_mask_warmup:
                    print('Start customizing masks using student\'s masks.')
                with torch.no_grad():
                    mask_s = mask_module.forward_mask(y_s)  # [N, T, H, W]
                mask = mask * mask_s

            # get the masked features
            masked_y_s = y_s.unsqueeze(1) * \
                mask.unsqueeze(2)  # [N, n_masks, C, H, W]
            masked_y_t = y_t.unsqueeze(1) * \
                mask.unsqueeze(2)  # [N, n_masks, C, H, W]

            # masked distillation
            loss = (masked_y_s - masked_y_t)**2
            loss = loss.sum((3, 4))  # [N, n_masks, C]
            loss = loss / mask.sum((2, 3)).unsqueeze(-1)
            if self.weight_mask:
                weights = mask_module.forward_prob(y_t).flatten(1)  # [N, T]
                loss = (loss.mean(-1) * weights).sum(-1)
            loss = loss.mean()
            losses.append(loss)

        loss = sum(losses)
        self._iter += 1
        return self.loss_weight * loss