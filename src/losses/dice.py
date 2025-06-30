import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self, data):
        fake_segm = data['fake_segm']
        real_segm = data['real_segm']
        # Ensure both are the same size
        if fake_segm.shape[-2:] != real_segm.shape[-2:]:
            fake_segm = F.interpolate(fake_segm, size=real_segm.shape[-2:], mode='bilinear', align_corners=False)
        numer = (2 * fake_segm * real_segm).sum()
        denom = (fake_segm + real_segm).sum() + 1e-8
        return 1 - numer / denom