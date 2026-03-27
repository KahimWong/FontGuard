import torch
import torch.nn as nn
import numpy as np

class Dropout(nn.Module):
    def __init__(self, keep_ratio_range):
        super(Dropout, self).__init__()
        self.keep_min = keep_ratio_range[0]
        self.keep_max = keep_ratio_range[1]

    def forward(self, enc_img_w_bg):
        dropout_mask = torch.zeros_like(enc_img_w_bg)
        mask_percent = np.random.uniform(self.keep_min, self.keep_max)
        mask = np.random.choice([0.0, 1.0], enc_img_w_bg.shape[2:], p=[1 - mask_percent, mask_percent])
        mask_tensor = torch.tensor(mask, device=enc_img_w_bg.device, dtype=torch.float)
        mask_tensor = mask_tensor.expand_as(enc_img_w_bg)
        enc_img_w_bg = enc_img_w_bg * mask_tensor + dropout_mask * (1-mask_tensor)
        return enc_img_w_bg


