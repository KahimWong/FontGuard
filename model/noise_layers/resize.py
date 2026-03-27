import torch.nn as nn
import torch.nn.functional as F

from model.noise_layers.crop import random_float
from utils import get_rand_interop_type

class Resize(nn.Module):
    def __init__(self, resize_ratio_range):
        super(Resize, self).__init__()
        self.resize_ratio_min = resize_ratio_range[0]
        self.resize_ratio_max = resize_ratio_range[1]

    def forward(self, enc_img_w_bg):
        resize_ratio = random_float(self.resize_ratio_min, self.resize_ratio_max)
        enc_img_w_bg = F.interpolate(enc_img_w_bg,
                                    scale_factor=(resize_ratio, resize_ratio),
                                    mode=get_rand_interop_type())
        return enc_img_w_bg
