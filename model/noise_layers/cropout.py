import torch
import torch.nn as nn
from model.noise_layers.crop import get_random_rectangle_inside


class Cropout(nn.Module):
    def __init__(self, height_ratio_range, width_ratio_range):
        super(Cropout, self).__init__()
        self.height_ratio_range = height_ratio_range
        self.width_ratio_range = width_ratio_range

    def forward(self, enc_img_w_bg):
        cropout_bg = torch.zeros_like(enc_img_w_bg)
        cropout_mask = torch.zeros_like(enc_img_w_bg)
        h_start, h_end, w_start, w_end = get_random_rectangle_inside(image=enc_img_w_bg,
                                                                     height_ratio_range=self.height_ratio_range,
                                                                     width_ratio_range=self.width_ratio_range)
        cropout_mask[:, :, h_start:h_end, w_start:w_end] = 1
        enc_img_w_bg = enc_img_w_bg * cropout_mask + cropout_bg * (1-cropout_mask)
        return  enc_img_w_bg