import torch
import torch.nn as nn
from kornia.geometry.transform import get_perspective_transform, warp_perspective


class PerspectiveWarp(nn.Module):
    def __init__(self, img_size=80, max_trans_f=0.1):
        super(PerspectiveWarp, self).__init__()
        self.font_img_size = img_size
        self.max_warp_f = max_trans_f

    def forward(self, enc_img_w_bg):  # [b, 3, h, w]
        b = enc_img_w_bg.shape[0]
        rand_corners = torch.rand(b, 4, 2, device=enc_img_w_bg.device) # [b, 4, 2]
        rand_corners = torch.round(rand_corners*self.max_warp_f * self.font_img_size)
        rand_corners = (-rand_corners-rand_corners)*torch.rand_like(rand_corners) + rand_corners
        rand_corners[:, 1, 0] += self.font_img_size
        rand_corners[:, 2, 0] += self.font_img_size
        rand_corners[:, 2, 1] += self.font_img_size
        rand_corners[:, 3, 1] += self.font_img_size

        dst = torch.tensor([[0, 0],
                            [self.font_img_size, 0],
                            [self.font_img_size, self.font_img_size],
                            [0, self.font_img_size]], device=enc_img_w_bg.device, dtype=torch.float32)
        dst = dst[None, ...].repeat(b, 1, 1)

        M = get_perspective_transform(rand_corners,  dst)
        img_warp = warp_perspective(enc_img_w_bg, M, dsize=(self.font_img_size, self.font_img_size))
        return  img_warp