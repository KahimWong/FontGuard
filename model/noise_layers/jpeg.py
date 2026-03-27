import torch
import torch.nn as nn
from random import randint
from model.noise_layers.DiffJPEG.diffjpeg import DiffJPEG


class JPEG(nn.Module):
    def __init__(self, img_size=80, qf=(50, 99)):
        super(JPEG, self).__init__()
        self.qf = qf
        self.img_size = img_size

    def forward(self, enc_img_w_bg):
        rand_qf = randint(self.qf[0], self.qf[1])
        jpeg = DiffJPEG(height=self.img_size,
                        width=self.img_size,
                        differentiable=True,
                        quality=rand_qf).to(enc_img_w_bg.device)
        jpeg_img = jpeg(enc_img_w_bg)
        return  jpeg_img