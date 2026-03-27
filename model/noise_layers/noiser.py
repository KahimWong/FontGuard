import random
from random import sample
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import get_rand_interop_type

class Noiser(nn.Module):

    def __init__(self, cfg):
        super(Noiser, self).__init__()
        self.noise_layers = cfg.noises
        self.font_img_size = cfg.font_img_size
        self.start_noise_epoch = cfg.start_noise_epoch
        self.full_noise_epoch = cfg.full_noise_epoch
        self.max_noise_p = cfg.max_noise_p

    def forward(self, enc_img, epoch):
        if epoch > self.start_noise_epoch:
            b = enc_img.shape[0]
            enc_img = (enc_img + 1) / 2  # [-1, 1] -> [0, 1]

            if self.full_noise_epoch == 0:
                apply_noise_p = self.max_noise_p
            else:
                apply_noise_p = self.max_noise_p * (epoch-self.start_noise_epoch) / self.full_noise_epoch  # fix 0.2
            random.shuffle(self.noise_layers)

            for layer in self.noise_layers:
                noise_img = layer(enc_img.clone())
                noise_idx =  random.sample(range(b), round(b*apply_noise_p))
                # fix size
                if noise_img.shape[2] != self.font_img_size or noise_img.shape[3] != self.font_img_size:
                    noise_img = F.interpolate(noise_img, size=self.font_img_size, mode=get_rand_interop_type())

                enc_img[noise_idx] = noise_img[noise_idx]

            enc_img = enc_img * 2 - 1  # [0, 1] -> [-1, 1]
        return enc_img

