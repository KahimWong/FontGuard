import torch
import torch.nn as nn
from numpy.random import uniform
from kornia.filters import GaussianBlur2d, MedianBlur, MotionBlur

class Blur(nn.Module):
    def __init__(self, kernel_size=3, max_sigma=3):
        super(Blur, self).__init__()
        self.kernel_size = kernel_size
        self.max_sigma = max_sigma
        self.blur_list = ['GaussianBlur2d', 'MedianBlur', 'MotionBlur']

    def forward(self, enc_img_w_bg):
        # TODO: diff blur kernels in one batch
        # sample blur type
        blur_type = self.blur_list[torch.randint(0, len(self.blur_list), (1,))]
        if blur_type == 'GaussianBlur2d':
            rand_sigma = torch.rand(1) * self.max_sigma
            blur = eval(blur_type)(kernel_size=(self.kernel_size, self.kernel_size), sigma=(rand_sigma, rand_sigma))
        elif blur_type == 'MedianBlur':
            blur = eval(blur_type)(kernel_size=(self.kernel_size, self.kernel_size))
        elif blur_type == 'MotionBlur':
            rand_angle = uniform()*360
            rand_direction = -2*uniform()+1   # sample from [-1, 1] uniformly
            blur = eval(blur_type)(kernel_size=self.kernel_size, angle=rand_angle, direction=rand_direction)
        blur_img = blur(enc_img_w_bg)
        return blur_img