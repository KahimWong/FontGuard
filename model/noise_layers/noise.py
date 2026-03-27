import torch
import torch.nn as nn

class Noise(nn.Module):
    def __init__(self, noise_var=0.02):
        super(Noise, self).__init__()
        self.noise_var = noise_var

    def forward(self, enc_img_w_bg):
        noise = torch.empty(enc_img_w_bg.shape).normal_(mean=0.0, std=1.0) * self.noise_var
        noise = noise.to(enc_img_w_bg.device)
        noise_img = noise + enc_img_w_bg
        noise_img = torch.clamp(noise_img, min=0.0, max=1.0)
        return  noise_img