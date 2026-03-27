import torch
import torch.nn as nn

class ColorJitter(nn.Module):
    """
    Randomly change the brightness, contrast, saturation and hue of an image.
    """
    def __init__(self, brightness=(-0.3, 0.3), contrast=(0.5, 1.5), saturation=1.0, hue=(-0.1, 0.1)):
        super(ColorJitter, self).__init__()
        self.bri = brightness
        self.con = contrast
        self.sat = saturation
        self.hue = hue

    def forward(self, enc_img_w_bg):
        b, c, _, _ = enc_img_w_bg.shape
        random_bri = (self.bri[0] - self.bri[1])*torch.rand(b, c, 1, 1) + self.bri[1]
        random_con = (self.con[0] - self.con[1])*torch.rand(b, c, 1, 1) + self.con[1]
        # random_hue = (self.hue[0] - self.hue[1])*torch.rand(b, c, 1, 1) + self.hue[1]

        enc_img_w_bg = enc_img_w_bg*random_con.to(enc_img_w_bg.device)
        enc_img_w_bg = enc_img_w_bg+random_bri.to(enc_img_w_bg.device)
        enc_img_w_bg = torch.clip(enc_img_w_bg, 0, 1)

        return enc_img_w_bg