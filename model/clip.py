import os
import clip

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Sequential, Module, Linear


class LASTED(nn.Module):
    def __init__(self, num_cls=4):
        super().__init__()
        self.clip_model, self.preprocess = clip.load("RN50", device='cpu', jit=False)
        self.output_layer = Sequential(
            nn.Linear(1024, 1280),
            nn.GELU(),
            nn.Linear(1280, 512),
        )
        self.fc = nn.Linear(512, num_cls)
        self.text_input = clip.tokenize([str(i) for i in range(num_cls)])

    def forward(self, img, is_train=True):
        if is_train:
            logits_per_img, _ = self.clip_model(img, self.text_input.to(img.device))
            return logits_per_img
        else:
            img_feat = self.clip_model.encode_image(img)
            img_feat = img_feat / img_feat.norm(dim=1, keepdim=True)
            return img_feat
