import clip
from torch import nn


class CLIP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.clip_model, self.preprocess = clip.load("RN50", device='cpu', jit=False)
        self.label_token = clip.tokenize([str(i) for i in range(cfg.num_cls)])

    def forward(self, enc_img):
        img_f = self.clip_model.encode_image(enc_img)
        img_f = img_f / img_f.norm(dim=1, keepdim=True)

        text_f = self.clip_model.encode_text(self.label_token.to(enc_img.device))
        text_f = text_f / text_f.norm(dim=1, keepdim=True)

        logit_scale = self.clip_model.logit_scale.exp()
        logits = logit_scale * img_f @ text_f.t()

        return logits
