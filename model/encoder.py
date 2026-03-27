import torch
import torch.nn as nn

from model.conv_bn_relu import ConvBNRelu

class Combinator(nn.Module):
    def __init__(self, num_sty_feat, msg_len, dim):
        super(Combinator, self).__init__()
        self.message_enc = nn.Linear(msg_len, dim)
        self.content_enc = nn.Sequential(
            ConvBNRelu(256, 256, 3, 5),
            ConvBNRelu(256, 256, 3),
            nn.Flatten(),
            nn.Linear(1024, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim)
            )
        self.num_base = num_sty_feat
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, 1))
    def forward(self,
                base_sty_feat, # [num_base, feat_dim], static
                content_feat,  # [b, feat_dim, h, w]
                message,  # [b, message_len]
                ):
        b, _ = message.shape
        message_feat = self.message_enc(message)  # [b, feat_dim]
        content_feat = self.content_enc(content_feat)
        message_feat = message_feat[:, None, :].repeat(1, self.num_base, 1)
        content_feat = content_feat[:, None, :].repeat(1, self.num_base, 1)
        base_sty_feat = base_sty_feat[None, :, :].repeat(b, 1, 1)

        # message_feat = message_feat / message_feat.norm(dim=-1, keepdim=True)
        content_feat = content_feat / content_feat.norm(dim=-1, keepdim=True)
        base_sty_feat = base_sty_feat / base_sty_feat.norm(dim=-1, keepdim=True)

        feat = base_sty_feat + message_feat + content_feat  # [b, num_base, feat_dim]
        unact_weight = self.mlp(feat)  # [b, num_base, 1]

        # convex combination
        interpo_weight = torch.softmax(unact_weight, dim=1)
        style_feat = torch.sum(interpo_weight * base_sty_feat, dim=1)  # [b, feat_dim]

        return style_feat, interpo_weight

class Encoder(nn.Module):
    def __init__(self,
                 cfg,
                 sty_feat,
                 font_model):

        super(Encoder, self).__init__()
        self.sty_feat = sty_feat
        self.combinator = Combinator(num_sty_feat=cfg.num_sty_feat,
                                     msg_len=cfg.msg_len,
                                     dim=cfg.sty_feat_dim)
        self.font_model = font_model

        # freeze the generator
        for param in self.font_model.parameters():
            param.requires_grad = False

    def forward(self, img, msg):
        content_feat, srcs_skip1, srcs_skip2  = self.font_model.cnt_encoder(img)
        style_feat, interpo_weight = self.combinator(self.sty_feat, content_feat, msg)
        enc_img, _ = self.font_model.decode(content_feat, style_feat, srcs_skip1, srcs_skip2)
        return enc_img, interpo_weight
