import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import os.path as op
import numpy as np
from tqdm import tqdm

import torch
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

from dec import CLIP
from ds import get_dl
import cfg

tensor2numpy = lambda x: x.detach().cpu().numpy()


def load_dec():
    dec = CLIP(cfg)
    dec = dec.to(cfg.device)
    dec_ckpt = torch.load(cfg.dec_ckpt_path)
    dec.load_state_dict(dec_ckpt)
    dec.eval()
    return dec


def load_gt(gt_path):
    with open(gt_path, 'r') as f:
        gts = f.read()
    gts = [int(gt) for gt in gts]
    gts = np.array(gts)
    return gts


def test(dl, dec, gts):
    preds = []
    with torch.no_grad():
        for font_img, path in tqdm(dl):
            font_img = font_img.to(cfg.device)
            logit = dec(font_img)
            pred = torch.argmax(logit, dim=1)
            preds.append(tensor2numpy(pred))

    preds = np.concatenate(preds, axis=0)
    acc = (preds == gts).sum().item() / len(gts)
    print(f'Accuracy: {acc:.4f}')
    return preds, acc


def main(test_dir):
    cfg.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    cfg.data_dir = test_dir
    dl = get_dl(cfg, transform=T.Compose([
        T.Resize((cfg.font_img_size, cfg.font_img_size), interpolation=InterpolationMode.BILINEAR),
        T.Resize((cfg.clip_img_size, cfg.clip_img_size), interpolation=InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]))
    dec = load_dec()
    gts = load_gt(cfg.gt_path)
    pred, acc = test(dl, dec, gts)
    return pred, acc


if __name__ == '__main__':
    for pt in cfg.pt_list:
        print('pt: {}'.format(pt))
        for scenario in cfg.scenario_list:
            print('scenario: {}'.format(scenario))
            main(test_dir=op.join(cfg.root, f'{scenario}/FontGuard_{cfg.font_name}_{pt}'))
