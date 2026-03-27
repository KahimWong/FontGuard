import random
from glob import glob

from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


class FontImgDs(ImageFolder):
    def __init__(self, font_dir, bg_dir, font_trans, bg_trans):
        super(FontImgDs, self).__init__(font_dir, font_trans)
        self.bg_transform = bg_trans
        self.bg_img_list = glob(bg_dir + "/*.jpg")

    def __getitem__(self, index):
        font_path, _ = self.samples[index]
        font_img = self.loader(font_path)
        font_img = self.transform(font_img)
        bg_path = random.choice(self.bg_img_list)
        bg_img = self.loader(bg_path)
        bg_img = self.bg_transform(bg_img)
        return font_img, bg_img, font_path


def get_dl(cfg):
    font_dir = cfg.font_dir
    bg_dir = cfg.bg_dir
    img_size = cfg.font_img_size

    font_trans = T.Compose(
        [T.ToTensor(), T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
    )

    bg_trans = T.Compose(
        [
            T.ToTensor(),
            T.RandomResizedCrop((img_size, img_size), scale=(0.005, 0.05)),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    train_ds = FontImgDs(font_dir, bg_dir, font_trans, bg_trans)
    train_dl = DataLoader(
        train_ds, batch_size=cfg.bs, shuffle=True, num_workers=cfg.workers
    )

    val_ds = FontImgDs(font_dir, bg_dir, font_trans, bg_trans)
    val_dl = DataLoader(
        val_ds,
        batch_size=cfg.bs,
        shuffle=False,
        num_workers=cfg.workers,
        drop_last=True,
    )

    return train_dl, val_dl
