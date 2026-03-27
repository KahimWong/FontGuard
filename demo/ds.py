from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


class Ds(ImageFolder):
    def __init__(self, data_dir, transform):
        super(Ds, self).__init__(data_dir, transform)

    def __getitem__(self, index):
        path, _ = self.samples[index]
        font_img = self.loader(path)
        if self.transform is not None:
            font_img = self.transform(font_img)
        return font_img, path


def get_dl(cfg, transform=None):
    val_ds = Ds(cfg.data_dir, transform)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    return val_dl
