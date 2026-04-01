import os
import os.path as op
import sys
sys.path.append(op.dirname(op.dirname(op.abspath(__file__))))
import random
from random import sample
import numpy as np
import copy
from collections import OrderedDict
from tensorboard_logger import Logger

import logging
import pprint

import torch
from torchmetrics.functional.pairwise import pairwise_cosine_similarity
import torchvision


interpo_type = ['bilinear']  # 'bilinear' , 'nearest', 'area'
tensor2numpy = lambda x: x.detach().cpu().numpy()

def get_rand_interop_type():
    return sample(interpo_type, 1)[0]

def get_acc(batch_size, dec_msg, msg):
    preds = tensor2numpy(dec_msg).round().clip(0, 1)
    gts = tensor2numpy(msg)
    acc = np.sum(np.all(preds == gts, axis=1)) / batch_size
    return acc

def get_clip_pred(dec_logit, msg):
    preds = torch.argmax(dec_logit, dim=1)
    acc = torch.sum(preds == msg) / msg.shape[0]
    return tensor2numpy(preds), float(tensor2numpy(acc))

def get_LASTED_pred(anchors, feats):
    cos_sim = pairwise_cosine_similarity(anchors, feats)
    preds = torch.argmax(cos_sim.T, dim=1)
    return tensor2numpy(preds)

def get_avg_bit_err(batch_size, dec_msg, msg):
    decoded_rounded = tensor2numpy(dec_msg).round().clip(0, 1)
    bitwise_avg_err = np.sum(np.abs(decoded_rounded - tensor2numpy(msg))) / (
            batch_size * msg.shape[1])
    return bitwise_avg_err


def add_bg(imgs, bin_method, bin_thresh, sigmoid_k):
    # tmp_check_dir = '/root/autodl-tmp/FontCode/output/tmp_check'
    #
    # convert_img = lambda img, idx_to_save: (img[:idx_to_save, :, :, :].cpu() + 1) / 2
    # save_img = lambda img, idx_to_save, name: torchvision.utils.save_image(convert_img(img, idx_to_save), os.path.join(tmp_check_dir, name + '.jpg'))

    (ori_font_img, enc_font_img, bg_img) = imgs

    # save_img(ori_font_img, 1, 'ori_font_img')
    # save_img(enc_font_img, 1, 'enc_font_img')
    # save_img(bg_img, 1, 'bg_img')

    b, c, h, w = ori_font_img.shape
    ori_bg_mask, ori_font_mask = ori_font_img == 1, ori_font_img != 1
    enc_bg_mask = torch.sum(enc_font_img.detach() > bin_thresh, dim=1)
    enc_bg_mask = enc_bg_mask != 0
    enc_bg_mask = enc_bg_mask[:, None, ...].repeat(1, 3, 1, 1)
    enc_font_mask = ~enc_bg_mask

    # save_img(enc_bg_mask, 1, 'enc_bg_mask')
    # save_img(enc_font_mask, 1, 'enc_font_mask')

    if bin_method == 'DB':  # differentiable binarization
        enc_font_bin = torch.sigmoid(
            torch.mul(torch.sub(enc_font_img, alpha=bin_thresh, other=1), sigmoid_k))  # [0, 1]
        enc_font_bin = 2 * enc_font_bin - 1  # [-1, 1]
        white_bg = torch.ones(enc_font_bin.shape).to(enc_font_bin.device)
        enc_font_bin = enc_font_bin * enc_font_mask + white_bg * enc_bg_mask
        ori_font_bin = torch.sigmoid(
            torch.mul(torch.sub(ori_font_img, alpha=bin_thresh, other=1), sigmoid_k))  # [0, 1]
        ori_font_bin = 2 * ori_font_bin - 1  # [-1, 1]
        ori_font_bin = ori_font_bin * ori_font_mask + white_bg * ori_bg_mask

    elif bin_method == 'binary':  # binarization
        # bg_img = torch.ones(bg_img.shape).to(bg_img.device)
        enc_font_bin = enc_font_img.detach().clone()
        enc_font_bin[enc_bg_mask] = 1
        enc_font_bin[~enc_bg_mask] = -1

    elif bin_method == 'none':
        ori_font_bin = ori_font_img
        enc_font_bin = enc_font_img
        pass

    # check_binary_method(bg_img, enc_bg_mask, enc_font_img, enc_font_mask, save_img)

    # inverse color depend on background
    inv_f = torch.mean(bg_img, dim=(1, 2, 3)) > 0
    inv_f = (inv_f * 2 - 1)[:, None, None, None].repeat(1, c, h, w) # .repeat(1, c, h, w)
    ori_inv_img = copy.deepcopy(ori_font_img)
    ori_inv_img = ori_inv_img * inv_f
    # enc_inv_img = enc_font_img.clone()  # .detach(), .clone()
    enc_inv_img = enc_font_bin * inv_f
    ori_img_w_bg = ori_inv_img * ori_font_mask + bg_img * ori_bg_mask  # add bg
    enc_img_w_bg = enc_inv_img * enc_font_mask + bg_img * enc_bg_mask

    # save_img(ori_img_w_bg, 1, 'ori_img_w_bg')
    # save_img(enc_img_w_bg, 1, 'enc_img_w_bg')
    # save_img(enc_font_img, 1, 'enc_font_img_2')

    return ori_font_bin, enc_font_bin, enc_img_w_bg, ori_img_w_bg


def check_binary_method(bg_img, enc_bg_mask, enc_font_img, enc_font_mask, save_img):
    # pure white bg
    white_bg = torch.ones(bg_img.shape).to(bg_img.device)
    # binary
    binary_img = enc_font_img.clone()
    binary_img[enc_bg_mask] = 1
    binary_img[~enc_bg_mask] = -1
    # none
    none_img = enc_font_img.clone()
    # binary w bg
    binary_img = binary_img * enc_font_mask + white_bg * enc_bg_mask
    # none w bg
    none_img = none_img * enc_font_mask + white_bg * enc_bg_mask
    # diff
    diff = torch.abs(binary_img - none_img) * 10
    save_img(binary_img, 4, 'binary_img')
    save_img(none_img, 4, 'none_img')
    save_img(diff, 4, 'diff')


def get_rnd_bg(bg_img, pure_bg, cfg, epoch):
    b = bg_img.shape[0]
    white_bg, black_bg = pure_bg
    if epoch > cfg.start_noise_epoch:
        if cfg.full_noise_epoch == 0:
            p_pure_bg = 1 - cfg.max_bg_
        else:
            p_pure_bg = 1 - (cfg.max_bg_p * min(epoch / cfg.full_noise_epoch, 1))
        pure_bg_idx = random.sample(range(b), round(b*p_pure_bg))
        white_bg_idx = random.sample(pure_bg_idx, len(pure_bg_idx)//2)
        black_bg_idx = list(set(pure_bg_idx) - set(white_bg_idx))
    else:
        white_bg_idx = random.sample(range(b), b // 2)
        black_bg_idx = list(set(range(b)) - set(white_bg_idx))
    bg_img[white_bg_idx] = white_bg[white_bg_idx]
    bg_img[black_bg_idx] = black_bg[black_bg_idx]

    return bg_img


def get_pure_bg(cfg):
    b = cfg.batch_size
    img_size = cfg.font_img_size
    white_bg = torch.ones((b, 3, img_size, img_size))
    black_bg = torch.zeros((b, 3, img_size, img_size)) - 1
    return (white_bg.to(cfg.device), black_bg.to(cfg.device))


def get_rnd_msg(batch_size, cfg, mode):
    if mode == 'train':
        msg = np.random.choice(list(range(cfg.num_cls)), (batch_size, 1))  # [b,]
        msg_bin = np.unpackbits(msg.astype(np.uint8), axis=1)[:, -cfg.msg_len:]
        msg = torch.Tensor(msg).to(cfg.device).view(-1).type(torch.int64)  # [b,]
        msg_bin = torch.Tensor(msg_bin).to(cfg.device)  # [b, log2(num_cls)]
    elif mode == 'eval':
        msg = np.array(list(range(cfg.num_cls)), dtype=np.uint8)
        msg = msg[..., np.newaxis]
        msg_bin = np.unpackbits(msg.astype(np.uint8), axis=1)[:, -cfg.msg_len:]
        msg = torch.Tensor(msg).to(cfg.device).view(-1).type(torch.int64)  # [num_cls,]
        msg_bin = torch.Tensor(msg_bin).to(cfg.device)  # [num_cls, log2(num_cls)]
        msg = msg.repeat(batch_size)  # [b*num_cls,]
        msg_bin = msg_bin.repeat(batch_size, 1)  # [b*num_cls, log2(num_cls)]

    return msg, msg_bin


def get_noise_layer(*args):
    return [layer for layer in args]


def get_logger(cfg):
    logging.basicConfig(level=logging.INFO,
                        format='%(message)s',
                        handlers=[
                            logging.FileHandler(os.path.join(cfg.exp_dir, f'{cfg.exp_name}.log')),
                            logging.StreamHandler(sys.stdout)
                        ])
    logging.info('Tensorboard is enabled. Creating logger.')
    tb_logger = Logger(os.path.join(cfg.exp_dir, 'tb-logs'))
    return TensorboardLoggerAdapter(tb_logger)


class TensorboardLoggerAdapter:
    def __init__(self, logger):
        self.logger = logger

    def save_losses(self, losses, epoch):
        for name, meter in losses.items():
            value = meter.avg if hasattr(meter, 'avg') else meter
            self.logger.log_value(name, float(value), int(epoch))

    def save_grads(self, epoch):
        return None

    def save_tensors(self, epoch):
        return None


def print_model(model, cfg):
    logging.info('Model: {}\n'.format(model.to_stirng()))
    logging.info('Cfg:\n')
    logging.info(pprint.pformat(vars(cfg)))
    
    
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if val != np.nan and val != np.inf:
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count

import numpy as np
import os
import os.path as op
import re
import csv
import cv2
import time
import pickle
import logging
import random
import shutil
from glob import glob
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils import data

import torchvision.utils
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder


def image_to_tensor(image):
    """
    Transforms a numpy-image into torch tensor
    :param image: (batch_size x height x width x channels) uint8 array
    :return: (batch_size x channels x height x width) torch tensor in range [-1.0, 1.0]
    """
    image_tensor = torch.Tensor(image)
    image_tensor.unsqueeze_(0)
    image_tensor = image_tensor.permute(0, 3, 1, 2)
    image_tensor = image_tensor / 127.5 - 1
    return image_tensor


def tensor_to_image(tensor):
    """
    Transforms a torch tensor into numpy uint8 array (image)
    :param tensor: (batch_size x channels x height x width) torch tensor in range [-1.0, 1.0]
    :return: (batch_size x height x width x channels) uint8 array
    """
    image = tensor.permute(0, 2, 3, 1).cpu().numpy()
    image = (image + 1) * 127.5
    return np.clip(image, 0, 255).astype(np.uint8)

# to cpu, scale values to range [0, 1] from original range of [-1, 1]
convert_img = lambda img, idx_to_save: (img[:idx_to_save, :, :, :].cpu() + 1) / 2

def save_images(font_img, output_imgs, msg, idx_to_save, epoch, step, folder, resize_to):
    (enc_img, enc_img_w_bg, noise_img) = output_imgs

    gts = msg
    if len(msg.shape) != 1:
        gts = convert_msg(idx_to_save, msg)

    gt_img = get_msg_img(gts[:idx_to_save])

    img_set_list = [font_img, enc_img, enc_img_w_bg, noise_img]
    img_set_list = [convert_img(each_set, idx_to_save) for each_set in img_set_list]
    img_set_list += [gt_img]

    # diff
    img_set_list.insert(2, torch.abs(img_set_list[0] - img_set_list[1]))

    if resize_to is not None:
        img_set_list = [F.interpolate(each_set, size=resize_to) for each_set in img_set_list]

    stacked_images = torch.cat(img_set_list, dim=0)
    filename = os.path.join(folder, f"epoch-{epoch}-{step}.png")
    torchvision.utils.save_image(stacked_images, filename)  # , original_images.shape[0], normalize=False


def convert_msg(idx_to_save, msg):
    msg = msg[:idx_to_save, :]
    msg = msg.cpu().numpy().astype(bool)
    msg = np.packbits(msg, axis=1, bitorder='little').reshape(-1)
    return msg

transform = transforms.ToTensor()

def get_msg_img(msg):
    msg_img = []
    if torch.is_tensor(msg):
        msg_values = msg.detach().cpu().numpy()
    else:
        msg_values = np.asarray(msg)
    for each_gt in msg_values:
        img = np.zeros((80, 80, 3), dtype=np.uint8)
        img = cv2.putText(img, str(each_gt), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA, False)
        img = transform(img)
        msg_img.append(img)
    return torch.stack(msg_img, dim=0)


def sorted_nicely(l):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def last_checkpoint_from_folder(folder: str):
    last_file = sorted_nicely(os.listdir(folder))[-1]
    last_file = os.path.join(folder, last_file)
    return last_file


def save_checkpoint(model, experiment_name: str, epoch: int, checkpoint_folder: str, acc_str: str):
    # shutil.rmtree(checkpoint_folder)

    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)

    checkpoint_filename = f'{experiment_name}--epoch-{epoch}--acc-{acc_str}.pyt'
    checkpoint_filename = os.path.join(checkpoint_folder, checkpoint_filename)
    logging.info('Saving checkpoint to {}'.format(checkpoint_filename))
    checkpoint = {
        'encoder': model.encoder.state_dict(),
        'decoder': model.decoder.state_dict(),
        'discriminator': model.discriminator.state_dict(),
        'optim_enc': model.optim_enc.state_dict(),
        'optim_dec': model.optim_dec.state_dict(),
        'optim_disc': model.optim_disc.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, checkpoint_filename)
    logging.info('Saving checkpoint done.')


# def load_checkpoint(hidden_net: Hidden, options: Options, this_run_folder: str):
def load_last_checkpoint(checkpoint_folder):
    """ Load the last checkpoint from the given folder """
    last_checkpoint_file = last_checkpoint_from_folder(checkpoint_folder)
    checkpoint = torch.load(last_checkpoint_file)

    return checkpoint, last_checkpoint_file


def model_from_checkpoint(model, checkpoint):
    model.encoder.load_state_dict(checkpoint['encoder'])
    model.decoder.load_state_dict(checkpoint['decoder'], strict=False)  # TODO
    model.discriminator.load_state_dict(checkpoint['discriminator'])
    model.optim_enc.load_state_dict(checkpoint['optim_enc'])
    # model.optim_dec.load_state_dict(checkpoint['optim_dec'])
    model.optim_disc.load_state_dict(checkpoint['optim_disc'])


def load_options(options_file_name):
    """ Loads the training, model, and noise configurations from the given folder """
    with open(os.path.join(options_file_name), 'rb') as f:
        train_options = pickle.load(f)
        noise_config = pickle.load(f)
        hidden_config = pickle.load(f)
        # for backward-capability. Some models were trained and saved before .enable_fp16 was added
        if not hasattr(hidden_config, 'enable_fp16'):
            setattr(hidden_config, 'enable_fp16', False)

    return train_options, hidden_config, noise_config

def get_enc_img_dl(cfg):
    # val dl only
    enc_img_dir = cfg.enc_img_dir
    bg_dir = cfg.bg_dir
    img_size = cfg.font_img_size

    font_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    bg_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomResizedCrop((img_size, img_size), scale=(0.005, 0.05)),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    val_ds = EncImgDs(enc_img_dir, bg_dir, font_transform, bg_transform, img_size)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=4)

    return val_loader


def get_dl(cfg):
    font_dir = cfg.font_dir
    bg_dir = cfg.bg_dir
    img_size = cfg.font_img_size

    font_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    bg_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomResizedCrop((img_size, img_size), scale=(0.005, 0.05)),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    train_ds = FontImgDs(font_dir, bg_dir, font_transform, bg_transform, img_size, cfg.font_type, 'train')
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=4)

    val_ds = FontImgDs(font_dir, bg_dir, font_transform, bg_transform, img_size, cfg.font_type, 'val')
    validation_loader = torch.utils.data.DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=4, drop_last=True)

    return train_loader, validation_loader


def get_dl_V3(cfg):
    img_size = cfg.font_img_size
    font_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    bg_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomResizedCrop((img_size, img_size), scale=(0.005, 0.05)),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    train_ds = DSV3(cfg, font_transform, bg_transform)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader


class DSV3(ImageFolder):
    def __init__(self, cfg, font_transform, bg_transform):
        super(DSV3, self).__init__(cfg.font_dir, font_transform)
        # self.cnt_base_font_dir = cfg.cnt_base_font_dir
        self.bg_transform = bg_transform
        self.bg_img_list = glob(cfg.bg_dir + '/*.jpg')
        self.img_size = cfg.font_img_size
        # self.base_cnt_idx = cfg.base_cnt_idx


    def __getitem__(self, index):
        path, _ = self.samples[index]
        font_img_name = op.basename(path)
        font_id = int(op.dirname(path).split('/')[-1].split('_')[-1])

        font_img = self.loader(path)
        font_img = self.transform(font_img)

        bg_path = random.choice(self.bg_img_list)
        bg_img = self.loader(bg_path)
        bg_img = self.bg_transform(bg_img)

        # base_cnt_imgs = []
        # for each_base in self.base_cnt_idx:
        #     img_path = op.join(self.cnt_base_font_dir, f'id_{each_base}', font_img_name)
        #     base_cnt_imgs.append(self.transform(self.loader(img_path)))
        # base_cnt_imgs = torch.stack(base_cnt_imgs)

        return font_img, bg_img, font_id, path, # base_cnt_imgs


class FontImgDs(ImageFolder):
    def __init__(self, font_root, bg_root, font_transform, bg_transform, img_size, font_type, mode):
        super(FontImgDs, self).__init__(font_root, font_transform)
        self.bg_transform = bg_transform
        self.bg_img_list = glob(bg_root + '/*.jpg')
        self.img_size = img_size
        self.font_type = font_type  # 'ch' or 'eng'
    #     if self.font_type == 'eng' and mode == 'train':
    #         # custom dataset length
    #         self.len = 12000
    #
    #
    # def __len__(self):
    #     return  self.len

    def __getitem__(self, index):
        # path, _ = self.samples[index%len(self.samples)]
        path, _ = self.samples[index]
        font_img = self.loader(path)
        if self.transform is not None:
            font_img = self.transform(font_img)
        # font_id = int(op.dirname(path).split('/')[-1].split('_')[-1])
        if self.bg_transform is not None:
            bg_path = random.choice(self.bg_img_list)
            bg_img = self.loader(bg_path)
            bg_img = self.bg_transform(bg_img)
        return font_img, bg_img, path


class EncImgDs(ImageFolder):
    def __init__(self, font_root, bg_root, font_transform, bg_transform, img_size):
        super(EncImgDs, self).__init__(font_root, font_transform)
        self.bg_transform = bg_transform
        self.bg_img_list = glob(bg_root + '/*.jpg')
        self.img_size = img_size

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.bg_transform is not None:
            bg_path = random.choice(self.bg_img_list)
            bg_img = self.loader(bg_path)
            bg_img = self.bg_transform(bg_img)
        return sample, bg_img, target, path


def log_progress(losses_accu):
    log_print_helper(losses_accu, logging.info)


def print_progress(losses_accu):
    log_print_helper(losses_accu, print)


def log_print_helper(losses_accu, log_or_print_func):
    max_len = max([len(loss_name) for loss_name in losses_accu])
    for loss_name, loss_value in losses_accu.items():
        log_or_print_func(loss_name.ljust(max_len + 4) + '{:.4f}'.format(loss_value.avg))


def create_output_dir(output_dir, exp_name):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    run_dir = os.path.join(output_dir, f'{exp_name} {time.strftime("%Y.%m.%d--%H-%M-%S")}')

    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(run_dir, 'images'), exist_ok=True)

    return run_dir


def write_losses(file_name, losses_accu, epoch, duration=0):
    with open(file_name, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if epoch == 1:
            row_to_write = ['epoch'] + [loss_name.strip() for loss_name in losses_accu.keys()] + ['duration']
            writer.writerow(row_to_write)
        row_to_write = [epoch] + ['{:.4f}'.format(loss_avg.avg) for loss_avg in losses_accu.values()] + [
            '{:.0f}'.format(duration)]
        writer.writerow(row_to_write)
