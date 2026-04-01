import cfg
import os
import pprint

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch
torch.set_num_threads(1)

import os
import time
import logging
from collections import defaultdict


import copy
import numpy as np
from collections import OrderedDict, defaultdict
from random import sample

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import RMSprop

from model.DGFont.generator import Generator
from model.DGFont.discriminator import Discriminator, add_sn
from model.noise_layers.noiser import Noiser
from model.encoder import Encoder
from model.discriminator import Discriminator
from model.clip import LASTED
from model.PCGrad.pcgrad import PCGrad_RMSprop

from utils import AverageMeter, get_logger, get_msg_img, get_rand_interop_type, get_clip_pred, log_progress, write_losses, save_checkpoint, convert_msg, convert_img
import os
import os.path as op
import logging
import time
import torchvision
from model.vgg import VGG
from ds import get_dl

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FontGuard:
    def __init__(self):
        super(FontGuard, self).__init__()
        self.wm_bit = cfg.msg_n
        self.b = cfg.bs
        self.clip_img_size = (cfg.clip_img_size, cfg.clip_img_size)
        self.real_label = .9
        self.fake_label = .1
        self.disc_real_gt = torch.full((self.b, 1), self.real_label, device=device, dtype=torch.float32)
        self.disc_fake_gt = torch.full((self.b, 1), self.fake_label, device=device, dtype=torch.float32)
        self.gen_fake_gt = torch.full((self.b, 1), self.real_label, device=device, dtype=torch.float32)
        self.save_img_dir = op.join(cfg.exp_dir, 'vis_img')
        self.logger = get_logger(cfg)

        # load mean style feature
        base_sty_feat = torch.load(cfg.base_sty_path).to(device)
        
        # define font model
        font_model = Generator(img_size=cfg.font_img_size,
                               sty_dim=cfg.sty_feat_dim,
                               use_sn=False,
                               mute=True,
                               baseline_idx=0)
        # load font model
        font_model_cp = torch.load(cfg.font_model_ckpt, map_location='cpu')['G_state_dict']
        font_model_cp = OrderedDict([(k.replace('module.', ''), v) for k, v in font_model_cp.items()])
        font_model.load_state_dict(font_model_cp)  # , strict=False

        # define encoder, decoder, discriminator, noiser
        self.encoder = Encoder(cfg, base_sty_feat, font_model).to(device)
        self.decoder = LASTED(num_cls=cfg.msg_n).to(device)
        self.noiser = Noiser(cfg)
        self.discriminator = Discriminator(cfg).to(device)
        add_sn(self.discriminator)

        # load decoder
        if cfg.pretrain_dec_ckpt is not None:
            pretrain_dec = torch.load(cfg.pretrain_dec_ckpt, map_location='cpu')
            pretrain_dec = OrderedDict([(k.replace('module.', ''), v) for k, v in pretrain_dec.items()])
            del pretrain_dec['fc.weight']
            del pretrain_dec['fc.bias']
            self.decoder.load_state_dict(pretrain_dec, strict=False)

        # define optimizer
        self.optim_enc = PCGrad_RMSprop(self.encoder.parameters(), lr=cfg.enc_lr, alpha=cfg.alpha, weight_decay=cfg.weight_decay)
        self.optim_dec = RMSprop(self.decoder.parameters(), lr=cfg.dec_lr, alpha=cfg.alpha, weight_decay=cfg.weight_decay)
        self.optim_disc = RMSprop(self.discriminator.parameters(), lr=cfg.disc_lr, alpha=cfg.alpha, weight_decay=cfg.weight_decay)

        self.load_ckpt()

        # define loss
        self.vgg = VGG(3, 1, False).to(device)
        self.ce_loss = nn.CrossEntropyLoss().to(device)
        self.bce_loss = nn.BCEWithLogitsLoss().to(device)
        self.mse_loss = nn.MSELoss().to(device)

        # define dataloader
        self.train_dl, self.val_dl = get_dl(cfg)
        self.white_bg = torch.ones((self.b, 3, cfg.font_img_size, cfg.font_img_size))
        self.black_bg = torch.zeros((self.b, 3, cfg.font_img_size, cfg.font_img_size)) - 1
        self.white_bg, self.black_bg = self.white_bg.to(device), self.black_bg.to(device)

        logging.info('Model: {}\n'.format(self.to_str()))
        logging.info('Cfg:\n')
        logging.info(pprint.pformat(vars(cfg)))
        
    def resize_img(self, img):
        return F.interpolate(img, size=self.clip_img_size, mode=get_rand_interop_type())

    def train_one_step(self, ori_img, bg_img, msg, msg_bin, epoch, is_save_img):
        self.encoder.train()
        self.decoder.train()
        self.discriminator.train()

        enc_img, blend_w = self.encoder(ori_img, msg_bin)  # enc_img: [-1(black), 1(white)]

        # train discriminator
        self.optim_disc.zero_grad()
        real_logit = self.discriminator(ori_img)
        fake_logit = self.discriminator(enc_img.detach())
        real_loss = self.bce_loss(real_logit, self.disc_real_gt)
        fake_loss = self.bce_loss(fake_logit, self.disc_fake_gt)
        disc_loss = real_loss + fake_loss
        disc_loss.backward()
        self.optim_disc.step()

        # train encoder and decoder
        self.encoder.zero_grad()
        self.decoder.zero_grad()

        ori_img, enc_img, enc_img_bg = \
            self.binarize_add_bg(ori_img, enc_img, bg_img, cfg.db_thresh, cfg.db_temp)

        noise_img = self.noiser(enc_img_bg.clone(), epoch)

        dec_logits = self.decoder(self.resize_img(enc_img_bg))
        dec_noise_logits = self.decoder(self.resize_img(noise_img))

        fake_logit = self.discriminator(enc_img_bg)
        enc_gan_l = cfg.gan_w * self.bce_loss(fake_logit, self.gen_fake_gt)

        ori_vgg, enc_vgg = self.vgg(ori_img), self.vgg(enc_img)
        enc_img_l = cfg.qlt_w * self.mse_loss(ori_vgg, enc_vgg)

        enc_nce_l = cfg.nce_w * self.get_nce_loss(blend_w, msg)

        dec_msg_l = cfg.msg_w * self.get_clip_loss(dec_logits, msg)
        dec_noise_msg_l = cfg.noise_msg_w * self.get_clip_loss(dec_noise_logits, msg)

        if epoch < cfg.init_epoch:  # stage 1
            dec_msg_l.backward(retain_graph=True)
            self.optim_enc.pc_backward([enc_nce_l, dec_msg_l])

        elif cfg.init_epoch <= epoch < cfg.start_noise_epoch:  # stage 2
            dec_msg_l.backward(retain_graph=True)
            self.optim_enc.pc_backward([enc_img_l, enc_gan_l, dec_msg_l])

        else:  # stage 3
            dec_noise_msg_l.backward(retain_graph=True)
            self.optim_enc.pc_backward([enc_img_l, enc_gan_l, dec_noise_msg_l])

        self.optim_enc.step()
        self.optim_dec.step()

        _, dec_acc = get_clip_pred(dec_logits, msg)
        _, dec_noise_acc = get_clip_pred(dec_noise_logits, msg)

        losses = {
            'discriminator_loss': disc_loss.item(),
            'enc_img_loss': enc_img_l.item(),
            'enc_gan_loss': enc_gan_l.item(),
            'enc_nce_loss': enc_nce_l.item(),
            'dec_msg_loss': dec_msg_l.item(),
            'dec_noise_msg_loss': dec_noise_msg_l.item(),
            'dec_acc': dec_acc,
            'dec_noise_acc': dec_noise_acc,
        }
        
        if is_save_img:
            self._save_img_preview(
                (
                    ori_img, 
                    enc_img.detach(), 
                    enc_img_bg.detach(), 
                    noise_img.detach()
                ),
                msg,
                epoch
            )
        
        return losses

    def get_clip_loss(self, features_logits, labels):
        # image-axis loss
        loss_img = self.ce_loss(features_logits, labels)
        # text-axis loss
        labels = labels.t()
        text_feats = features_logits.t()
        tmp_loss = []
        for tmp_class_idx in range(self.wm_bit):
            cur_tmp_loss = [text_feats[tmp_class_idx][labels == tmp_class_idx].mean().unsqueeze(0)]
            for cur_tmp_inner_idx in range(self.wm_bit):
                if cur_tmp_inner_idx == tmp_class_idx:
                    continue
                cur_tmp_loss.append(text_feats[tmp_class_idx][labels == cur_tmp_inner_idx].mean().unsqueeze(0))
            tmp_loss.append(torch.cat(cur_tmp_loss))
        loss_text = self.ce_loss(torch.stack(tmp_loss),
                                 torch.zeros(self.wm_bit).long().to(labels.device))
        # total loss
        loss = (loss_img + loss_text) / 2 if not torch.isnan(loss_text).any() else loss_img

        return loss

    def get_nce_loss(self, interpo_weight, msg):
        # interpo_weight: [b, num_base]
        # msg: [b,]
        interpo_weight = interpo_weight.squeeze()
        b = msg.size(0)

        distance_matrix = torch.cdist(interpo_weight, interpo_weight, p=2)
        sim = torch.exp(-distance_matrix / 0.1) # temp

        # Mask for positive pairs (1 where labels are same, 0 otherwise)
        p_mask = (msg.view(b, 1) == msg.view(1, b)).float()
        # Set diagonal to 0 as we don't want self-pairs
        p_mask.fill_diagonal_(0)
        p_num = torch.sum(p_mask, dim=-1)
        a_mask = torch.ones(b, b, device=device)
        a_mask.fill_diagonal_(0)
        # # Mask for negative pairs (inverse of positive mask)
        # negative_mask = 1 - p_mask
        # negative_mask.fill_diagonal_(0)

        denominator = torch.sum(sim * a_mask, dim=1, keepdim=True)
        logits = torch.sum(torch.log(sim / denominator) * p_mask, dim=-1)
        logits = logits / p_num

        return torch.sum(-logits)

    def binarize_add_bg(self, ori_img, enc_img, bg_img, thresh, temp):
        _, c, h, w = ori_img.shape
        ori_bg_mask, ori_font_mask = ori_img == 1, ori_img != 1
        enc_bg_mask = torch.sum(enc_img.detach() > thresh, dim=1)
        enc_bg_mask = enc_bg_mask != 0
        enc_bg_mask = enc_bg_mask[:, None, ...].repeat(1, 3, 1, 1)
        enc_font_mask = ~enc_bg_mask

        # differentiable binarization
        enc_font_bin = torch.sigmoid(
            torch.mul(torch.sub(enc_img, alpha=thresh, other=1), temp))  # [0, 1]
        enc_font_bin = 2 * enc_font_bin - 1  # [-1, 1]
        white_bg = torch.ones(enc_font_bin.shape).to(enc_font_bin.device)
        enc_font_bin = enc_font_bin * enc_font_mask + white_bg * enc_bg_mask
        ori_font_bin = torch.sigmoid(
            torch.mul(torch.sub(ori_img, alpha=thresh, other=1), temp))  # [0, 1]
        ori_font_bin = 2 * ori_font_bin - 1  # [-1, 1]
        ori_font_bin = ori_font_bin * ori_font_mask + white_bg * ori_bg_mask

        # inverse color depend on background
        inv_f = torch.mean(bg_img, dim=(1, 2, 3)) > 0
        inv_f = (inv_f * 2 - 1)[:, None, None, None].repeat(1, c, h, w) # .repeat(1, c, h, w)
        ori_inv_img = copy.deepcopy(ori_img)
        ori_inv_img = ori_inv_img * inv_f
        enc_inv_img = enc_font_bin * inv_f
        enc_img_bg = enc_inv_img * enc_font_mask + bg_img * enc_bg_mask

        return ori_font_bin, enc_font_bin, enc_img_bg
    
    def _save_img_preview(self, img, msg, epoch, save_n=8, size=(80,80)):
        
        (ori_img, enc_img, enc_img_bg, noise_img) = img
        
        if msg.ndim == 1:
            msg = msg[:save_n]
        else:
            msg = msg[:save_n, :]
            msg = msg.cpu().numpy().astype(bool)
            msg = np.packbits(msg, axis=1, bitorder='little').reshape(-1)

        msg_img = get_msg_img(msg)

        img_list = [ori_img, enc_img, enc_img_bg, noise_img]
        img_list = [convert_img(each_set, save_n) for each_set in img_list]
        img_list += [msg_img]

        # diff img
        img_list.insert(2, torch.abs(img_list[0] - img_list[1]))

        img_list = [F.interpolate(each_set, size=size) for each_set in img_list]

        stacked_images = torch.cat(img_list, dim=0)
        save_dir = op.join(self.save_img_dir, f"epoch_{epoch}.png")
        torchvision.utils.save_image(stacked_images, save_dir)  # , original_images.shape[0], normalize=False

    def get_rnd_bg(self, bg_img, epoch):
        if epoch > cfg.start_noise_epoch:
            if cfg.full_noise_epoch == 0:
                p_pure_bg = 1 - cfg.max_bg_p
            else:
                p_pure_bg = 1 - (cfg.max_bg_p * min(epoch / cfg.full_noise_epoch, 1))
            pure_bg_idx = sample(range(self.b), round(self.b * p_pure_bg))
            white_idx = sample(pure_bg_idx, len(pure_bg_idx) // 2)
            black_idx = list(set(pure_bg_idx) - set(white_idx))
        else:
            white_idx = sample(range(self.b), self.b // 2)
            black_idx = list(set(range(self.b)) - set(white_idx))
        bg_img[white_idx] = self.white_bg[white_idx]
        bg_img[black_idx] = self.black_bg[black_idx]

        return bg_img

    def get_rnd_msg(self, mode):
        if mode == 'train':
            msg_idx = np.random.choice(list(range(cfg.msg_n)), (self.b, 1))  # [b,]
            msg_bin = np.unpackbits(msg_idx.astype(np.uint8), axis=1)[:, -cfg.msg_bit:]
            msg_idx = torch.Tensor(msg_idx).to(device).view(-1).type(torch.int64)  # [b,]
            msg_bin = torch.Tensor(msg_bin).to(device)  # [b, log2(num_cls)]
        elif mode == 'eval':
            msg_idx = np.array(list(range(cfg.msg_n)), dtype=np.uint8)
            msg_idx = msg_idx[..., np.newaxis]
            msg_bin = np.unpackbits(msg_idx.astype(np.uint8), axis=1)[:, -cfg.msg_bit:]
            msg_idx = torch.Tensor(msg_idx).to(device).view(-1).type(torch.int64)  # [num_cls,]
            msg_bin = torch.Tensor(msg_bin).to(device)  # [num_cls, log2(num_cls)]
            msg_idx = msg_idx.repeat(self.b)  # [b*num_cls,]
            msg_bin = msg_bin.repeat(self.b, 1)  # [b*num_cls, log2(num_cls)]
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        return msg_idx, msg_bin

    def save_img(self, font_img, output_imgs, msg, idx_to_save, epoch, step, folder, resize_to=None):
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
        filename = op.join(folder, f"epoch-{epoch}-{step}.png")
        torchvision.utils.save_image(stacked_images, filename)

    def to_str(self):
        return '{}\n{}\n{}'.format(
            str(self.encoder),
            str(self.decoder),
            str(self.discriminator),
        )

    def load_ckpt(self):
        if cfg.fontguard_ckpt is not None:
            ckpt = torch.load(cfg.fontguard_ckpt, map_location='cpu')
            try:
                self.encoder.load_state_dict(ckpt['encoder'])
                self.decoder.load_state_dict(ckpt['decoder'], strict=False)
                self.discriminator.load_state_dict(ckpt['discriminator'])
                self.optim_enc.load_state_dict(ckpt['optim_enc'])
                self.optim_dec.load_state_dict(ckpt['optim_dec'])
                self.optim_disc.load_state_dict(ckpt['optim_disc'])
            except RuntimeError as err:
                logging.warning(f"Skip incompatible checkpoint load: {err}")

    def train(self):
        logger = self.logger
        run_device = device
        step_per_epoch = len(self.train_dl) if cfg.step_per_epoch == -1 else cfg.step_per_epoch
        epochs = cfg.epochs
        exp_dir = cfg.exp_dir
        print_freq = cfg.print_freq
        loss_record = defaultdict(AverageMeter)
        
        test_acc_best = -1

        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
            step, is_save_img = 1, True

            for font_img, bg_img, _ in self.train_dl:
                font_img, bg_img = font_img.to(run_device), bg_img.to(run_device)
                bg_img = self.get_rnd_bg(bg_img, epoch)
                msg_idx, msg_bin = self.get_rnd_msg("train")

                losses = self.train_one_step(
                    font_img, bg_img, msg_idx, msg_bin, epoch, is_save_img)

                if is_save_img:
                   is_save_img = False

                for name, loss in losses.items():
                    loss_record[name].update(loss)
                    
                if step % print_freq == 0:
                    logging.info("Epoch: {}/{} Step: {}/{}".format(epoch, epochs, step, step_per_epoch)
                    )
                    log_progress(loss_record)

                step += 1
                if step > step_per_epoch:
                    break

            write_losses(
                op.join(exp_dir, "train.csv"),
                loss_record,
                epoch,
                time.time() - epoch_start,
            )

            logger.save_losses(loss_record, epoch)

            avg_acc = loss_record["dec_noise_acc"].avg
            if epoch % cfg.save_cp_freq == 0:
                if avg_acc > test_acc_best:
                    test_acc_best = avg_acc
                acc_str = str(round(avg_acc, 4))
                save_cp_dir = op.join(exp_dir, "checkpoints")
                save_checkpoint(self, cfg.exp_name, epoch, save_cp_dir, acc_str)

            write_losses(
                op.join(exp_dir, "eval.csv"),
                loss_record,
                epoch,
                time.time() - epoch_start,
            )



if __name__ == "__main__":
    model = FontGuard()
    model.train()
