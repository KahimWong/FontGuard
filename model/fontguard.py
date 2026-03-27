import copy
import numpy as np
from collections import OrderedDict, defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import RMSprop

from model.DGFont.generator import Generator
from model.DGFont.discriminator import Discriminator
from model.noise_layers.noiser import Noiser
from model.encoder import Encoder
# from model.encoderV2 import Encoder
from model.discriminator import Discriminator
from model.clip import LASTED
from model.PCGrad.pcgrad import PCGrad_RMSprop

from utils import *
import os
import os.path as op
import logging
import time
from model.vgg import VGG


def add_sn(m):
    for name, layer in m.named_children():
        m.add_module(name, add_sn(layer))
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        return nn.utils.spectral_norm(m)
    else:
        return m


class FontGuard:
    def __init__(self, cfg, logger):
        super(FontGuard, self).__init__()
        device = cfg.device
        
        # load mean style feature
        sty_feat = torch.load(cfg.base_sty_path)  
        
        # define font model
        font_model = Generator(img_size=cfg.font_img_size,
                               sty_dim=cfg.sty_feat_dim,
                               use_sn=False,
                               mute=True,
                               baseline_idx=0)
        # load font model
        font_model_cp = torch.load(cfg.font_model_cp, map_location='cpu')['G_state_dict']
        font_model_cp = OrderedDict([(k.replace('module.', ''), v) for k, v in font_model_cp.items()])
        font_model.load_state_dict(font_model_cp)  # , strict=False

        # build encoder, decoder, discriminator, noiser
        self.encoder = Encoder(cfg, sty_feat.to(device), font_model).to(device)
        self.decoder = LASTED(num_cls=cfg.num_cls).to(device)
        self.noiser = Noiser(cfg)
        self.discriminator = Discriminator(cfg).to(device)
        add_sn(self.discriminator)

        # load pretrain decoder
        if cfg.pretrain_dec_ckpt is not None:
            pretrain_dec = torch.load(cfg.pretrain_dec_ckpt)
            pretrain_dec = OrderedDict([(k.replace('module.', ''), v) for k, v in pretrain_dec.items()])
            del pretrain_dec['fc.weight']
            del pretrain_dec['fc.bias']
            self.decoder.load_state_dict(pretrain_dec, strict=False)

        # build optimizer
        self.optim_enc = PCGrad_RMSprop(self.encoder.parameters(), lr=cfg.enc_lr, alpha=cfg.alpha, weight_decay=cfg.weight_decay)
        self.optim_dec = RMSprop(self.decoder.parameters(), lr=cfg.dec_lr, alpha=cfg.alpha, weight_decay=cfg.weight_decay)
        self.optim_disc = RMSprop(self.discriminator.parameters(), lr=cfg.disc_lr, alpha=cfg.alpha, weight_decay=cfg.weight_decay)

        self.cfg = cfg
        self.device = device
        self.num_cls = cfg.num_cls
        self.clip_in_size = (cfg.clip_input_size, cfg.clip_input_size)

        self.vgg_loss = VGG(3, 1, False)
        self.vgg_loss.to(device)
        self.ce_loss = nn.CrossEntropyLoss().to(device)
        self.bce_loss_w_logit = nn.BCEWithLogitsLoss().to(device)
        self.mse_loss = nn.MSELoss().to(device)

        self.real_label = .9
        self.fake_label = .1

        self.logger = logger

        if cfg.load_cp:
            self.load_checkpoint()
        
    def resize_img(self, img):
        return F.interpolate(img, size=self.clip_in_size, mode=get_rand_interop_type())

    def train_on_batch(self, batch, epoch):
        font_img, bg_img, font_id, msg, msg_bin = batch

        batch_size = font_img.shape[0]
        self.encoder.train()
        self.decoder.train()
        self.discriminator.train()
        with torch.enable_grad():
            enc_img, interpo_weight = self.encoder(font_img, msg_bin)  # enc_img: [-1(black), 1(white)]
            # enc_img, interpo_weight = self.encoder(font_img, font_id, msg)  # enc_img: [-1(black), 1(white)]

            font_img, enc_img, enc_img_w_bg, ori_img_w_bg = \
                add_bg((font_img, enc_img, bg_img), 'DB', self.cfg.bg_thresh, self.cfg.sigmoid_k)

            # <editor-fold desc="train discriminator">
            self.optim_disc.zero_grad()

            d_real_gt = torch.full((batch_size, 1), self.real_label, device=self.device, dtype=torch.float32)
            d_fake_gt = torch.full((batch_size, 1), self.fake_label, device=self.device, dtype=torch.float32)

            d_real_logit = self.discriminator(font_img)
            d_fake_logit = self.discriminator(enc_img.detach())
            d_real_l = self.bce_loss_w_logit(d_real_logit, d_real_gt)
            d_fake_l = self.bce_loss_w_logit(d_fake_logit, d_fake_gt)
            d_l = d_real_l + d_fake_l
            d_l.backward()
            self.optim_disc.step()
            # </editor-fold>

            # <editor-fold desc="train encoder, decoder">
            self.encoder.zero_grad()
            self.decoder.zero_grad()

            g_fake_gt = torch.full((batch_size, 1), self.real_label, device=self.device, dtype=torch.float32)

            noise_img = self.noiser(enc_img_w_bg.clone(), epoch)

            _, dec_logit = self.decoder(self.resize_img(enc_img_w_bg))
            _, dec_logit_no = self.decoder(self.resize_img(noise_img))

            fake_logit = self.discriminator(enc_img_w_bg)
            enc_gan_l = self.cfg.enc_gan_l_w * self.bce_loss_w_logit(fake_logit, g_fake_gt)

            img_vgg = self.vgg_loss(font_img)
            enc_img_vgg = self.vgg_loss(enc_img)
            enc_img_l = self.cfg.enc_img_l_w * self.mse_loss(img_vgg, enc_img_vgg)

            enc_nce_l = self.cfg.enc_nce_l_w * self.get_enc_nce_loss(interpo_weight, msg)

            dec_msg_l = self.cfg.dec_msg_l_w * self.get_clip_loss(dec_logit, msg)
            dec_noise_msg_l = self.cfg.dec_noise_msg_l_w * self.get_clip_loss(dec_logit_no, msg)

            if epoch < self.cfg.max_init_epoch:  # stage 1
                l = enc_nce_l + dec_msg_l
                l.backward()

                # dec_msg_l.backward(retain_graph=True)
                # self.optim_enc.pc_backward([enc_nce_l, dec_msg_l])

            elif self.cfg.max_init_epoch <= epoch < self.cfg.start_noise_epoch:  # stage 2
                l = enc_img_l + enc_gan_l + dec_msg_l
                l.backward()

                # dec_msg_l.backward(retain_graph=True)
                # self.optim_enc.pc_backward([enc_img_l, enc_gan_l, dec_msg_l])

            else:  # stage 3
                dec_noise_msg_l.backward(retain_graph=True)
                self.optim_enc.pc_backward([enc_img_l, enc_gan_l, dec_noise_msg_l])

            self.optim_enc.step()
            self.optim_dec.step()
            # </editor-fold>

        _, dec_acc = get_clip_pred(dec_logit, msg)
        _, dec_noise_acc = get_clip_pred(dec_logit_no, msg)

        losses = {
            'discriminator_loss': d_l.item(),
            'enc_img_loss': enc_img_l.item(),
            'enc_gan_loss': enc_gan_l.item(),
            'enc_nce_loss': enc_nce_l.item(),
            'dec_msg_loss': dec_msg_l.item(),
            'dec_noise_msg_loss': dec_noise_msg_l.item(),
            'dec_acc': dec_acc,
            'dec_noise_acc': dec_noise_acc,
        }
        return losses, (enc_img.detach(), enc_img_w_bg.detach(), noise_img.detach()), msg

    def get_clip_loss(self, features_logits, labels):
        # image-axis loss
        loss_img = self.ce_loss(features_logits, labels)
        # text-axis loss
        labels = labels.t()
        text_feats = features_logits.t()
        tmp_loss = []
        for tmp_class_idx in range(self.num_cls):
            cur_tmp_loss = [text_feats[tmp_class_idx][labels == tmp_class_idx].mean().unsqueeze(0)]
            for cur_tmp_inner_idx in range(self.num_cls):
                if cur_tmp_inner_idx == tmp_class_idx:
                    continue
                cur_tmp_loss.append(text_feats[tmp_class_idx][labels == cur_tmp_inner_idx].mean().unsqueeze(0))
            tmp_loss.append(torch.cat(cur_tmp_loss))
        loss_text = self.ce_loss(torch.stack(tmp_loss),
                                 torch.zeros(self.num_cls).long().to(labels.device))
        # total loss
        loss = (loss_img + loss_text) / 2 if not torch.isnan(loss_text).any() else loss_img

        return loss

    def get_enc_nce_loss(self, interpo_weight, msg):
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
        a_mask = torch.ones(b, b, device=self.device)
        a_mask.fill_diagonal_(0)
        # # Mask for negative pairs (inverse of positive mask)
        # negative_mask = 1 - p_mask
        # negative_mask.fill_diagonal_(0)

        denominator = torch.sum(sim * a_mask, dim=1, keepdim=True)
        logits = torch.sum(torch.log(sim / denominator) * p_mask, dim=-1)
        logits = logits / p_num

        return torch.sum(-logits)

    def validate_on_batch(self, batch, epoch):

        font_img, bg_img, _, msg, msg_bin = batch

        self.encoder.eval()
        self.decoder.eval()
        with torch.no_grad():
            enc_img = self.encoder(font_img, msg_bin)

            enc_img, enc_img_w_bg, ori_img_w_bg = add_bg((font_img, enc_img, bg_img), 'binary', self.cfg.bg_thresh, -1)

            noise_img = self.noiser(enc_img_w_bg, epoch)
            _, dec_logit = self.decoder(self.resize_img(enc_img_w_bg))
            _, dec_noise_logit = self.decoder(self.resize_img(noise_img))

            img_vgg = self.vgg_loss(font_img)
            enc_img_vgg = self.vgg_loss(enc_img)
            enc_img_l = self.cfg.enc_img_l_w * self.mse_loss(img_vgg, enc_img_vgg)

            dec_msg_l = self.cfg.dec_msg_l_w * self.get_clip_loss(dec_logit, msg)
            dec_noise_msg_l = self.cfg.dec_noise_msg_l_w * self.get_clip_loss(dec_noise_logit, msg)

        _, dec_acc = get_clip_pred(dec_logit, msg)
        _, dec_noise_acc = get_clip_pred(dec_noise_logit, msg)

        losses = {
            'enc_img_loss': enc_img_l.item(),
            'dec_msg_loss': dec_msg_l.item(),
            'dec_noise_msg_loss': dec_noise_msg_l.item(),
            'dec_acc': dec_acc,
            'dec_noise_acc': dec_noise_acc}
        return losses, (enc_img, enc_img_w_bg, noise_img), msg

    def to_stirng(self):
        return '{}\n{}'.format(str(self.encoder),
                               str(self.decoder),
                               str(self.discriminator))

    def load_checkpoint(self, checkpoint_base_path=None):
        """Load the last checkpoint from the given folder into this FontGuard instance.

        If `checkpoint_base_path` is None, uses `self.cfg.load_cp_path/checkpoints`.
        Sets `self.cfg.cp_epoch` to the checkpoint epoch.
        """
        if checkpoint_base_path is None:
            checkpoint_base_path = os.path.join(self.cfg.load_cp_path, "checkpoints")
        checkpoint, cp_name = load_last_checkpoint(checkpoint_base_path)
        # preserve old behavior of parsing epoch from filename
        try:
            self.cfg.cp_epoch = int(op.basename(cp_name).split("epoch")[-1].split("-")[1])
        except Exception:
            self.cfg.cp_epoch = checkpoint.get('epoch', 0)
        model_from_checkpoint(self, checkpoint)

    def train(self, cfg=None, logger=None):
        """Train loop moved inside FontGuard. If `cfg` or `logger` are None,
        use the ones stored on the instance.
        """
        if cfg is None:
            cfg = self.cfg
        if logger is None:
            logger = self.logger

        device = cfg.device

        train_dl, _ = get_dl(cfg)

        steps = cfg.step_per_epoch
        if steps == -1:
            steps = len(train_dl)

        epochs = cfg.epochs
        run_dir = cfg.run_dir

        print_each = cfg.print_freq
        idx_to_save = 8
        saved_images_size = (80, 80)
        test_acc_best = -1

        pure_bg = get_pure_bg(cfg)

        os.makedirs(os.path.join(cfg.run_dir, "train_images"), exist_ok=True)

        start_epoch = 1
        if cfg.load_cp:
            start_epoch = cfg.cp_epoch

        for epoch in range(start_epoch, epochs + 1):
            logging.info(
                "Batch size = {}\nSteps in epoch = {}".format(cfg.batch_size, steps)
            )
            training_losses = defaultdict(AverageMeter)

            epoch_start = time.time()
            step = 1
            first_iter = True

            for font_img, bg_img, _ in train_dl:
                b = font_img.shape[0]
                font_img, bg_img = font_img.to(device), bg_img.to(device)
                bg_img = get_rnd_bg(bg_img, pure_bg, cfg, epoch)

                msg, msg_bin = get_rnd_msg(b, cfg, "train")

                losses, output_imgs, msg_ret = self.train_on_batch(
                    [font_img, bg_img, 0, msg, msg_bin], epoch
                )

                if first_iter:
                    save_images(
                        font_img,
                        output_imgs,
                        msg_ret,
                        idx_to_save,
                        epoch,
                        step,
                        os.path.join(run_dir, "train_images"),
                        resize_to=saved_images_size,
                    )
                    first_iter = False

                for name, loss in losses.items():
                    training_losses[name].update(loss)
                if step % print_each == 0 or step == steps:
                    logging.info(
                        "Epoch: {}/{} Step: {}/{}".format(epoch, epochs, step, steps)
                    )
                    log_progress(training_losses)
                    logging.info("-" * 40)
                step += 1
                if step > steps:
                    break

            train_duration = time.time() - epoch_start
            logging.info(
                "Epoch {} training duration {:.2f} sec".format(epoch, train_duration)
            )
            logging.info("-" * 40)
            write_losses(
                os.path.join(run_dir, "train.csv"), training_losses, epoch, train_duration
            )

            if logger is not None:
                logger.save_losses(training_losses, epoch)
                logger.save_grads(epoch)
                logger.save_tensors(epoch)

            avg_acc = training_losses["dec_noise_acc"].avg
            if epoch % cfg.save_cp_freq == 0:
                if avg_acc > test_acc_best:
                    test_acc_best = avg_acc
                acc_str = str(round(avg_acc, 4))
                save_cp_dir = os.path.join(run_dir, "checkpoints")
                save_checkpoint(self, cfg.exp_name, epoch, save_cp_dir, acc_str)

            write_losses(
                os.path.join(run_dir, "eval.csv"),
                training_losses,
                epoch,
                time.time() - epoch_start,
            )

