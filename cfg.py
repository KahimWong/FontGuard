import time
import os
import os.path as op
from utils import get_noise_layer
from model.noise_layers.crop import Crop
from model.noise_layers.cropout import Cropout
from model.noise_layers.dropout import Dropout
from model.noise_layers.resize import Resize
from model.noise_layers.blur import Blur
from model.noise_layers.jpeg import JPEG
from model.noise_layers.noise import Noise
from model.noise_layers.color_jitter import ColorJitter
from model.noise_layers.perspective_warp import PerspectiveWarp

# path
exp_name = "SimSun"
fontguard_ckpt = "/data/jesonwong47/FontCode/FontGuard/SimSun/exp_output/SimSun_m1_final/checkpoints/SimSun_m1_final--epoch-150--acc-0.pyt"
root = "/data/jesonwong47/FontCode/FontGuard/exp_data"

font_dir = op.join(root, "ori_png")  # the directory of font images for training
base_sty_path = op.join(root, "base_sty_feat_CH.pth")  # the path of the extracted style features of font images
pretrain_dec_ckpt = op.join(root, "clip_cls_CH.pt")  # the checkpoint of the pre-trained decoder, we pretrain the clip image encoder with the font classification task to provide a better initialization for the decoder. 
bg_dir = op.join(root, "val2017")  # the directory of background images of font for background augmentation, you can use any natural images as background images. We use the COCO 2017 val images in our experiments.
exp_dir = op.join(root, f'{exp_name} {time.strftime("%Y.%m.%d--%H-%M-%S")}')

os.makedirs(root, exist_ok=True)
os.makedirs(exp_dir, exist_ok=True)
os.makedirs(op.join(exp_dir, "ckpt"), exist_ok=True)
os.makedirs(op.join(exp_dir, "vis_img"), exist_ok=True)

# curriculum
init_epoch = 5
start_noise_epoch = 50
full_noise_epoch = 100

max_bg_p = 0.8
max_noise_p = 0.2

# loss weight
qlt_w = 0.02
gan_w = 0.1
nce_w = 0.01
msg_w = 1.0
noise_msg_w = 1.0

# optimizer
enc_lr = 1e-3
dec_lr = 1e-4
disc_lr = 1e-3
alpha = 0.99
weight_decay = 1e-4

# dataloader
bs = 64
workers = 1
msg_bit = 1
msg_n = int(2**msg_bit)
bs = bs // msg_n  # real batch size
step_per_epoch = -1  # uni: 1000; single font: -1
epochs = 150


# noise
noises = get_noise_layer(
    Crop((0.7, 0.99), (0.7, 0.99)),
    Cropout((0.7, 0.99), (0.7, 0.99)),
    Dropout((0.7, 0.99)),
    Resize((0.7, 0.99)),
    JPEG(img_size=80, qf=(70, 99)),
    PerspectiveWarp(img_size=80, max_trans_f=0.1),
    Blur(kernel_size=3, max_sigma=1),
    Noise(noise_var=0.02),
    ColorJitter(brightness=(-0.3, 0.3), contrast=(0.7, 1.3)),
)

# others
db_temp = 10
db_thresh = 0.3  # ch: 0.3  larger thresh, more text pixels
font_img_size = 80
clip_img_size = 224
clip_img_f_dim = 1024
num_sty_feat = 240 + 1  # ch single font: 240 + 1; eng single font: 104 + 1
sty_feat_dim = 128
print_freq = 100
save_cp_freq = 50
start_save_cp_epoch = 10
