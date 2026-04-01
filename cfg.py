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
fontguard_ckpt = None
root = "/data/jesonwong47/FontCode/script/FontGuard/exp_data"
exp_root = "/data/jesonwong47/FontCode/script/FontGuard/exp_out"
font_dir = op.join(root, 'SimSun')  # the directory of font images for training
base_sty_path = op.join(root, "base_sty_feat_CH.pth")  # the path of the extracted style features of font images
pretrain_dec_ckpt = op.join(root, "clip_cls_CH.pt")  # the checkpoint of the pre-trained decoder, we pretrain the clip image encoder with the font classification task to provide a better initialization for the decoder. 
font_model_ckpt = op.join(root, "font_model_CH.ckpt")  # the checkpoint of the pre-trained font recognition model, we use it to extract the style features of font images. You can also use any other font recognition model to extract style features.
bg_dir = op.join(root, "val2017")  # the directory of background images of font for background augmentation, you can use any natural images as background images. We use the COCO 2017 val images in our experiments.
exp_dir = op.join(exp_root, f'{exp_name} {time.strftime("%Y.%m.%d--%H-%M-%S")}')

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
workers = 0
msg_bit = 1
msg_n = int(2**msg_bit)
msg_len = msg_bit
num_cls = msg_n
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
num_sty_feat = 240
sty_feat_dim = 128
print_freq = 1
save_cp_freq = 50
start_save_cp_epoch = 10

# model defaults required by discriminator/encoder modules
discriminator_channels = 64
discriminator_blocks = 3
