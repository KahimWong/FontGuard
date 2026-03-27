import os.path as op

root = '/data/jesonwong47/FontCode/demo_data'  # TODO: modify the root path
font_name = 'SimSun'
dec_ckpt_path = op.join(root, 'dec.pth')  # decoder checkpoint path
gt_path = op.join(root, 'bit_seq.txt')  # ground truth path

pt_list = [16]  # font size
scenario_list = ['screen_camera', 'screenshots', 'print_camera', 'Whatsapp', 'Facebook', 'WeChat', 'Weibo']

batch_size = 64
num_workers = 1
msg_len = 1
num_cls = int(2 ** msg_len)
clip_img_size = 224
font_img_size = 80