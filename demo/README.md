## Demo  
We provide the SVG files for the 1-bit watermarked SimSun font which includes two variants, along with a test set collected across 7 distribution scenarios. Each scenario's test set contains 1000 segmented character images.
All the data can be found [here](https://pan.baidu.com/s/1n8z7o1pPgJpfsHl5hXP-5w?pwd=rocu) (Password: rocu).  

The data is organized as follows:  
```
|-WeChat  # test set for OSNs scenario
|---FontGuard_SimSun_16
|-Weibo  # test set for OSNs scenario
|---FontGuard_SimSun_16
|-Whatsapp  # test set for OSNs scenario
|---FontGuard_SimSun_16
|-Facebook  # test set for OSNs scenario
|---FontGuard_SimSun_16
|-print_camera  # test set for cross-media scenario
|---FontGuard_SimSun_16
|-screen_camera  # test set for cross-media scenario
|---FontGuard_SimSun_16
|-screenshots  # test set for cross-media scenario
|---FontGuard_SimSun_16
|-svg  # 1-bit watermarked SimSun SVG
|---msg_0
|---msg_1
|-bit_seq.txt  # bitstream ground truth for the test set
|-dec.pth  # decoder checkpoint
|-GB2312_CN6763.txt  # character set
```

To extract the bitstream from test set font images, first update the paths in `cfg.py`. Then, execute the following command:
```bash
python test.py
```
 
