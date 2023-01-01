# UNet
UNet for container keyhole segmentation
### Usage
#### extract
```bash
cd param/
cat unet.pth.tar.gz.* > unet.pth.tar.gz
tar -xzvf unet.pth.tar.gz
```
#### train
```bash
python train.py --epochs 20
```
#### test
```bash
python test.py
```
#### detect
```bash
python detect.py -i data/test/JPEGImages/00001.jpg
```