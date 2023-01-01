import os
import cv2
import time
import torch
import argparse
import numpy as np
from model import UNet
from utils import resize, threshold, transform

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--image_path', help='path of keyhole image', default='data/test/JPEGImages/00001.jpg')
parser.add_argument('--weight_path', help='path of UNet parameters', default='param/unet.pth')
args = parser.parse_args()


if __name__ == '__main__':
    unet = UNet().cuda().eval()

    if os.path.exists(args.weight_path):
        unet.load_state_dict(torch.load(args.weight_path))
        print('Successfully loaded weights')
    else:
        print('Failed to load weights')

    image = resize(cv2.imread(args.image_path))
    input_image = torch.unsqueeze(transform(image), dim=0).cuda()

    start = time.time()
    output_image = unet(input_image)
    end = time.time()
    print('Inference took %.4fs complete' % (end - start))

    output_image = output_image.cpu().detach().numpy().reshape(output_image.shape[-2:]) * 255
    binary = threshold(output_image.astype('uint8'))
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, (0, 255, 0), 3)

    y, x = map(int, np.mean(np.where(binary > 0), axis=1))
    cv2.circle(image, (x, y), 3, (0, 0, 255), 2)

    cv2.imshow('result', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
