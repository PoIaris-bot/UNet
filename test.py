import os
import cv2
import torch
import argparse
import numpy as np
from math import sqrt
from model import UNet
from utils import resize, threshold, transform

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', help='path of keyhole dataset', default='data/test')
parser.add_argument('--weight_path', help='path of UNet parameters', default='param/unet.pth')
args = parser.parse_args()

if __name__ == '__main__':
    unet = UNet().cuda().eval()

    if os.path.exists(args.weight_path):
        unet.load_state_dict(torch.load(args.weight_path))
        print('Successfully loaded weights')
    else:
        print('Failed to load weights')

    image_dir = os.path.join(args.data_path, 'JPEGImages')
    segment_dir = os.path.join(args.data_path, 'SegmentationClass')
    image_names = os.listdir(image_dir)

    avg_error = 0
    max_error = 0
    for image_name in image_names:
        image_path = os.path.join(image_dir, image_name)
        segment_image_path = os.path.join(segment_dir, image_name)
        image = resize(cv2.imread(image_path))
        segment_image = resize(cv2.imread(segment_image_path, 0))

        input_image = torch.unsqueeze(transform(image), dim=0).cuda()
        output_image = unet(input_image)

        output_image = output_image.cpu().detach().numpy().reshape(output_image.shape[-2:]) * 255
        output_binary = threshold(output_image.astype('uint8'))
        binary = threshold(segment_image)
        output_contours, _ = cv2.findContours(output_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, output_contours, -1, (0, 255, 0), 3)
        cv2.drawContours(image, contours, -1, (0, 0, 255), 3)

        y, x = map(int, np.mean(np.where(output_binary > 0), axis=1))
        y0, x0 = map(int, np.mean(np.where(binary > 0), axis=1))
        cv2.circle(image, (x, y), 3, (0, 255, 0), 2)
        cv2.circle(image, (x0, y0), 3, (0, 0, 255), 2)
        cv2.imwrite(f'result/{image_name}', image)

        error = sqrt((x - x0) ** 2 + (y - y0) ** 2)
        avg_error += error
        max_error = max_error if max_error >= error else error
    print(f'average error: {avg_error / len(image_names)} maximum error: {max_error}')
