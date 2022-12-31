import os
import torch
import argparse
from torch import nn, optim
from torch.utils.data import DataLoader
from model import UNet
from dataset import KeyholeDataset

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', help='path of keyhole dataset', default='data')
parser.add_argument('--weight_path', help='path of UNet parameters', default='param/unet.pth')
parser.add_argument('-e', '--epochs', help='number of epochs', type=int, default=10)
args = parser.parse_args()

if __name__ == '__main__':
    data_loader = DataLoader(KeyholeDataset(args.data_path), batch_size=4, shuffle=True)
    unet = UNet().cuda()

    if os.path.exists(args.weight_path):
        unet.load_state_dict(torch.load(args.weight_path))
        print('Successfully loaded weights')
    else:
        print('Failed to load weights')

    opt = optim.Adam(unet.parameters(), amsgrad=True)
    loss_func = nn.BCELoss()

    epoch = 1
    while epoch <= args.epochs:
        for i, (image, segment_image) in enumerate(data_loader):
            image, segment_image = image.cuda(), segment_image.cuda()

            output_image = unet(image)
            train_loss = loss_func(output_image, segment_image)

            opt.zero_grad()
            train_loss.backward()
            opt.step()

            if i % 100 == 0:
                print('\repoch: {:>5d}/{:<5d} batch: {:>5d}/{:<5d} loss: {:>10.8f}'.format(
                    epoch, args.epochs, i, len(data_loader), train_loss.item()
                ), end='\n' if i + 100 > len(data_loader) - 1 else '')
                torch.save(unet.state_dict(), args.weight_path)
        epoch += 1
        torch.save(unet.state_dict(), args.weight_path)