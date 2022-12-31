import os
import cv2
from torch.utils.data import Dataset
from utils import resize, threshold, transform


class KeyholeDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.segment_image_names = os.listdir(os.path.join(data_path, 'SegmentationClass'))

    def __len__(self):
        return len(self.segment_image_names)

    def __getitem__(self, index):
        segment_image_name = self.segment_image_names[index]
        segment_image_path = os.path.join(self.data_path, 'SegmentationClass', segment_image_name)
        image_path = os.path.join(self.data_path, 'JPEGImages', segment_image_name)

        segment_image = resize(threshold(cv2.imread(segment_image_path, 0)))
        image = resize(cv2.imread(image_path))
        return transform(image), transform(segment_image)
