"""Dataset process
Reference
    Original author: YutaroOgawa
    https://github.com/YutaroOgawa/pytorch_advanced/blob/master/3_semantic_segmentation/utils/dataloader.py

    @article{mshahsemseg,
        Author = {Meet P Shah},
        Title = {Semantic Segmentation Architectures Implemented in PyTorch.},
        Journal = {https://github.com/meetshah1995/pytorch-semseg},
        Year = {2017}
    }
    https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/loader/pascal_voc_loader.py
"""

from PIL import Image
import numpy as np

import torch
import torch.utils.data as data

from data_process.utils.data_augmentation import Compose, Scale, RandomRotation, RandomMirror, Resize, Normalize_Tensor


class DataTransform():
    def __init__(self, input_size, color_mean, color_std, mode):
        if mode == 'train':
            self.data_transform = Compose([
                Scale(scale=[0.5, 1.5]),
                RandomRotation(angle=[-10, 10]),
                RandomMirror(),
                Resize(input_size),
                Normalize_Tensor(color_mean, color_std)
            ])
        elif mode == 'test':
            self.data_transform = Compose([
                Resize(input_size),
                Normalize_Tensor(color_mean, color_std)
            ])

    def __call__(self, img, anno_class_img):
        return self.data_transform(img, anno_class_img)


class VOCDataset(data.Dataset):
    def __init__(self, img_list, anno_list, transform, label_color_map):
        self.img_list = img_list
        self.anno_list = anno_list
        self.transform = transform
        self.label_color_map = label_color_map # list [[]]

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img, anno, img_filepath = self.pull_item(index)
        return img, anno, img_filepath

    def pull_item(self, index):
        
        img_filepath = self.img_list[index]
        img = Image.open(img_filepath)

        anno_filepath = self.anno_list[index]
        anno = Image.open(anno_filepath)
        # anno = Image.fromarray(self.encode_segmap(np.array(anno)))

        img, anno = self.transform(img, anno)

        return img, anno, img_filepath

    def encode_segmap(self, mask):

        mask = mask.astype(int)
        label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)

        for ii, label in enumerate(np.asarray(self.label_color_map)):
            label_mask[np.where(np.all(mask==label, axis=-1))[:2]] = ii
        label_mask = label_mask.astype(np.uint8)
        return label_mask