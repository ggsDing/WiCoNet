import os
import cv2
import torch
import numpy as np
from skimage import io
from torch.utils import data
import utils.transform as transform
from skimage.transform import rescale

num_classes = 6
BLU_COLORMAP = [[0, 0, 0], [255, 255, 255], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [125, 125, 125]]
BLU_CLASSES  = ['Invalid',   'Background',  'Building', 'Vegetation', 'Water',     'Farmland',     'Road']
BLU_MEAN = np.array([122.19, 121.35, 117.29])
BLU_STD  = np.array([63.26, 62.45, 61.65])

def normalize_image(im):
    return (im - BLU_MEAN) / BLU_STD


def normalize_images(imgs):
    for i, im in enumerate(imgs):
        imgs[i] = normalize_image(im)
    return imgs


colormap2label = np.zeros(256 ** 3)
for i, cm in enumerate(BLU_COLORMAP):
    colormap2label[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i


def Index2Color(pred):
    colormap = np.asarray(BLU_COLORMAP, dtype='uint8')
    x = np.asarray(pred, dtype='int32')
    return colormap[x, :]


def Colorls2Index(ColorLabels):
    for i, data in enumerate(ColorLabels):
        ColorLabels[i] = Color2Index(data)
    return ColorLabels


def Color2Index(ColorLabel):
    data = ColorLabel.astype(np.int32)
    idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
    IndexMap = colormap2label[idx]
    IndexMap = IndexMap * (IndexMap <= num_classes)
    return IndexMap.astype(np.uint8)


def read_RSimages(data_dir, mode, rescale_ratio=False):
    print('Reading data from '+data_dir+':')
    assert mode in ['train', 'val', 'test']
    data_list = []
    img_dir = os.path.join(data_dir, mode, 'image')
    item_list = os.listdir(img_dir)
    for item in item_list:
        if (item[-4:]=='.png'): data_list.append(os.path.join(img_dir, item))
    data_length = int(len(data_list))
    count=0
    data, labels = [], []
    for it in data_list:
        # print(it)
        img_path = it
        mask_path = img_path.replace('image', 'label')
        img = io.imread(img_path)
        label = Color2Index(io.imread(mask_path))
        if rescale_ratio:
            img = rescale_image(img, rescale_ratio, 2)
            label = rescale_image(label, rescale_ratio, 0)
        data.append(img)
        labels.append(label)
        count+=1
        if not count%10: print('%d/%d images loaded.'%(count, data_length))
    print(data[0].shape)
    print(str(len(data)) + ' ' + mode + ' images' + ' loaded.')
    return data, labels


def rescale_images(imgs, scale, order=0):
    rescaled_imgs = []
    for im in imgs:
        rescaled_imgs.append(rescale_image(im, scale, order))
    return rescaled_imgs


def rescale_image(img, scale=8, order=0):
    flag = cv2.INTER_NEAREST
    if order==1: flag = cv2.INTER_LINEAR
    elif order==2: flag = cv2.INTER_AREA
    elif order>2:  flag = cv2.INTER_CUBIC
    im_rescaled = cv2.resize(img, (int(img.shape[0]/scale), int(img.shape[1]/scale)), interpolation=flag)
    return im_rescaled


class Loader(data.Dataset):
    def __init__(self, data_dir, mode, random_crop=False, crop_nums=40, random_flip=False, sliding_crop=False,\
                 size_context=256*3, size_local=256, scale=4):
        self.size_context = size_context
        self.size_local = size_local
        self.scale = scale
        self.crop_nums = crop_nums
        self.random_flip = random_flip
        self.random_crop = random_crop
        data, labels = read_RSimages(data_dir, mode)
        self.data = data
        self.labels = labels
        
        data_s = rescale_images(data, scale, 2)
        labels_s = rescale_images(labels, scale, 0)
        padding_size = (size_context-size_local)/scale/2
        self.data_s, self.labels_s = transform.data_padding_fixsize(data_s, labels_s, [padding_size, padding_size])
        if sliding_crop:
            self.data_s, self.labels_s, self.data, self.labels = transform.slidding_crop_WC(self.data_s, self.labels_s,\
                self.data, self.labels, size_context, size_local, scale)
            
        if self.random_crop:
            self.len = crop_nums*len(self.data)
        else:
            self.len = len(self.data)
    
    def __getitem__(self, idx):
        if self.random_crop:
            idx = int(idx/self.crop_nums)
            data_s, label_s, data, label = transform.random_crop2(self.data_s[idx], self.labels_s[idx],\
                self.data[idx], self.labels[idx], self.size_context, self.size_local, self.scale)
        else:
            data = self.data[idx]
            label = self.labels[idx]
            data_s = self.data_s[idx]
            label_s = self.labels_s[idx]
        if self.random_flip:
            data_s, label_s, data, label = transform.rand_flip2(data_s, label_s, data, label)
            
        data_s = normalize_image(data_s)
        data_s = torch.from_numpy(data_s.transpose((2, 0, 1)))   
        data = normalize_image(data)
        data = torch.from_numpy(data.transpose((2, 0, 1)))
        return data_s, label_s, data, label

    def __len__(self):
        return self.len



