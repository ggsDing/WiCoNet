import os
import cv2
import torch
import numpy as np
from skimage import io
from torch.utils import data
import matplotlib.pyplot as plt
from torchvision import transforms
import utils.transform as transform
from torchvision.transforms import functional as F

num_classes = 6
PD_COLORMAP = [[0, 0, 0], [255, 255, 255], [0, 0, 255], [0, 255, 255],
                [0, 255, 0], [255, 255, 0], [255, 0, 0] ]
PD_CLASSES  = ['Invalid', 'Impervious surfaces','Building', 'Low vegetation',
                'Tree', 'Car', 'Clutter/background']
PD_MEAN = np.array([85.8, 91.7, 84.9, 96.6, 47])
PD_STD  = np.array([35.8, 35.2, 36.5, 37, 55])


def normalize_image(im):
    return (im - PD_MEAN) / PD_STD


def normalize_images(imgs):
    for i, im in enumerate(imgs):
        imgs[i] = normalize_image(im)
    return imgs


colormap2label = np.zeros(256 ** 3)
for i, cm in enumerate(PD_COLORMAP):
    colormap2label[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i


def Index2Color(pred):
    colormap = np.asarray(PD_COLORMAP, dtype='uint8')
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

def get_file_name(mode='train'):
    assert mode in ['train', 'val']
    if mode == 'train':
        img_path = os.path.join(root, 'train')
        pred_path = os.path.join(root, 'numpy', 'train')
    else:
        img_path = os.path.join(root, 'val')
        pred_path = os.path.join(root, 'numpy', 'val')

    data_list = os.listdir(img_path)
    numpy_path_list = [os.path.join(pred_path, it) for it in data_list]
    return numpy_path_list

def read_RSimages(data_dir, mode):
    print('Reading data from '+data_dir+':')
    assert mode in ['train', 'val', 'test']
    data_list = []
    img_dir = os.path.join(data_dir, mode)
    dsm_dir = os.path.join(data_dir, mode, 'dsm')
    mask_dir = os.path.join(data_dir, 'groundtruth_noBoundary')
    item_list = os.listdir(img_dir)
    for item in item_list:
        if (item[-4:]=='.tif'): data_list.append(item)
    data_length = int(len(data_list))
    count=0
    data, labels = [], []
    for it in data_list:
        # print(it)
        img_name = it
        dsm_name = 'dsm' + it[3:-10] + '.jpg'
        mask_name = it[:-10] + '_label_noBoundary.tif'
        img_path = os.path.join(img_dir, img_name)
        dsm_path = os.path.join(dsm_dir, dsm_name)
        mask_path = os.path.join(mask_dir, mask_name)
        img = io.imread(img_path)
        dsm = io.imread(dsm_path)
        img = np.concatenate((img, np.expand_dims(dsm, axis=2)), axis=2)
        label = Color2Index(io.imread(mask_path))
        data.append(img)
        labels.append(label)
        count+=1
        if not count%5: print('%d/%d images loaded.'%(count, data_length))
    print(data[0].shape)
    print(str(len(data)) + ' ' + mode + ' images' + ' loaded.')
    return data, labels

def rescale_images(imgs, scale, order):
    rescaled_imgs = []
    for im in imgs:
        im_r = rescale_image(im, scale, order)
        rescaled_imgs.append(im_r)
    return rescaled_imgs
    
def rescale_image(img, scale=8, order = 0):
    flag = cv2.INTER_NEAREST
    if order==1: flag = cv2.INTER_LINEAR
    elif order==2: flag = cv2.INTER_AREA
    elif order>2:  flag = cv2.INTER_CUBIC
    im_rescaled = cv2.resize(img, (int(img.shape[0]/scale), int(img.shape[1]/scale)), interpolation=flag)
    return im_rescaled

class Loader(data.Dataset):
    def __init__(self, data_dir, mode, random_crop=False, crop_nums=40, random_flip = False, sliding_crop=False,
                 size_context=256*3, size_local=320, scale=4):
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



