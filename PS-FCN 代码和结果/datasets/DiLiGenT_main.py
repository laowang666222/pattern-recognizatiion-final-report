from __future__ import division
import os
import numpy as np
#from scipy.ndimage import imread
from imageio import imread
import scipy.io as sio

import torch
import torch.utils.data as data

from datasets import pms_transforms
from . import util
np.random.seed(0)

class DiLiGenT_main(data.Dataset):
    def __init__(self, args, split='train'):
        self.root   = os.path.join(args.bm_dir)
        self.split  = split
        self.args   = args
        #读取所有obj的名字
        self.objs   = util.readList(os.path.join(self.root, 'objects.txt'), sort=False)
        self.names  = util.readList(os.path.join(self.root, 'filenames.txt'), sort=False)
        #这是光源方向，程序自带呦
        self.l_dir  = util.light_source_directions()
        print('[%s Data] \t%d objs %d lights. Root: %s' % 
                (split, len(self.objs), len(self.names), self.root))
        self.intens = {}
        intens_name = 'light_intensities.txt'
        print('Files for intensity: %s' % (intens_name))
        #读取所有obj文件对应的光照强度
        for obj in self.objs:
            self.intens[obj] = np.genfromtxt(os.path.join(self.root, obj, intens_name))

    #获取所有文件的mask图，也就是图像中物体的位置
    def _getMask(self, obj):
        mask = imread(os.path.join(self.root, obj, 'mask.png'))
        if mask.ndim > 2: mask = mask[:,:,0]
        mask = mask.reshape(mask.shape[0], mask.shape[1], 1)
        return mask / 255.0

    #根据obj的索引读取数据，也即不同obj对应的96张图像
    def __getitem__(self, index):
        np.random.seed(index)
        #得到名字
        obj = self.objs[index]

        #随机选取32个方向作为输入，因为网络形状吧
        select_idx = np.random.permutation(len(self.names))[:self.args.in_img_num]
        print(select_idx)

        #得到要读取文件的名字列表
        img_list   = [os.path.join(self.root, obj, self.names[i]) for i in select_idx]
        intens     = [np.diag(1 / self.intens[obj][i]) for i in select_idx]

        #得到obj对应的法向量的图
        normal_path = os.path.join(self.root, obj, 'Normal_gt.mat')
        normal = sio.loadmat(normal_path)
        normal = normal['Normal_gt']

        print("******************读取图片中*********************")
        imgs = []
        #读取96张图片，并且将图片除以光照强度，得到标准单位的图片
        for idx, img_name in enumerate(img_list):
            img = imread(img_name).astype(np.float32) / 255.0
            img = np.dot(img, intens[idx])
            imgs.append(img)
        #对图片进行归一化，也就是让图片所有像素平方和加起来等于1，当然不清楚是不是某一行square和是1或者某个通道是1
        if self.args.normalize:
            imgs = pms_transforms.normalize(imgs)
        #将96张图片直接拼接在一起
        img = np.concatenate(imgs, 2)
        #对img这个东西归一化
        if self.args.normalize:
            img = img * np.sqrt(len(imgs) / self.args.train_img_num) # TODO

        #得到mask图，也就是物体的位置
        mask = self._getMask(obj)
        #对输入图片进行padding操作，在img的最上面和最右面padd，是的img的形状可以被down整除，这是干嘛的？
        down = 4
        if mask.shape[0] % down != 0 or mask.shape[1] % down != 0:
            pad_h = down - mask.shape[0] % down
            pad_w = down - mask.shape[1] % down
            img = np.pad(img, ((0,pad_h), (0,pad_w), (0,0)), 'constant', constant_values=((0,0),(0,0),(0,0)))
            mask = np.pad(mask, ((0,pad_h), (0,pad_w), (0,0)), 'constant', constant_values=((0,0),(0,0),(0,0)))
            normal = np.pad(normal, ((0,pad_h), (0,pad_w), (0,0)), 'constant', constant_values=((0,0),(0,0),(0,0)))
        #repeat：复制mask，使得img可以和mask直接相乘
        img  = img * mask.repeat(img.shape[2], 2)
        #构建obj物体的结构体，有法向量、预处理后的图片、mask图、obj名字和一个神秘的light
        item = {'N': normal, 'img': img, 'mask': mask}

        #转换成张量
        for k in item.keys():
            item[k] = pms_transforms.arrayToTensor(item[k])

        #obj的item里增加光源方向
        if self.args.in_light:
            #将light方向flatten成一维
            item['light'] = torch.from_numpy(self.l_dir[select_idx]).view(-1, 1, 1).float()
        item['obj'] = obj
        return item

    def __len__(self):
        return len(self.objs)
