import torch
import torch.cuda as cuda
import torch.utils.data as data
import os
import string
from os import listdir
from os.path import isfile, join
import torchvision
from PIL import Image
import numpy as np
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import config
import Model_Classes


class IMGDS(data.Dataset):
    #Reuires a directiory with imgs and json folder in it
    def __init__(self, label_dict,root_dir,imglist, transform=None):
        """
        Args:
            label_dict: mapping from labels to class
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.label_dict=label_dict
        self.images_list=imglist

    def loadimage(self,index):
        im = Image.open(self.root_dir+"/imgs/"+self.images_list[index]+".jpeg")
        
        desired_size = 200
        
        old_size = im.size  # old_size[0] is in (width, height) format

        #ratio = float(desired_size)/max(old_size)
        #new_size = tuple([int(x*ratio) for x in old_size])
        # use thumbnail() or resize() method to resize the input image

        # thumbnail is a in-place operation

        # im.thumbnail(new_size, Image.ANTIALIAS)

        #im = im.resize(new_size, Image.ANTIALIAS)
        # create a new image and paste the resized on it

        new_im = Image.new("RGB", (desired_size, desired_size),color = (255,255, 255))
        a=(desired_size-old_size[0])//2
        b=(desired_size-old_size[1])//2
        new_im.paste(im, (a,b))
        image=np.array(new_im)
        image=image/255
        image=image-1
        image=image.astype('float32')
        image=torchvision.transforms.functional.to_tensor(image)
        #image=image.float()
        #image = torch.from_numpy(image)
        #image=torchvision.transforms.functional.to_tensor(image)
        #image=image.float()
        return image
        #return image
    def loadlabel(self,index):
        with open(self.root_dir+"/json/"+self.images_list[index]+".json") as f:
            d= json.load(f)
            label=d['image']['character']
            a=np.zeros((97))
            a=a+0.5
            a[self.label_dict[label]]=0.9
            a=torch.from_numpy(a)
            return a
    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image=self.loadimage(idx)
        label=self.loadlabel(idx)
#         if self.transform:
#             sample = self.transform(sample)
        return image,label


class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield utils.to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)



