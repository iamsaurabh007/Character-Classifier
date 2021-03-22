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
import DataUtils
import Model_Classes

dir_path=config.dir_path
device=config.device
num_epochs=config.num_epochs
l_r=config.learning_rate

if __name__ =='__main__':

    if device==None:
        device = utils.get_default_device()
    label_dict=utils.create_label_dict(config.symbols)
    imglist_train,imglist_val=utils.get_images_list(dir_path+"/imgs")
    ds_train=DataUtils.IMGDS(label_dict,dir_path,imglist_train)
    ds_val=DataUtils.IMGDS(label_dict,dir_path,imglist_val)
    train_gen = torch.utils.data.DataLoader(ds_train ,batch_size=1,shuffle=True,num_workers =2)
    valid_gen= torch.utils.data.DataLoader(ds_val,batch_size=1,shuffle=True,num_workers =2)
    train_gen = DataUtils.DeviceDataLoader(train_gen, device)
    valid_gen = DataUtils.DeviceDataLoader(valid_gen, device)
    model=Model_Classes.Symbol_Model()
    model=utils.to_device(model, device)
    history=Model_Classes.fit(num_epochs,l_r,model,train_gen, valid_gen, opt_func=torch.optim.SGD)

