import Model_Classes
import config
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
import pandas as pd
import torch.nn.functional as F
def loadimage(path):
    im = Image.open(path)
    image=np.array(im)
    image=image/255
    image=image-1
    image=image.astype('float32')
    image=torchvision.transforms.functional.to_tensor(image)
    return image

def create_label_dict():
    symbols=list(string.printable[:94])
    symbols.append(u"\u00A9")
    symbols.append(u"\u2122")
    symbols.append(" ")
    label={}
    rev={}
    for i,sym in enumerate(symbols):
        label[sym]=i
        rev[i]=sym
    print("Dictionary Created with {} symbols".format(len(symbols)))
    return label,rev

def get_images_list(mypath,number=None):
    #Currently jpeg implementation only
    onlyfiles = [f[:-5] for f in listdir(mypath) if isfile(join(mypath, f))]
    random.shuffle(onlyfiles)
    if number:
        onlyfiles=onlyfiles[:number]   
    return onlyfiles

def returnclass(outputs):
    _, preds = torch.max(outputs, dim=1)
    return preds.item()


if __name__ =='__main__':
    device=config.device
    if device==None:
        device = utils.get_default_device()
    model=Model_Classes.FC_Model()
    checkpoint = torch.load("/home/ubuntu/data/ocr/ModelPTfullrun/epoch-21.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    print("MODEL LOADED SUCCESSFULLY")
    model.eval()
    
    files=get_images_list("/home/ubuntu/data/ocr/out/imgs",4000)
    ls=[]

    for file in files:
        with open("/home/ubuntu/data/ocr/out/json/"+file+".json") as f:
            d= json.load(f)
            ls.append(d['image'])
    print("IMAGE LIST LOADED")
    ls_logits=[]
    ls_label=[]
    df=pd.DataFrame(ls)
    label,rev=create_label_dict()
    for cnt,i in enumerate(df['image_id']):
        a="/home/ubuntu/data/ocr/out/out/imgs/"+i+".jpeg"
        p=loadimage(a)
        with torch.no_grad():    
            q=model(torch.unsqueeze(p,0))
        ls_logits.append(q.numpy())
        if cnt%100==1:
            print(cnt)	
        print()
        ls_label.append(rev_dict[returnclass(q)])    

    df['predicted_class']=ls_label
    df['logits']=ls_logits
    #df['pred']=k
    df['logits']=ls_logits
    df["x"]=df['character']==df['predicted_class']
    print(df['x'].sum())
    input("Press Enter")
