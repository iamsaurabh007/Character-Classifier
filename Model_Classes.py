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


class Conv_block(nn.Module):
    def __init__(self,inp=3):
        super(Conv_block,self).__init__()
        self.conv1=nn.Conv2d(in_channels=inp,out_channels=64,kernel_size=3)
        self.conv2a=nn.Conv2d(in_channels=inp,out_channels=64,kernel_size=3,padding=(1,1))
        self.conv2b=nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3)
        self.conv3=nn.Conv2d(in_channels=inp,out_channels=64,kernel_size=3,dilation=2,padding=(1,1))
        self.instance_norm=nn.InstanceNorm2d(num_features=64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        self.drop=nn.Dropout2d(p=0.05, inplace=False)
        self.drop2=torch.nn.Dropout(p=0.5, inplace=False)
        self.pool=nn.MaxPool2d(2, stride=2, padding=0, dilation=1, return_indices=False, ceil_mode=False)
    
    def _forward(self,x):
        branch1=self.conv1(x)
        branch2=self.conv2a(x)
        branch2=self.conv2b(branch2)
        branch3=self.conv3(x)
        outputs = [branch1,branch2,branch3]
        return outputs
    
    def c_relu(self,x):
        x_m=x*-1
        ###CHECK WITH DIMENSION AXIS
        a=torch.cat([x,x_m],1)
        return F.relu(a,inplace=True)
    
    def forward(self,x):
        outputs=self._forward(x)
        a=torch.cat(outputs,1)
        #print(a.shape)
        a=self.c_relu(a)
        #print(a.shape)
        b=self.instance_norm(a)
        #print(b.shape)
        b=self.drop(b)
        #print(b.shape)
        b=self.drop2(b)
        b=self.pool(b)
        #print(b.shape)
        return b
    
class Symbol_Model(nn.Module):
    def __init__(self  ):
        super(Symbol_Model,self).__init__()
        self.conv_block1=Conv_block()
        self.conv_block2=Conv_block(384)
        self.conv1=nn.Conv2d(in_channels=384,out_channels=config.num_classes,kernel_size=1)
    
    def forward(self,x):
        x=self.conv_block1(x)
        x=self.conv_block2(x)
        x=self.conv1(x)
        #print(x.shape)
        x=x.abs_()
        x=x.mean(dim=(2,3))
        x=torch.sigmoid(x) 
        #print("FINAL TENSOR")
        #print(x.shape)
        return x
    
    def loss_fn(self, output, target):
        loss = torch.exp(torch.sum((output - target)**2))
        return loss
    
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = self.loss_fn(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = self.loss_fn(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    _,lab= torch.max(labels, dim=1)
    return torch.tensor(torch.sum(preds == lab).item() / len(preds))
def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader,writer,opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase 
        running_loss=0.0
        for batch in train_loader:
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            running_loss+=loss.item()
        print("Training done at epoch",epoch)
        writer.add_scalar('training loss per epoch',running_loss,epoch)
        # Validation phase
        result = evaluate(model, val_loader)
        writer.add_scalar('validation loss per epoch',result['val_loss'],epoch)
        writer.add_scalar('validation acc per epoch',result['val_acc'],epoch)
        model.epoch_end(epoch, result)
        history.append(result)
    return history