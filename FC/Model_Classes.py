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

out_chnl=config.channel
model_dir=config.MODELCHECKPOINT_PATH

class LINEAR_BLOCK(nn.Module):
    def __init__(self,in_features,out_features,bias):
        super(LINEAR_BLOCK,self).__init__()
        self.linear=nn.Linear(in_features=in_features, out_features=out_features, bias=bias)
        self.lrelu=nn.LeakyReLU()
    def forward(self,x):
        x=self.linear(x)
        x=self.lrelu(x)
        return x
class Conv_block(nn.Module):
    def __init__(self,inp=3):
        super(Conv_block,self).__init__()
        self.conv1=nn.Conv2d(in_channels=inp,out_channels=out_chnl,kernel_size=3)
        self.conv2a=nn.Conv2d(in_channels=inp,out_channels=out_chnl,kernel_size=3,padding=(1,1))
        self.conv2b=nn.Conv2d(in_channels=out_chnl,out_channels=out_chnl,kernel_size=3)
        self.conv3=nn.Conv2d(in_channels=inp,out_channels=out_chnl,kernel_size=3,dilation=2,padding=(1,1))
        '''self.instance_norm=nn.InstanceNorm2d(num_features=out_chnl*6, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)'''
        self.layernorm=nn.LayerNorm(out_chnl*6) #CHECK WITH DIMENSION OVER NORMALIZATION NOW AT CHANNEL
        self.drop=nn.Dropout2d(p=0.005, inplace=False)
        self.drop2=torch.nn.Dropout(p=0.2, inplace=False)
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
        b=self.layernorm(a)
        #print(b.shape)
        b=self.drop(b)
        #print(b.shape)
        b=self.drop2(b)
        b=self.pool(b)
        #print(b.shape)
        return b
    
class FC_Model(nn.Module):
    def __init__(self):
        super(FC_Model,self).__init__()
        self.conv_block1=Conv_block()
        self.conv_block2=Conv_block(out_chnl*6)
        self.conv1=nn.Conv2d(in_channels=out_chnl*6,out_channels=512,kernel_size=1)
        self.linearblock1=LINEAR_BLOCK(in_features=512, out_features=512, bias=True)
        self.linearblock2=LINEAR_BLOCK(in_features=512, out_features=256, bias=True)
        self.linear2=nn.Linear(in_features=256, out_features=config.num_classes, bias=True)
        self.loss_fn=nn.CrossEntropyLoss()
    
    def forward(self,x):
        #print(x.shape)
        x=self.conv_block1(x)
        #print(x.shape)
        x=self.conv_block2(x)
        #print(x.shape)
        x=self.conv1(x)
        #print(x.shape)
        #x=x.abs()
        x=x.sum(dim=(2,3))
        #print(x.shape)
        x=self.linearblock1(x)
        x=self.linearblock2(x)
        #print(x.shape)
        x=self.linear2(x)
        #print("FINAL TENSOR")
        #print(x.shape)
        return x
    
    
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = self.loss_fn(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        with torch.no_grad(): 
            out = self(images)                    # Generate predictions
            loss = self.loss_fn(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss, 'val_acc': acc}
        
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
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader,writer,opt_func):
    history = []
    optimizer = opt_func(model.parameters(), lr, weight_decay=1e-4)
    for epoch in range(epochs):
        # Training Phase 
        running_loss=[]
        for batch in train_loader:
            optimizer.zero_grad()
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            running_loss.append(loss.item())
        loss_mean=sum(running_loss) / len(running_loss)
        print("Training done at epoch",epoch,"training_loss=",loss_mean)
        writer.add_scalar('training loss per epoch',loss_mean,epoch)
        # Validation phase
        result = evaluate(model, val_loader)
        writer.add_scalar('validation loss per epoch',result['val_loss'],epoch)
        writer.add_scalar('validation acc per epoch',result['val_acc'],epoch)
        model.epoch_end(epoch, result)
        history.append(result)
        if epoch%5==1:
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss_mean,
                        }, os.path.join(model_dir, 'epoch-{}.pt'.format(epoch)))
    return history
