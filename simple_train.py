
from torch.utils.data import Dataset, DataLoader,random_split
import os
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from  torchvision.datasets import ImageFolder

data_transform = transforms.Compose([
    
        # # transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(15),
        transforms.Resize([224,168]),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

train_dataset = ImageFolder('/home/xu/MyData/phone_pic/train',transform=data_transform)

val_dataset = ImageFolder('/home/xu/MyData/phone_pic/val',transform=data_transform)




device=torch.device('cuda:0')

weight='/home/xu/utest/0_0.pt'


model=models.MobileNetV2(num_classes=2).to(device)
model.load_state_dict(torch.load(weight))

model=model.to(device)

criterion=nn.CrossEntropyLoss()
            


t_batch_size=128
v_batch_size=256
train_loder=torch.utils.data.DataLoader(train_dataset,shuffle=True,batch_size=t_batch_size,num_workers=8,pin_memory=True)
val_loder=torch.utils.data.DataLoader(val_dataset,shuffle=True,batch_size=v_batch_size,num_workers=8,pin_memory=True)


criterion=nn.CrossEntropyLoss()
optimizers=torch.optim.Adam(model.parameters(),lr=1e-3)


for epoch in range(120):
    train_losses=[]
    train_acc=[]
  


    model.train()
    for step,(features,labels) in enumerate(train_loder):

        features,labels=features.to(device),labels.to(device)
       
        model.train()
        optimizers.zero_grad()
        result=model(features)

        acc=torch.sum(torch.eq(torch.max(result,1)[1].view_as(labels),labels)).item()/t_batch_size
        train_acc.append(acc)
    
        loss=criterion(result,labels)
        # print(loss)
        train_losses.append(loss.item())

        loss.backward()
        optimizers.step()
    
        if step%20==0:
            
            print("epoch :",epoch,"train_acc ",sum(train_acc)/len(train_acc) ,"loss ",sum(train_losses)/len(train_losses)) 

            train_losses=[]
            train_acc=[]
            torch.save(model.state_dict(), str(epoch)+'_'+str(step)+'.pt')

            
            
    val_losses=[]
    val_acc=[]
    model.eval()
    with torch.no_grad():
        for features,labels in val_loder:
    
          features,labels=features.to(device),labels.to(device)
          model.eval()
          result=model.forward(features)
          acc=torch.sum(torch.eq(torch.max(result,1)[1].view_as(labels),labels)).item()/v_batch_size

          val_acc.append(acc)
  
          loss=criterion(result,labels)
          val_losses.append(loss.item())
    print("epoch :",epoch,"val_acc ",sum(val_acc)/len(val_acc) ,"loss ",sum(val_losses)/len(val_losses)) 
    






















