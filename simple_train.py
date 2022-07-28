from torch.utils.data import Dataset, DataLoader,random_split
import os
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from  torchvision.datasets import ImageFolder


train_transform = transforms.Compose([
        # # transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(15),
        # transforms.Resize([156,156]),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(3. / 4., 4. / 3.)),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
        transforms.ToTensor(),
    ])


val_transform = transforms.Compose([
        # transforms.Resize([128,128]),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        # transforms.RandomResizedCrop([128,128],scale=(0.8,1.0),ratio=(1,1),interpolation=2),
        # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
        transforms.ToTensor(),
    ])

train_dataset = ImageFolder('/home/xu/MyData/dogs-vs-cats/dog_cat/train/',transform=train_transform)

val_dataset = ImageFolder('/home/xu/MyData/dogs-vs-cats/dog_cat/val/',transform=val_transform)


device=torch.device('cuda:0')

# weight='/home/xu/utest/0_0.pt'


model=models.MobileNetV2().to(device)

model.load_state_dict(torch.load('/home/xu/MyDemo/DeepLearning/simple_classification/mobilenet_v2-b0353104.pth'))



for param in model.parameters():
    param.requires_grad = False
    
model.fc = nn.Sequential(*[
    nn.Dropout(p=0.2, inplace=False),
    nn.Linear(in_features=1280, out_features=2),
])

for param in model.classifier.parameters():
    param.requires_grad = True
    

model=model.to(device)

criterion=nn.CrossEntropyLoss()
            


t_batch_size=128
v_batch_size=256
train_loder=torch.utils.data.DataLoader(train_dataset,shuffle=True,batch_size=t_batch_size,num_workers=8,pin_memory=True)
val_loder=torch.utils.data.DataLoader(val_dataset,shuffle=True,batch_size=v_batch_size,num_workers=8,pin_memory=True)


criterion=nn.CrossEntropyLoss()
optimizers=torch.optim.Adam(model.parameters(),lr=1e-2)

max_val_acc=0

for epoch in range(120):
    train_losses=[]
    train_acc=[]
  


    model.train()
    for step,(features,labels) in enumerate(train_loder):

        features,labels=features.to(device),labels.to(device)
        # features/=255
        model.train()
        optimizers.zero_grad()
        result=model(features)
        
        
        # print(result.argmax(dim=1))

        acc=torch.sum(torch.eq(torch.max(result,1)[1].view_as(labels),labels)).item()/t_batch_size
        train_acc.append(acc)
    
        loss=criterion(result,labels)
        # print(loss)
        train_losses.append(loss.item())

        loss.backward()
        optimizers.step()
    
        if step%20==0:
            
            print("epoch:{} train_acc :{:.4f} train_loss: {:.4f}".format(epoch,sum(train_acc)/len(train_acc),sum(train_losses)/len(train_losses))) 

            train_losses=[]
            train_acc=[]
            # torch.save(model.state_dict(), str(epoch)+'_'+str(step)+'.pt')

            
            
    val_losses=[]
    val_acc=[]
    model.eval()
    with torch.no_grad():
        for features,labels in val_loder:
    
          features,labels=features.to(device),labels.to(device)
          # features/=255

          model.eval()
          result=model.forward(features)
          acc=torch.sum(torch.eq(torch.max(result,1)[1].view_as(labels),labels)).item()/v_batch_size

          val_acc.append(acc)
  
          loss=criterion(result,labels)
          val_losses.append(loss.item())
    print("epoch :{} val_acc :{:.4f} val_loss: {:.4f}".format(epoch,sum(val_acc)/len(val_acc),sum(val_losses)/len(val_losses))) 
    if sum(val_acc)/len(val_acc)>max_val_acc:
        max_val_acc=sum(val_acc)/len(val_acc)
        torch.save(model.state_dict(),'best.pt')
        
    








