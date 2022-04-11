
from torch.utils.data import Dataset, DataLoader,random_split
import os
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
root="/home/xu/MyDemo/DeepLearning/cvpr2020-plant-pathology/data/"
root_images=root+"images/"
root_train_labels=root+'train.csv'
root_test_labels=root+'test.csv'




images=os.listdir(root_images)
train_labels=pd.read_csv(root_train_labels)
test_labels=pd.read_csv(root_test_labels)


train_info=train_labels['image_id'].values
test_info=test_labels['image_id'].values


data_transform = transforms.Compose([
        # transforms.RandomResizedCrop(224),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


class MYDataset(Dataset):
    def __init__(self,info,data_type='train'):
       
        self.info=info
        self.data_type=data_type
        
    def __len__(self):
     
        return len(self.info)
    
    def __getitem__(self, idx):
        
        
        img=cv2.imread(root_images+self.info[idx]+'.jpg')
        img=cv2.resize(img, (224,224))
        img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img=data_transform(img)
        # img = img[:, :, ::-1].transpose(2, 0, 1)
        if self.data_type=='train':
        
           
            label=train_labels[train_labels['image_id']==self.info[idx]]
            
            if label['healthy'].item()==1:
                l=0
            elif label['multiple_diseases'].item()==1:
                l=1
            elif label['rust'].item()==1:
                l=2
            elif label['scab'].item()==1:
                l=3
            
            return img,l
        
        elif self.data_type=='test':

            return img
            
    
my_data=MYDataset(train_info)

device=torch.device('cuda:0')






model = models.convnext_tiny(pretrained=False)
model.load_state_dict(torch.load('convnext_tiny.pth'))


model_list=list(model.children())[:-1]

model_head=torch.nn.Sequential(
    # torch.nn.LayerNorm((768,), eps=1e-06, elementwise_affine=True),
    torch.nn.Flatten(start_dim=1, end_dim=-1),
    torch.nn.Linear(in_features=768, out_features=4, bias=True)
    
    )

model_list.append(model_head)


model=torch.nn.Sequential(*model_list)


model=model.to(device)

criterion=nn.CrossEntropyLoss()
optimizers=torch.optim.Adam(model.parameters(),lr=1e-3)
            


train_dataset, val_dataset = random_split(
    dataset=my_data,
    lengths=[1600, 221],
    generator=torch.Generator().manual_seed(0)
)

batch_size=10
train_loder=torch.utils.data.DataLoader(train_dataset,shuffle=True,batch_size=batch_size,num_workers=4,pin_memory=True)
val_loder=torch.utils.data.DataLoader(val_dataset,shuffle=True,batch_size=10,num_workers=4,pin_memory=True)


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

        acc=torch.sum(torch.eq(torch.max(result,1)[1].view_as(labels),labels)).item()/batch_size
        train_acc.append(acc)
    
        loss=criterion(result,labels)
        # print(loss)
        train_losses.append(loss.item())

        loss.backward()
        optimizers.step()
    
     
    print("epoch :",epoch,"train_acc ",sum(train_acc)/len(train_acc) ,"loss ",sum(train_losses)/len(train_losses)) 


    val_losses=[]
    val_acc=[]
    model.eval()
    with torch.no_grad():
        for features,labels in val_loder:
    
          features,labels=features.to(device),labels.to(device)
          model.eval()
          result=model.forward(features)
          acc=torch.sum(torch.eq(torch.max(result,1)[1].view_as(labels),labels)).item()/batch_size
          val_acc.append(acc)
    
    
          loss=criterion(result,labels)
          val_losses.append(loss.item())
    print("epoch :",epoch,"val_acc ",sum(val_acc)/len(val_acc) ,"loss ",sum(val_losses)/len(val_losses)) 
    






















