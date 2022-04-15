
from torch.utils.data import Dataset, DataLoader,random_split
import os
import pandas as pd
import cv2
import torch
import torch.nn as nn
# import torchvision.models as models
import torchvision.transforms as transforms
import time
import torchvision
import torch.nn.functional as F

class FocalLoss(nn.Module):
    '''Multi-class Focal loss implementation'''
    def __init__(self, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        """
        input: [N, C]
        target: [N, ]
        """
        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        logpt = (1-pt)**self.gamma * logpt
        loss = F.nll_loss(logpt, target, self.weight)
        return loss

root_images="/home/xu/MyData/CUB200(2011)/CUB_200_2011/CUB_200_2011/images/"

dirs=[root_images+d+'/' for d in os.listdir(root_images)]

images=[]
labels=[]
for d in dirs:
    images.extend([d+imgs for imgs in os.listdir(d)])
for img in images:
    labels.append(int(img.split('/')[-2].split('.')[0])-1)
    
    

data_transform = transforms.Compose([
        
        transforms.ToPILImage(),
        # color_aug,
        
        transforms.Resize(224),
        # transforms.CenterCrop(224),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(15),
        # transforms.RandomResizedCrop(224),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


class MYDataset(Dataset):
    def __init__(self):
       
        self.images=images
        self.labels=labels
      
        
    def __len__(self):
     
        return len(self.images)
    
    def __getitem__(self, idx):
        
        img_root=self.images[idx]
        label=self.labels[idx]
        img=cv2.imread(img_root)
        # img=cv2.resize(img, (224,224))
        img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img=data_transform(img)

        return img,label

my_data=MYDataset()

train_dataset, val_dataset = random_split(
    dataset=my_data,
    lengths=[9788, 2000],
    generator=torch.Generator().manual_seed(0)
)

device=torch.device('cuda')



class ArcLoss(nn.Module):
    def __init__(self,class_num,feature_num,s=30,m=0.5):
        super().__init__()
        self.class_num=class_num
        self.feature_num=feature_num
        self.s = s
        self.m =  torch.tensor(m)
        self.w=nn.Parameter( torch.rand(feature_num,class_num))
    #2*10
    def forward(self,feature):
        feature = torch.nn.functional.normalize(feature,dim=1) 
        w = torch.nn.functional.normalize(self.w,dim=0)
        cos_theat =  torch.matmul(feature,w)/self.s
 
        
        cos_theat_m = torch.cos(torch.acos(cos_theat)+self.m)
        
        cos_theat_ =  torch.exp(cos_theat * self.s)
        sum_cos_theat = torch.sum( torch.exp(cos_theat*self.s),dim=1,keepdim=True)-cos_theat_
        
        top =  torch.exp(cos_theat_m*self.s)
        divide = (top/(top+sum_cos_theat))

        return divide



class Net(nn.Module):
    def __init__(self, num_classes, loss_type='arcface'):
        super().__init__()
        # self.convlayers = model
                
        self.convlayers  = torchvision.models.convnext_tiny(pretrained=False)
        self.convlayers .load_state_dict(torch.load('/home/xu/MyDemo/DeepLearning/my_plant/convnext_tiny.pth'))
        # self.convlayers.load_state_dict(torch.load('/home/xu/XU/my_plant/convnext_tiny_1k_224_ema.pth')['model'])
        ct=0
        for child in self.convlayers.children():
            ct += 1
            if ct < 4:
                for param in self.convlayers.parameters():
                    param.requires_grad = False
        
        
        self.output_layer = nn.Linear(1000,num_classes,bias=False)

    def forward(self, x, labels=None):
        feature = self.convlayers(x)

     
        output = self.output_layer(feature)
        return feature, output
 
net=Net(1000).cuda()
arcloss = ArcLoss( 200,1000,).cuda()
nllloss = nn.NLLLoss(reduction="sum").cuda()
optimizer =  torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizerarc =  torch.optim.Adam(arcloss.parameters()) 


t_batch_size=512
v_batch_size=512
train_loader=torch.utils.data.DataLoader(train_dataset,shuffle=True,batch_size=t_batch_size,num_workers=16,pin_memory=True)
val_loader=torch.utils.data.DataLoader(val_dataset,shuffle=True,batch_size=v_batch_size,num_workers=16,pin_memory=True)

focalloss=FocalLoss()
# net = nn.DataParallel(net)
# arcloss=nn.DataParallel(arcloss)

lr_func = lambda epoch: 0.99 ** epoch
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)


best_result=[0,0]
num_epochs=120
for epoch in range(num_epochs):
    scheduler.step()
    
    train_loss=[]
    train_acc=[]
    
    num_step=len(train_loader)
    for step, (x, y) in enumerate(train_loader):

         
            x = x.cuda()
            y = y.cuda()
            xs, ys = net(x) 
            
            ############ ACCURACY CAL. ###########
            value =  torch.argmax(ys, dim=1)
            acc =  torch.sum((value == y).float()) / len(y)
            
            ################ LOSS CAL. ################
            arc_loss =  torch.log(arcloss(xs))
            
            focal_loss = focalloss(ys, y)
            
            arcface_loss = nllloss(arc_loss, y)
            loss = focal_loss+arcface_loss
            
            # loss=loss.mean()
            iter_loss = loss.item()
            ###########################################
            
            ############ BACK PROP #################
            optimizer.zero_grad()
            optimizerarc.zero_grad()
            loss.backward()
            optimizer.step()
            optimizerarc.step()
            
            train_loss.append(loss)
            train_acc.append(acc)
            
            if step%1==0:
                print ('Epoch {}/{}, Step {}/{},  Training Loss: {:.3f}, Training Accuracy: {:.3f}'
                            .format(epoch+1,num_epochs, step,num_step, sum(train_loss)/len(train_loss), sum(train_acc)/len(train_acc)) )  
        
    
    test_loss=[]
    test_acc=[]
    
    with torch.no_grad():
            for step, (x, y) in enumerate(val_loader):

                
                x = x.cuda()
                y = y.cuda()
                xs, ys = net(x) 
                
                ############ ACCURACY CAL. ###########
                value =  torch.argmax(ys, dim=1)
                acc =  torch.sum((value == y).float()) / len(y)
                
  

                test_acc.append(acc)
    print ('Epoch {}/{},test Accuracy: {:.3f}'
                .format(epoch+1, num_epochs, sum(test_acc)/len(test_acc)) )
    
    if sum(test_acc)/len(test_acc)>best_result[0]:
        best_result=[sum(test_acc)/len(test_acc),epoch]
    print('best acc :{} best_epoch :{}'.format(best_result[0],best_result[1]))
    
    
