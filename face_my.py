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
from sklearn.preprocessing import LabelEncoder
from myconv import ConvNeXt
from tqdm import tqdm
le = LabelEncoder()
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

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

root_images="/home/xu/casia-maxpy-clean/CASIA-maxpy-clean/"

dirs=[root_images+d+'/' for d in os.listdir(root_images)]
nums_output=len(dirs)
images=[]
labels=[]
for d in dirs:
    images.extend([d+imgs for imgs in os.listdir(d)])
for img in images:
    labels.append(img.split('/')[-2])
labels=le.fit_transform(labels)
# refine_labels=[]
# for l in labels:
    
    
    
    

data_transform = transforms.Compose([
        
        transforms.ToPILImage(),
        # color_aug,
        
        transforms.Resize(224),
        # transforms.CenterCrop(224),
        # transforms.RandomCrop(224),
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

# train_dataset, val_dataset = random_split(
#     dataset=my_data,
#     lengths=[int(len(labels)*0.8), len(labels)-int(len(labels)*0.8)],
#     generator=torch.Generator().manual_seed(0)
# )

device=torch.device('cuda')






class Net(nn.Module):
    def __init__(self, num_classes, loss_type='arcface'):
        super().__init__()
        
        # self.convlayers  =  ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024])
    
        self.convlayers  =  ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768])
        
        self.output_layer = nn.Linear(1000,num_classes,bias=False)

    def forward(self, x, labels=None):
        
        feature = self.convlayers(x)
      
        # feature=feature.reshape(-1,1000)
        output = self.output_layer(feature)
        return output
import  math
class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False, ls_eps=0.0):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.ls_eps = ls_eps  # label smoothing
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))

        # fix nan problem:
        sine = torch.sqrt(torch.clamp((1.0 - torch.pow(cosine, 2)),1e-9,1))

        phi = cosine * self.cos_m - sine * self.sin_m
        
        # print(cosine)
        # phi=torch.cos(torch.acos(cosine)+self.m)
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output
# net=Net(len(dirs)).cuda()
net=Net(512).cuda()

arc_margin=ArcMarginProduct(512,len(dirs)).cuda()
# arcloss = ArcLoss( len(dirs),1000).cuda()
# nllloss = nn.NLLLoss(reduction="sum").cuda()
optimizer =  torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0005)
optimizerarc =  torch.optim.Adam(arc_margin.parameters()) 


t_batch_size=60
# v_batch_size=60
train_loader=torch.utils.data.DataLoader(my_data,shuffle=True,batch_size=t_batch_size,num_workers=16,pin_memory=True)
# val_loader=torch.utils.data.DataLoader(val_dataset,shuffle=True,batch_size=v_batch_size,num_workers=16,pin_memory=True)

focalloss=FocalLoss()
net = nn.DataParallel(net)
arc_margin=nn.DataParallel(arc_margin)


# scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
#             optimizer, mode='max', factor=0.1, patience=3, verbose=True,
#             threshold=1e-4)

scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

best_result=[0,0]
num_epochs=100
for epoch in range(num_epochs):
    

    
    num_step=len(train_loader)
    for step, (x, y) in enumerate(train_loader):

            train_loss=[]
            train_acc=[]
            x = x.cuda()
            y = y.cuda()
            ys = net(x) 
            
            ############ ACCURACY CAL. ###########
         
            ################ LOSS CAL. ################
            
            # arc_loss =  torch.log(arcloss(xs))
            
            # focal_loss = focalloss(ys, y)
            
            # arcface_loss = nllloss(arc_loss, y)
            # loss = focal_loss+arcface_loss
            
            # loss=loss.mean()
            # iter_loss = loss.item()
            ###########################################
            m=arc_margin(ys,y)
            loss= focalloss(m, y)
            
            iter_loss = loss.item()

            # value =  torch.argmax(m, dim=1)
            # acc =  torch.sum((value == y).float()) / len(y)
               
            
            
            
            
            ############ BACK PROP #################
            optimizer.zero_grad()
            optimizerarc.zero_grad()
            loss.backward()
            optimizer.step()
            optimizerarc.step()
            
            train_loss.append(loss)
            # train_acc.append(acc)
            
            if step%100==0:
                print ('Epoch {}/{}, Step {}/{},  Training Loss: {:.3f}'
                            .format(epoch+1,num_epochs, step,num_step, sum(train_loss)/len(train_loss)) )  
        
    scheduler.step()

    # test_loss=[]
    # test_acc=[]
    
    # with torch.no_grad():
    #         for step, (x, y) in enumerate(val_loader):

                
    #             x = x.cuda()
    #             y = y.cuda()
    #             ys = net(x) 
    #             m=arc_margin(ys,y)
    #             ############ ACCURACY CAL. ###########
    #             value =  torch.argmax(m, dim=1)
    #             acc =  torch.sum((value == y).float()) / len(y)
                
  

    #             test_acc.append(acc)
    # # scheduler.step(acc)
    # print ('Epoch {}/{},test Accuracy: {:.3f}'
    #             .format(epoch+1, num_epochs, sum(test_acc)/len(test_acc)) )
    
    # if sum(test_acc)/len(test_acc)>best_result[0]:
    #     best_result=[sum(test_acc)/len(test_acc),epoch]
    #     torch.save(net.state_dict(), './face_save/face_best{}.pt'.format(epoch))
    # print('best acc :{} best_epoch :{}'.format(best_result[0],best_result[1]))
    
    
