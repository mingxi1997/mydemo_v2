
from torch.utils.data import Dataset, DataLoader,random_split
import os
import pandas as pd
import cv2
import torch
import torch.nn as nn
# import torchvision.models as models
import torchvision.transforms as transforms
from loss_functions import AngularPenaltySMLoss
from myconv import ConvNeXt
import time
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'




root_train_images="/home/xu/XU/archive/cropped_train_images/cropped_train_images/"
root_test_images="/home/xu/XU/archive/cropped_test_images/cropped_test_images/"
root_train_labels='/home/xu/XU/archive/train.csv'
root_test_labels='/home/xu/XU/archive/sample_submission.csv'

train_images=os.listdir(root_train_images)
train_labels=pd.read_csv(root_train_labels)
test_labels=pd.read_csv(root_test_labels)


train_info=train_labels['image'].values
test_info=test_labels['image'].values


names_onehot={}

my_key='species'
names=train_labels[my_key]
count=0
for i in range(len(names)):
    if names[i] not in names_onehot:
        names_onehot[names[i]]=count
        count+=1

color_aug = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)


data_transform = transforms.Compose([
        
        transforms.ToPILImage(),
        # color_aug,
        # # transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(15),
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
        if self.data_type=='train':
            # m=root_train_images+self.info[idx]
            # return m
        
            img=cv2.imread(root_train_images+self.info[idx])
            img=cv2.resize(img, (224,224))
            img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img=data_transform(img)
            label=train_labels[train_labels['image']==self.info[idx]]
            label=label[my_key].item()
            label=names_onehot[label]
            
            return img,label
        
        elif self.data_type=='test':
            img_name=self.info[idx]
            img=cv2.imread(root_test_images+self.info[idx])
            img=cv2.resize(img, (224,224))
            img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img=data_transform(img)
      
            return img,img_name
            
    
my_data=MYDataset(train_info)

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
                
        self.convlayers  =  ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768])
        self.convlayers.load_state_dict(torch.load('/home/xu/XU/my_plant/convnext_tiny_1k_224_ema.pth')['model'])
        ct=0
        for child in self.convlayers.children():
            ct += 1
            if ct < 4:
                for param in self.convlayers.parameters():
                    param.requires_grad = False
        
        
        self.output_layer = nn.Linear(1000,len(names_onehot),bias=False)

    def forward(self, x, labels=None):
        feature = self.convlayers(x)

     
        output = self.output_layer(feature)
        return feature, torch.nn.functional.log_softmax(output,dim=1)
 
net=Net(1000).cuda()
arcloss = ArcLoss( len(names_onehot),1000,).cuda()
nllloss = nn.NLLLoss(reduction="sum").cuda()
optimizer =  torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
optimizerarc =  torch.optim.Adam(arcloss.parameters()) 


train_dataset, val_dataset = random_split(
    dataset=my_data,
    lengths=[40000, 11033],
    generator=torch.Generator().manual_seed(0)
)

t_batch_size=512
v_batch_size=2048
train_loader=torch.utils.data.DataLoader(train_dataset,shuffle=True,batch_size=t_batch_size,num_workers=16,pin_memory=True)
val_loader=torch.utils.data.DataLoader(val_dataset,shuffle=True,batch_size=v_batch_size,num_workers=16,pin_memory=True)


net = nn.DataParallel(net)
arcloss=nn.DataParallel(arcloss)
# # conv_angular_pen.load_state_dict(torch.load('3save.pt'))

# # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizers, max_lr=0.1, steps_per_epoch=10, epochs=10)
best_result=[0,0]
num_epochs=120
for epoch in range(num_epochs):
    
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
            nll_loss = nllloss(ys, y)
            arcface_loss = nllloss(arc_loss, y)
            loss = nll_loss+arcface_loss
            
            loss=loss.mean()
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
            
            if step%10==0:
                print ('Epoch {}/{}, Step {}/{},  Training Loss: {:.3f}, Training Accuracy: {:.3f}'
                           .format(epoch+1,num_epochs, step,num_step, train_loss[-1], sum(train_acc)/len(train_acc)) )  
        
    
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
    
    
    
    
    

# test_data=MYDataset(test_info,'test')


# def find(item):
#     for k in names_onehot:
#         if names_onehot[k]==item:
#             return k

# # model.load_state_dict(torch.load('save.pt'))


# import csv   
# from tqdm import tqdm
# out = open("submission.csv", "w", newline='')
# writer = csv.writer(out)
# writer.writerow(['image','predictions'])    

# with torch.no_grad():

#     for img,name in tqdm(test_data):
        
#         _,y=conv_angular_pen(img.unsqueeze(0).to(device))
#         y=y.squeeze()
        
#         result=torch.topk(y,5)[1].cpu().numpy()
        
#         result=[find(item) for item in result]
#         sub=''
#         for i in range(len(result)):
#             sub+=result[i]+' '
            
#         sub=sub[:-1]
        
#         writer.writerow([name,sub])
            
# out.close()
            


















