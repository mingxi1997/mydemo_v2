
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision 
import torch.utils.data as data
import matplotlib.pyplot as plt
import torch.optim.lr_scheduler as lr_scheduler
import os

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
        feature = F.normalize(feature,dim=1) #128*2
        w = F.normalize(self.w,dim=0) #2*10
        cos_theat =  torch.matmul(feature,w)/self.s
 
        # sin_theat =  torch.sqrt(1.0- torch.pow(cos_theat,2))
        # cos_theat_m = cos_theat* torch.cos(self.m) - sin_theat* torch.sin(self.m)
        
        cos_theat_m = torch.cos(torch.acos(cos_theat)+arcloss.m)
        
        cos_theat_ =  torch.exp(cos_theat * self.s)
        sum_cos_theat = torch.sum( torch.exp(cos_theat*self.s),dim=1,keepdim=True)-cos_theat_
        
        
        top =  torch.exp(cos_theat_m*self.s)
        divide = (top/(top+sum_cos_theat))

        return divide
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_layer = nn.Sequential(
            nn.Conv2d(1, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64,256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256,256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256,64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64,16,3, 2, 1),
            nn.ReLU())
        self.linear_layer = nn.Linear(16*4*4,3)
        self.output_layer = nn.Linear(3,10,bias=False)
    def forward(self, xs):
        feat = self.hidden_layer(xs)
        # print(feature.shape)
        fc = feat.reshape(-1,16*4*4)
        # print(fc.data.size())
        feature = self.linear_layer(fc)
        output = self.output_layer(feature)
        return feature, F.log_softmax(output,dim=1)
def decet(feature,targets,epoch):
    color = ["red", "black", "yellow", "green", "pink",
    "gray", "lightgreen", "orange", "blue", "teal"]
    cls = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    plt.ion()
    plt.clf()
    for j in cls:
        mask = [targets == j]
        feature_ = feature[mask].numpy()
        x = feature_[:, 1]
        y = feature_[:, 0]
        label = cls
        plt.plot(x, y, ".", color=color[j])
        plt.legend(label, loc="upper right") 
        plt.title("epoch={}".format(str(epoch+1)))
    plt.savefig('./save/{}.jpg'.format(epoch+1))
    plt.draw()
    plt.pause(0.01)
    
Batch_Size = 128

transforms =torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize((0.1307,), (0.3081,))])

train_data = torchvision.datasets.MNIST('./data', train=True, download=True,
                   transform=transforms)

test_data = torchvision.datasets.MNIST('./data', train=False,
                   transform=transforms)

train_loader = data.DataLoader(train_data,
                batch_size=Batch_Size, shuffle=True,
                drop_last=True,num_workers=2)

test_loader = data.DataLoader(dataset = test_data, 
                            batch_size = Batch_Size,
                            shuffle = False)
net = CNN()
CUDA = torch.cuda.is_available()
if CUDA:
    net = net.cuda()    
      
arcloss = ArcLoss(10, 3).cuda()
nllloss = nn.NLLLoss(reduction="sum").cuda()
optmizer =  torch.optim.SGD(net.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.0005)
optmizerarc =  torch.optim.Adam(arcloss.parameters())


save_pic_path=""
if __name__ == '__main__':

    num_epochs = 50
    # net = net.to(device)
    for epoch in range(num_epochs):
        train_loss = []
        test_loss = []
        train_accuracy = []
        correct = 0
        iterations = 0
        iter_loss = 0.0
        feat = []
        target = []
        net.train()  
        feat = []
        target = []
        for i, (x, y) in enumerate(train_loader):

            if CUDA:
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
            iter_loss = loss.item()
            ###########################################
            
            ############ BACK PROP #################
            optmizer.zero_grad()
            optmizerarc.zero_grad()
            loss.backward()
            optmizer.step()
            optmizerarc.step()
            ########################################
            feat.append(xs) 
            target.append(y)
            iterations += 1 

        
        features =  torch.cat(feat, 0)
        targets =  torch.cat(target, 0)
        decet(features.data.cpu(), targets.data.cpu(),epoch)
        train_loss.append(iter_loss/ iterations)
        train_accuracy.append(acc)
        print ('Epoch {}/{}, Training Loss: {:.3f}, Training Accuracy: {:.3f}'
           .format(epoch+1, num_epochs, train_loss[-1], sum(train_accuracy)/len(train_accuracy)))    
        
        test_accuracy = []

        with torch.no_grad():
            for i, (x, y) in enumerate(test_loader):

                if CUDA:
                    x = x.cuda()
                    y = y.cuda()
                xs, ys = net(x) 
                
                ############ ACCURACY CAL. ###########
                value =  torch.argmax(ys, dim=1)
                acc =  torch.sum((value == y).float()) / len(y)
                
  

                test_accuracy.append(acc)
            print ('Epoch {}/{},test Accuracy: {:.3f}'
                .format(epoch+1, num_epochs, sum(test_accuracy)/len(test_accuracy)) )
    
    
    
    
    PATH = "model.pth"
    torch.save(net.state_dict(),PATH)
    
    
    
    
    net.load_state_dict(torch.load(PATH))
    net.eval()
    
    from tqdm import tqdm

    emb = []
    y = []
    
    with torch.no_grad():
        for images,labels in tqdm(test_loader):
            
            images = images.cuda()
            embeddings = net(images)[0]
            
            emb += [embeddings.detach().cpu()]
            y += [labels]
            
        embs = torch.cat(emb).cpu().numpy()
        y = torch.cat(y).cpu().numpy()
    import pandas as pd
    import numpy as np
    import plotly.express as px
    
    tsne_df = pd.DataFrame(
        np.column_stack((embs, y)),
        columns = ["x","y","z","targets"]
    
    )

    fig = px.scatter_3d(tsne_df, x='x', y='y', z='z',
                  color='targets')
    fig.show()
            
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
