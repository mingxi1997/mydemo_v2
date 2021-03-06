
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision 
import torch.utils.data as data
import matplotlib.pyplot as plt
import numpy as np
import  math
no_plus=False
print('no_plus :{}'.format(no_plus))


theta_clip=False

print('theta_clip :{}'.format(theta_clip))

theta_clip_and_focalloss=False
print('theta_clip_and_focalloss :{}'.format(theta_clip_and_focalloss))

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
    
    
    
class ArcLoss(nn.Module):
    def __init__(self,feature_num,class_num,s=30,m=0.5):
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
        cos_theat_m = torch.cos(torch.acos(cos_theat)+self.m)        
        cos_theat_ =  torch.exp(cos_theat * self.s)
        sum_cos_theat = torch.sum( torch.exp(cos_theat*self.s),dim=1,keepdim=True)-cos_theat_       
        top =  torch.exp(cos_theat_m*self.s)
        divide = (top/(top+sum_cos_theat))

        return divide
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
        self.fc=nn.Linear(3,10)
    def forward(self, xs):
        feat = self.hidden_layer(xs) 
        feature = feat.reshape(-1,16*4*4)
        feature = self.linear_layer(feature)
        out=self.fc(feature)
        return feature,out
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
      
arcloss = ArcLoss(3,10).cuda()
pro_arcloss = ArcMarginProduct(3, 10).cuda()


nllloss= nn.NLLLoss(reduction="sum").cuda()
focalloss=FocalLoss(gamma=0)


optmizer =  torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
if theta_clip:
    optmizerarc =  torch.optim.Adam(pro_arcloss.parameters())
else:
    optmizerarc =  torch.optim.Adam(arcloss.parameters())

save_pic_path=""

best_acc=0

if __name__ == '__main__':

    num_epochs = 20
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

          
            x = x.cuda()
            y = y.cuda()
            xs,ys = net(x) 
            
            if no_plus:
                loss=nn.CrossEntropyLoss()(ys,y)
            else:
            
            
                if theta_clip:
                    arc_loss= pro_arcloss(xs,y)
                    
                    if theta_clip_and_focalloss:
                        loss = focalloss(arc_loss, y)
                    else:
                        loss = nn.CrossEntropyLoss()(arc_loss, y)
    
                else:
                    arc_loss =  torch.log(arcloss(xs))
                    arcface_loss = nllloss(arc_loss, y)
                    loss = arcface_loss
                    
                
            
            iter_loss = loss.item() 

            ############ ACCURACY CAL. ###########
            if no_plus:
                   value =  torch.argmax(ys, dim=1)
            else:
                   if theta_clip:
                       arc_loss= pro_arcloss(xs,y)
                   else:
                       arc_loss =  torch.log(arcloss(xs))
                   value =  torch.argmax(arc_loss, dim=1)
            acc =  torch.sum((value == y).float()) / len(y)
            ############ BACK PROP #################
            if no_plus:
                optmizer.zero_grad()
                loss.backward()
                optmizerarc.step()
            else:
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
        print ('Epoch {}/{}, Training Loss: {:.5f}, Training Accuracy: {:.3f}'
           .format(epoch+1, num_epochs, train_loss[-1], sum(train_accuracy)/len(train_accuracy)))    
        
        test_accuracy = []

        with torch.no_grad():
            for i, (x, y) in enumerate(test_loader):

             
                x = x.cuda()
                y = y.cuda()
                xs,ys = net(x) 
                
                # ################ ARCFACE PRODUCT LOSS CAL. ################
              

                ############ ACCURACY CAL. ###########
                if no_plus:
                    value =  torch.argmax(ys, dim=1)
                else:
                    if theta_clip:
                        arc_loss= pro_arcloss(xs,y)
                    else:
                        arc_loss =  torch.log(arcloss(xs))
                    value =  torch.argmax(arc_loss, dim=1)
                acc =  torch.sum((value == y).float()) / len(y)
                
  

                test_accuracy.append(acc)
            print ('Epoch {}/{},test Accuracy: {:.3f}'
                .format(epoch+1, num_epochs, sum(test_accuracy)/len(test_accuracy)) )
        if best_acc <sum(test_accuracy)/len(test_accuracy):
            best_acc=sum(test_accuracy)/len(test_accuracy)
    
    
    
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
    import plotly.express as px
    
    tsne_df = pd.DataFrame(
        np.column_stack((embs, y)),
        columns = ["x","y","z","targets"]
    
    )

    fig = px.scatter_3d(tsne_df, x='x', y='y', z='z',
                  color='targets')
    fig.show()
            
    
print('best_acc:{}'.format(best_acc))    
print('no_plus :{}'.format(no_plus))
print('theta_clip :{}'.format(theta_clip))
print('theta_clip_and_focalloss :{}'.format(theta_clip_and_focalloss))
    
    
    
    
    
    
    
    
    
    
    
    
    
