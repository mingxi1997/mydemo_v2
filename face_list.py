import os
import numpy as np
root='/home/xu/XU/casia-maxpy-clean/CASIA-maxpy-clean/'
dirs=[root+d for d in os.listdir(root)]
dirs.sort()
images=[]

for d in dirs:
    images.extend([d+'/'+m for m in os.listdir(d)])

record={}
c=0
labels=[]
for image in images:
    idx=int(image.split('/')[-2])
    if idx in record:
        label=record[idx]
    else:
        record[idx]=c
        label=c
        c+=1
        
    labels.append(label)
        
with open('train_list.txt','w')as f:
    for i in range(len(images)):
        f.write(images[i]+' '+str(labels[i])+'\n')
  
