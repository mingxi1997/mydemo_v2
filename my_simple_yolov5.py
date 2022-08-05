
import torch

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_coords,check_img_size

import numpy as np
import time
import cv2
from utils.augmentations import letterbox


url='/home/xu/Pictures/1 行人检测测试视频.flv'
# # ipcam = ipcamCapture('rtsp://admin:qw123456@192.168.0.3:554/h264/ch1/main/av_stream')

cap=cv2.VideoCapture(url)



if __name__ == "__main__":
    
    weights='/home/xu/ai_box/detect/weights/head.pt'  

    imgsz=(1280, 1280)  # inference size (height, width)
    conf_thres=0.6  # confidence threshold
    iou_thres=0.6  # NMS IOU threshold
    classes=None# filter by class: --class 0, or --class 0 2 3
   
    with torch.no_grad():
        
        device = torch.device('cuda:0')
        model = DetectMultiBackend(weights, device=device)
        stride, names, pt = model.stride, model.names, model.pt
        imgsz = check_img_size(imgsz, s=stride) 
        

        while True:
            ret,frame=cap.read()
            
            imgsz = check_img_size(imgsz, s=stride) 

            
            frame=letterbox(frame, imgsz, stride=stride)[0]
            
            im = torch.tensor(frame).to(torch.float32).unsqueeze(0).permute(0,3,1,2).to(device)/255
            
            pred = model(im, augment=False, visualize=False)
         
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, False, max_det=1000)[0]
            
            if len(pred):
                print(len(pred))
               
                pred[:, :4] = scale_coords(im.shape[2:], pred[:, :4], frame.shape).round()
                
                for d in pred:
                    print(d)
                    d=d.cpu().numpy().astype(np.int64)
                    x1,y1,x2,y2,conf,obj=d
                    
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (255,255,0))

                 
               
            cv2.imshow('tt', frame)
            cv2.waitKey(1)  # 1 millisecond


    
    
    
    
    
    
    
    
    
    
    
    

    
    
    
    
    
    
