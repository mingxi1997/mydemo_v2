

import torch
import torch.backends.cudnn as cudnn


from models.common import DetectMultiBackend
from utils.general import check_img_size,  cv2, non_max_suppression,  scale_coords
from utils.augmentations import letterbox

import numpy as np

device = torch.device('cuda:0')


weight='/home/xu/yolov5/yolov5/weights/head.pt'



imgsz=(640, 640)  # inference size (height, width)
conf_thres=0.6  # confidence threshold
iou_thres=0.6  # NMS IOU threshold
max_det=1000  # maximum detections per image
classes=1 # filter by class: --class 0, or --class 0 2 3
agnostic_nms=False  # class-agnostic NMS
augment=False  # augmented inference
visualize=False  # visualize features
line_thickness=3  # bounding box thickness (pixels)
hide_labels=False  # hide labels
hide_conf=False  # hide confidences
dnn=False  # use OpenCV DNN for ONNX inference
start=False
basic_frame=[]

import os
root='/home/xu/MyData/data_phone/train2/normal_images/'
nroot='/home/xu/MyData/data_phone/modify/normal/'


pics=os.listdir(root)

# cap=cv2.VideoCapture(0)
with torch.no_grad():
    
        
    model=DetectMultiBackend(weight, device=device)
   
    cudnn.benchmark = True  # set True to speed up constant image size inference
    count=0
    for c in pics:
            count+=1

            frame=cv2.imread(root+c)
            
            
    
    # while True:
    #         ret,frame=cap.read()
            stride, names, pt = model.stride, model.names, model.pt
            imgsz = check_img_size(imgsz, s=stride)  # check image size

            
            
            img = letterbox(frame, imgsz, stride=stride)[0]  
            
            
            im = torch.tensor(img).unsqueeze(0).permute(0,3,1,2).to(torch.float32).to(device)
            im /= 255  # 0 - 255 to 0.0 - 1.0
            
            pred = model(im, augment=augment, visualize=False)
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)[0]  
            
            if len(pred):
                # Rescale boxes from img_size to im0 size
                pred[:, :4] = scale_coords(im.shape[2:], pred[:, :4], frame.shape).round()
                
     
                x,y,x2,y2,conf,cls_=pred[0].cpu().numpy().astype(np.int32)
                
                w,h,c=frame.shape
                
                pic=frame[y:y2,x:x2,:]
                
                uh=y2-y
                uw=x2-x
                
                a=int(x-uw*0.5)
                b=int(x2+uw*0.5)
                
                
                if x-uw*0.5<0:
                    a=0
                    sec=int(uw*0.5-x)
                    b=int(x2+uw*0.5+sec)
                    
              
                
                pic=frame[y:y+b-a,a:b,:]
                
                cv2.imwrite(nroot+'3_'+str(count)+'.jpg',pic)
                
                
                
               
               
     
    
                cv2.imshow('tt', pic)
                cv2.waitKey(1)  # 1 millisecond




  











