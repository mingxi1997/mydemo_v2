


import torch

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_coords

import numpy as np
import time
import cv2
import threading


class ipcamCapture():
    def __init__(self, URL):
        self.Frame = []
        self.status = False
        self.isstop = False
        # 摄影机连接。
        self.capture = cv2.VideoCapture(URL)
        self.capture.set(3, 1024)  # width=1920
        self.capture.set(4, 576)  # height=1080

    def start(self):
        # 把程序放进子线程，daemon=True 表示该线程会随着主线程关闭而关闭。
        print('ipcam started!')
        threading.Thread(target=self.queryframe, daemon=True, args=()).start()

    def stop(self):
        # 记得要设计停止无限循环的开关。
        self.isstop = True
        print('ipcam stopped!')

    def getframe(self):
        # 当有需要影像时，再回传最新的影像。
        return self.Frame

    def queryframe(self):
        while (not self.isstop):
            self.status, self.Frame = self.capture.read()
            # print(self.Frame)


url=0
# # ipcam = ipcamCapture('rtsp://admin:qw123456@192.168.0.3:554/h264/ch1/main/av_stream')

ipcam = ipcamCapture(url)

ipcam.start()
time.sleep(1)



if __name__ == "__main__":
    
    weights='/home/xu/yolov5/yolov5/weights/helmet.pt'  

    source=0  # file/dir/URL/glob, 0 for webcam
    imgsz=(640, 640)  # inference size (height, width)
    conf_thres=0.6  # confidence threshold
    iou_thres=0.6  # NMS IOU threshold
    classes=0,1# filter by class: --class 0, or --class 0 2 3
   
    with torch.no_grad():
        
            
      
            source = str(source)         
            # Load model
            device = torch.device('cpu')
            
            
            
            model = DetectMultiBackend(weights, device=device)
            # stride, names, pt = model.stride, model.names, model.pt
            

            while True:
                frame=ipcam.getframe()
                
                im = torch.tensor(frame).to(torch.float32).unsqueeze(0).permute(0,3,1,2).to(device)/255
                
                pred = model(im, augment=False, visualize=False)
             
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, False, max_det=1000)[0]
                
                if len(pred):
                   
                    pred[:, :4] = scale_coords(im.shape[2:], pred[:, :4], frame.shape).round()
                    
                    for d in pred:
                        d=d.numpy().astype(np.int64)
                        x1,y1,x2,y2,conf,obj=d
                        
                        cv2.rectangle(frame, (x1,y1), (x2,y2), (255,255,0))
    
                     
                   
                    cv2.imshow('tt', frame)
                    cv2.waitKey(1)  # 1 millisecond
    

    
    
    
    
    
    
    
    
    
    
    
    
