# YOLOv5 ğŸš€ by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage - sources:
    $ python path/to/detect.py --weights yolov5s.pt --source 0              # webcam
                                                             img.jpg        # image
                                                             vid.mp4        # video
                                                             path/          # directory
                                                             path/*.jpg     # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python path/to/detect.py --weights yolov5s.pt                 # PyTorch
                                         yolov5s.torchscript        # TorchScript
                                         yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                         yolov5s.xml                # OpenVINO
                                         yolov5s.engine             # TensorRT
                                         yolov5s.mlmodel            # CoreML (macOS-only)
                                         yolov5s_saved_model        # TensorFlow SavedModel
                                         yolov5s.pb                 # TensorFlow GraphDef
                                         yolov5s.tflite             # TensorFlow Lite
                                         yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync


import requests



if __name__ == "__main__":
    
    weights='best.pt'  # model.pt path(s)
    source=0  # file/dir/URL/glob, 0 for webcam
    data=ROOT / 'data/coco128.yaml' # dataset.yaml path
    imgsz=(640, 640)  # inference size (height, width)
    conf_thres=0.25  # confidence threshold
    iou_thres=0.45  # NMS IOU threshold
    max_det=1000  # maximum detections per image
    device=''  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    view_img=False  # show results
    save_txt=False  # save results to *.txt
    save_conf=False  # save confidences in --save-txt labels
    save_crop=False  # save cropped prediction boxes
    nosave=False  # do not save images/videos
    classes=None  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False  # class-agnostic NMS
    augment=False  # augmented inference
    visualize=False  # visualize features
    update=False  # update all models
    project=ROOT / 'runs/detect'  # save results to project/name
    name='exp' # save results to project/name
    exist_ok=False  # existing project/name ok, do not increment
    line_thickness=3  # bounding box thickness (pixels)
    hide_labels=False  # hide labels
    hide_conf=False  # hide confidences
    half=False  # use FP16 half-precision inference
    dnn=False  # use OpenCV DNN for ONNX inference
    
    with torch.no_grad():
        
            
      
            source = str(source)
               # Load model
            device = torch.device('cpu')
            
            
            
            model = DetectMultiBackend(weights, device=device,  data=data)
            stride, names, pt = model.stride, model.names, model.pt
            
            imgsz = check_img_size(imgsz, s=stride)  # check image size
    
            # Dataloader
        
            view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
            bs = len(dataset)  # batch_size
     
            vid_path, vid_writer = [None] * bs, [None] * bs
    
            # Run inference
            # model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
            seen, windows, dt = 0, [], [0.0, 0.0, 0.0]
            for path, im, im0s, vid_cap, _ in dataset:
                t1 = time_sync()
                im = torch.tensor(im).to(torch.float32).to(device)
                
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim
                t2 = time_sync()
                dt[0] += t2 - t1
    
       
                pred = model(im, augment=augment, visualize=False)
                t3 = time_sync()
                dt[1] += t3 - t2
    
                # NMS
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
                print(pred[0].numpy())
                
                
                boxes = pred[0].numpy()
                if boxes is not None: 
                    
                    out_data='none'
                    out_data=''
                    # print(boxes)
                    
                    for box in boxes:
                        for i in range(len(box)):
                            if i!=4:
                                out_data+=str(int(box[i]))+' '
                    print(out_data)
                    
                    # def generate_report(out_data):
                    #     format = out_data

                    # with app.test_request_context(
                    #         'http://127.0.0.1:8088/boxes_storage', data='format'):
                    #     generate_report(out_data)

                    # r = requests.post('http://127.0.0.1:8088/boxes_storage',data=out_data)
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                dt[2] += time_sync() - t3
    
           
                for i, det in enumerate(pred):  # per image
                    seen += 1
                    # if webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                        # s += f'{i}: '
                    # else:
                    #     p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
    
                    p = Path(p)  # to Path
                    # save_path = str(save_dir / p.name)  # im.jpg
                    # txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
                    # s += '%gx%g ' % im.shape[2:]  # print string
                    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                    imc = im0.copy() if save_crop else im0  # for save_crop
                    annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                    if len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
    
                        # Print results
                        for c in det[:, -1].unique():
                            n = (det[:, -1] == c).sum()  # detections per class
                            # s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
    
                        # Write results
                        for *xyxy, conf, cls in reversed(det):
                            if save_txt:  # Write to file
                                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                                line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                              

                            if  save_crop or view_img:  # Add bbox to image
                                c = int(cls)  # integer class
                                label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                                annotator.box_label(xyxy, label, color=colors(c, True))
                           
                    # Stream results
                    im0 = annotator.result()
                    if view_img:
                        if p not in windows:
                            windows.append(p)
                            cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                            cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                        cv2.imshow(str(p), im0)
                        cv2.waitKey(1)  # 1 millisecond
    
                    # Save results (image with detections)
              
                # Print time (inference-only)
                # LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')
    
            # Print results
            # t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
            # LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
           
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
