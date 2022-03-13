import torch, torchvision
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger

import numpy as np
import os, json, cv2, random
import matplotlib.pyplot as plt

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog


im = cv2.imread('test.jpg')
# cv2_imshow(im)

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file('COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml'))
# cfg.merge_from_file('/home/xu/detectron2/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml')
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url('COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml')  # Let training initialize from model zoo


# cfg.MODEL.WEIGHTS = 'model_final_f10217.pkl'
predictor = DefaultPredictor(cfg)
outputs = predictor(im)

# print(outputs['instances'].pred_classses)

v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
v = v.draw_instance_predictions(outputs['instances'].to('cpu'))
plt.figure(figsize = (14, 10))
plt.imshow(cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
plt.savefig('tes2.jpg')
