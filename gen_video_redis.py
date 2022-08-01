import cv2
from flask import Flask, Response,send_file,request,jsonify
import flask_cors
app = Flask(__name__,static_folder='',static_url_path='')
flask_cors.CORS(app, supports_credentials=True)
import time
import threading
import io
import json
import requests
import redis 
import numpy as np
import msgpack
import msgpack_numpy as m


pool = redis.ConnectionPool(host='localhost', port=6379)
r = redis.Redis(host='localhost', port=6379)  



class ipcamCapture:
    def __init__(self, URL):
        # self.Frame = []
        self.status = False
        self.isstop = False
        self.capture = cv2.VideoCapture(URL)
    def start(self):
        print('ipcam started!')
        threading.Thread(target=self.queryframe, daemon=True, args=()).start()
    def stop(self):
        self.isstop = True
        print('ipcam stopped!')
    def getframe(self):
        return self.Frame
    def queryframe(self):
        while (not self.isstop):
            self.status, frame = self.capture.read()
            # print(frame.shape)
            
            
            
            pic_b=m.packb(frame)
            
            r.set('pp', pic_b) 

            
            
            
            
            # print(self.Frame)



cam=ipcamCapture(0) 

cam.start()

time.sleep(1)

while True:
    
   
    pic=m.unpackb(r.get('pp'))
    
    # frame = cam.getframe()
    
    
    # pic_b=frame.tobytes()
    
    # r.set('pp', pic_b) 

    # # frame = cam.getframe()
    cv2.imshow('tt',pic)

    cv2.waitKey(1)
    









