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
import cv2
from celery.result import AsyncResult
from fastapi import Body, FastAPI, Form, Request,Header,Response
from fastapi.responses import JSONResponse,HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()
# templates = Jinja2Templates(directory="templates")
origins = [
    "*"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



with open('config.json','r')as f:
    config=json.load(f)
camerals=config['config']


def vue_to_dict(tt):
    return json.loads(list(dict(tt).keys())[0])


def single_dict_key(mydict):
      key=list(mydict.keys())[0]
      value=list(mydict.values())[0]
      return key,value

def formation_modify(formation, config):

      mid=formation[0]
      mcontent_key,mcontent_value=single_dict_key(formation[1])
    
      if mcontent_key!='algrithm':
          config[mid][mcontent_key]=mcontent_value
      else:
          key,value=single_dict_key(mcontent_value)
          if key=='allow':
              config[mid][mcontent_key][key]=value
          elif key=='main':
              sid=value[0]
              sk,sv=single_dict_key(value[1])
              config[mid][mcontent_key][key][sid][sk]=sv
      
      return {'config':config}



class ipcamCapture:
    def __init__(self, URL):
        self.Frame = []
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
            self.status, self.Frame = self.capture.read()
           

rstp_addresses=[c['url'] for c in camerals]

cams=[ipcamCapture(url) for url in rstp_addresses][:4]

for cam in cams:
    cam.start()

time.sleep(8)

print('start back')
def gen(n):
    time.sleep(1)
    while True:
        frame = cams[n].getframe()
        image = cv2.imencode('.jpg', frame)[1].tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')

@app.get('/video/{n}')
async def video_feed(n:int):
    return  StreamingResponse(gen(n),
                    media_type='multipart/x-mixed-replace; boundary=frame')


@app.get('/image/{n}')
async def image_feed(n:int):
    frame = cams[int(n)].getframe()
    is_success, buffer = cv2.imencode(".jpeg", frame)    
    file_object = io.BytesIO(buffer)
    file_object.seek(0)
    return StreamingResponse(file_object, media_type='image/jpeg')




if __name__ =="__main__":
    import uvicorn 
    
    uvicorn.run("video_b1:app", host="0.0.0.0", port=8090, log_level="info",reload=True)
        
    
    
