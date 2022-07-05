import cv2
from flask import Flask, render_template, Response,send_file
import flask_cors
import random
app = Flask(__name__,static_folder='',static_url_path='')
def gen():
    
    # cap = cv2.VideoCapture('test.mp4')
    # cap = cv2.VideoCapture("rtsp://192.168.0.119:554/user=admin&password=&channel=1&stream=0.sdp?")
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        
       
        image = cv2.imencode('.jpg', frame)[1].tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')



if __name__ == '__main__':
    app.run()
