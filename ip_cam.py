class ipcamCapture:
    def __init__(self, url):
        self.url=url
        self.Frame = []
        self.status = False
        self.isstop = False
        self.capture = cv2.VideoCapture(self.url)
    def start(self):
        print('ipcam started!')
        self.thread=threading.Thread(target=self.queryframe, daemon=False, args=())
        self.thread.start()
        time.sleep(1)

    def stop(self):
        self.isstop = True
        self.thread.join()
        print('ipcam stopped!')
    def getframe(self):
        return self.status,self.Frame
    def queryframe(self):
        while True:
            self.status, self.Frame = self.capture.read()
            if self.isstop:
                break
    def reconnect(self):
        self.isstop = True
        self.thread.join()
        self.capture.release()
        time.sleep(2)
        self.capture=cv2.VideoCapture(self.url)
        self.isstop = False
        self.start()
        time.sleep(1)
