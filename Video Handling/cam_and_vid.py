import cv2
import platform


class Handler(object):
    def __init__(self, device_id=None, path=None, width=None, height=None, fps=None):
        self.device_id = device_id
        self.path = path
        self.width = width
        self.height = height
        self.fps = fps
    
    def start(self):
        if self.device_id is not None and self.path is None:
            if platform.system() != "Windows":
                self.vid = cv2.VideoCapture(self.device_id)
            else:
                self.vid = cv2.VideoCapture(self.device_id, cv2.CAP_DSHOW)
            self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.vid.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.vid.set(cv2.CAP_PROP_FPS, self.fps)
        elif self.device_id is None and self.path:
            self.vid = cv2.VideoCapture(self.path)
        elif self.device_id is None and self.path is None:
            raise ValueError("Enter either a device id or a path to a video file")

    def stop(self):
        if self.vid.isOpened():
            self.vid.release()

    def getframe(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                return ret, frame
            else:
                return ret, None
    
    def display(self):
        while True:
            ret, frame = self.getframe()
            if self.device_id:
                if ret:
                    cv2.imshow("Feed", frame)
                    if cv2.waitKey(1) == ord("q"):
                        break
                else:
                    break
            else:
                if ret:
                    if self.width is not None:
                        frame = cv2.resize(src=frame, dsize=(self.width, int(self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))), interpolation=cv2.INTER_AREA)
                    elif self.height is not None:
                        frame = cv2.resize(src=frame, dsize=(int(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)), self.height), interpolation=cv2.INTER_AREA)
                    elif self.width is not None and self.height is not None:
                        frame = cv2.resize(src=frame, dsize=(self.width, self.height), interpolation=cv2.INTER_AREA)
                    cv2.imshow("Video", frame)
                    if cv2.waitKey(15) == ord("q"):
                        break
                else:
                    self.vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
