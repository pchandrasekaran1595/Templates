import cv2
import tkinter as tk
from PIL import Image, ImageTk
from cam_and_vid import Handler

WIDTH, HEIGHT, FPS = 640, 480, 30

class VideoFrame(tk.Frame):
    def __init__(self, master, V=None, *args, **kwargs):
        tk.Frame.__init__(self, master, *args, **kwargs)

        self.master = master
        self.V = V
        self.image = None

        self.canvas = tk.Canvas(self, width=WIDTH, height=HEIGHT, background="black")
        self.canvas.pack()

        self.delay = 15
        self.id = None
    
    def update(self):
        ret, frame = self.V.getframe()
        if ret:
            self.image = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2RGB)))
            self.canvas.create_image(0, 0, anchor="nw", image=self.image)
        self.id = self.after(self.delay, self.update)
    
    def start(self):
        self.update()
    
    def stop(self):
        if self.id:
            self.after_cancel(self.id)
            self.id = None


class MainWrapper(object):
    def __init__(self, master=None, V=None):
        self.master = master
        self.V = V

        VideoWidget = VideoFrame(self.master, V=self.V)
        VideoWidget.start()
        VideoWidget.pack()

        self.master.mainloop()


def build_gui():
    root = tk.Tk()
    
    rw, rh = int(WIDTH*1.25), int(HEIGHT*1.25)
    root.geometry("{}x{}".format(rw, rh))
    root.title("Webcam Feed")

    CaptureObject = Handler(device_id=0, width=WIDTH, height=HEIGHT, fps=30)
    CaptureObject.start()
    MainWrapper(root, V=CaptureObject)
    CaptureObject.stop()

    root.mainloop()

