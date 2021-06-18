# FIX: Do not initialize capture object within the init



import sys
import cv2
import platform
import tkinter as tk
from PIL import Image, ImageTk

WIDTH, HEIGHT = 640, 360

class Video:
    def __init__(self, device_id=0, width=640, height=360, fps=30):
        self.device_id = device_id
        self.width = width
        self.height = height
        self.fps = fps

        if platform.system() != "Windows":
            self.vid = cv2.VideoCapture(self.device_id)
        else:
            self.vid = cv2.VideoCapture(self.device_id, cv2.CAP_DSHOW)
        self.vid.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.vid.set(cv2.CAP_PROP_FPS, self.fps)
    
    def __getframe__(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                return ret, cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2RGB)
            else:
                return ret, None
    
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()

# Better to keep this global. Allows to be used in multiple tkinter frames.
VID = Video(width=WIDTH, height=HEIGHT)


class VideoFrame(tk.Frame):
    def __init__(self, master, V=None, bg=None, canvas_width=None, canvas_height=None, *args, **kwargs):
        tk.Frame.__init__(self, master, background=bg, *args, **kwargs)

        self.master = master

        self.image = None
        self.V = V
        self.canvas = tk.Canvas(self, background=bg, width=canvas_width, height=canvas_height)
        self.canvas.pack()

        self.delay = 15
        self.after(self.delay, self.update)
    
    def update(self):
        ret, frame = self.V.__getframe__()
        if ret:
            self.image = ImageTk.PhotoImage(Image.fromarray(frame))
            self.canvas.create_image(0, 0, anchor="nw", image=self.image)
        self.after(self.delay, self.update)




############################################# Example #############################################

def app():
    root = tk.Tk()

    root.geometry("{}x{}".format(WIDTH, HEIGHT))
    root.title("Camera Feed")
    VideoFrame(root, V=VID, bg="0000FF", canvas_height=HEIGHT, canvas_width=WIDTH).pack()

    root.mainloop()


def main():
    app()


if __name__ == "__main__":
    sys.exit(main() or 0)

################################################################################################### 
     