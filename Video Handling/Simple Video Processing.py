import os
import cv2
import sys
import getpass
import numpy as np

from cam_and_vid import Handler

def handle_video(path=None):
    width, height, fps = 640, 480, 30

    args_1 = "--width"
    args_2 = "--height"

    if args_1 in sys.argv:
        width = int(sys.argv[sys.argv.index(args_1) + 1])
    if args_2 in sys.argv:
        height = int(sys.argv[sys.argv.index(args_2) + 1])
    
    if not path:
        CaptureObject = Handler(device_id=0, width=width, height=height, fps=fps)
        CaptureObject.start()
    else:
        CaptureObject = Handler(path=path,  width=width, height=height, fps=fps)
        CaptureObject.start()
    
    while True:
        ret, frame = CaptureObject.getframe()
        if ret:
            """
                Process the frame
            """
            cv2.imshow("Processed", frame)
            if cv2.waitKey(1) == ord("q"):
                break
        else:
            if path is not None:
                CaptureObject.vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
            else:
                break
    
    CaptureObject.stop()
    cv2.destroyAllWindows()
