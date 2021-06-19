import os
import cv2
import sys
import getpass
import numpy as np

from cam_and_vid import Handler

USER = getpass.getuser()

def alpha_blend():
    width, height, fps = 640, 480, 30
    path, name = "C:/Users/" + USER + "/Pictures", "Photo.jpg"
    alpha = 0.10

    args_1 = "--width"
    args_2 = "--height"
    args_3 = "--image-dir"
    args_4 = "--filename"
    args_5 = "--alpha"

    if args_1 in sys.argv:
        width = int(sys.argv[sys.argv.index(args_1) + 1])
    if args_2 in sys.argv:
        height = int(sys.argv[sys.argv.index(args_2) + 1])
    if args_3 in sys.argv:
        path = sys.argv[sys.argv.index(args_3) + 1]
    if args_4 in sys.argv:
        name = sys.argv[sys.argv.index(args_4) + 1]
    if args_5 in sys.argv:
        alpha = float(sys.argv[sys.argv.index(args_5) + 1])

    image = cv2.resize(src=cv2.imread(os.path.join(path, name), cv2.IMREAD_COLOR), dsize=(width, height), interpolation=cv2.INTER_AREA)
    image = image/255
    CaptureObject = Handler(device_id=0, width=width, height=height, fps=fps)
    CaptureObject.start()
    while True:
        ret, frame = CaptureObject.getframe()
        if ret:
            frame = frame/255
            frame = image*alpha + frame*(1-alpha)
            frame = np.clip(frame*255, 0, 255).astype("uint8")

            cv2.imshow("Blended", frame)
            if cv2.waitKey(1) == ord("q"):
                break
        else:
            break
    CaptureObject.stop()
    cv2.destroyAllWindows()