import os
import cv2
import numpy as np

# out = alpha * fg + (1-alpha) * bg
# Use frame as bg, image as fg

WIDTH, HEIGHT = 1280, 720
PATH = "PATH_TO_IMAGE_FILE"
NAME = "IMAGE_FILE_NAME"
ALPHA = 0.10


def __initvideo__(device_id=0):
    vid = cv2.VideoCapture(device_id, cv2.CAP_DSHOW)
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    vid.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    vid.set(cv2.CAP_PROP_FPS, 30)

    return vid


def __getalphaimage__(path, name):
    return cv2.resize(src=cv2.imread(os.path.join(path, name), cv2.IMREAD_COLOR), dsize=(WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)


def app():
    vid = __initvideo__()
    image = __getalphaimage__(PATH, NAME)
    image = image/255

    while vid.isOpened():
        _, frame = vid.read()
        frame = frame/255
        frame = (image - frame)*ALPHA + frame  
        frame = np.clip((255*frame), 0, 255).astype("uint8")

        cv2.imshow("Feed", frame)

        if cv2.waitKey(1) == ord("q"):
            break
    
    vid.release()
    cv2.destroyAllWindows()
