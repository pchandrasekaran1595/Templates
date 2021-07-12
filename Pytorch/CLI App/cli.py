import cv2
import sys
import platform

import utils as u
from Models import Model


def app():
    args_1 = "--classify"
    args_2 = "--detect"
    args_3 = "--segment"
    args_4 = "--id"

    do_classify, do_detect, do_segment = None, None, None

    if args_1 in sys.argv:
        do_classify = True
    if args_2 in sys.argv:
        do_detect = True
    if args_3 in sys.argv:
        do_segment = True
    if args_4 in sys.argv:
        u.ID = int(sys.argv[sys.argv.index(args_4) + 1])
    
    if do_classify:
        model = Model(modeltype="classifier")
        model.eval()
    
    if do_detect:
        model = Model(modeltype="detector")
        model.eval()
    
    if do_segment:
        model = Model(modeltype="segmentor")
        model.eval()
    
    model.to(u.DEVICE)

    if platform.system() != "Windows":
        cap = cv2.VideoCapture(u.ID)
    else:
        cap = cv2.VideoCapture(u.ID, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, u.CAM_HEIGHT)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, u.CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FPS, u.FPS)

    while cap.isOpened():
        _, frame = cap.read()

        if do_classify:
            frame = u.classify(model, frame)

        if do_detect:
            frame = u.detect(model, frame)
        
        if do_segment:
            frame = u.segment(model, frame)
        
        cv2.imshow("Feed", frame)
        if cv2.waitKey(1) == ord("q"):
            break
    
    cap.release()
    cv2.destroyAllWindows()

    
