import cv2
import platform

WIDTH, HEIGHT, FPS = 640, 360, 30

def __initvideo__(device_id):
    if platform.system != "Windows":
        vid = cv2.VideoCapture(device_id)
    else:
        vid = cv2.VideoCapture(device_id, cv2.CAP_DSHOW)
    vid.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    vid.set(cv2.CAP_PROP_FPS, FPS)

    return vid


def read_video(path=None):
    if not path:
        vid = __initvideo__(0)
        while vid.isOpened():
            _, frame = vid.read()
            disp_frame = frame.copy()

            """
                Processing ...
            """

            cv2.imshow("Feed", disp_frame)
            if cv2.waitKey(1) == ord("q"):
                break
    else:
        vid = cv2.VideoCapture(path)
        while vid.isOpened():
            ret, frame = vid.read()
            if ret:
                disp_frame = frame.copy()

                """
                    Processing ...
                """

                cv2.imshow("Feed", disp_frame)
                if cv2.waitKey(1) == ord("q"):
                    break
            else:
                vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    vid.release()
    cv2.destroyAllWindows()
