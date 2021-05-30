import os
import cv2
import platform

WIDTH, HEIGHT, FPS = 640, 360, 30
OUT_DIR = "PATH_TO_SAVE_VIDEO_FILES"

def __initvideo__(device_id):
    if platform.system != "Windows":
        vid = cv2.VideoCapture(device_id)
    else:
        vid = cv2.VideoCapture(device_id, cv2.CAP_DSHOW)
    vid.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    vid.set(cv2.CAP_PROP_FPS, FPS)

    return vid


def __initwriter__(filename, codec, fps, width, height):
    return 


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


def save_video(path=None):

    filename = "MyVideo.mp4"
    codec = cv2.VideoWriter_fourcc(*"mp4v")  # MP4
    # codec = cv2.VideoWriter_fourcc(*"mpjg")  # AVI
    fps = 30

    out = cv2.VideoWriter(os.path.join(OUT_DIR, filename), fps, codec, (WIDTH, HEIGHT))

    if not path:
        vid = __initvideo__(0)
        while vid.isOpened():
            _, frame = vid.read()
            disp_frame = frame.copy()

            """
                Processing ...
            """

            out.write(disp_frame)
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

                out.write(disp_frame)
                if cv2.waitKey(1) == ord("q"):
                    break
            else:
                vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    out.release()
    vid.release()
    cv2.destroyAllWindows()
