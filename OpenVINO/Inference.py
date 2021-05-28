import os
import cv2
import platform
from openvino import inference_engine

"""
    Can be found in getModel.bat; variable name is irs_path and target_precision
"""
BASE_PATH = "PATH_TO_XML_AND_BIN_FILES"
PRECISION = "FP16"


def get_model(model_name=None):
    ie = inference_engine.IECore()
    path = os.path.join(os.path.join(BASE_PATH, model_name), PRECISION)
    device = "MYRIAD" if "MYRIAD" in ie.available_devices else "CPU"

    network = ie.read_network(model=os.path.join(path, model_name+".xml"),
                              weights=os.path.join(path, model_name+".bin"))
    network = ie.load_network(network, device, num_requests=0)
    input_blob, output_blob = next(iter(network.input_info)), next(iter(network.outputs))
    N, C, H, W = network.input_info[input_blob].input_data.shape

    return network, (input_blob, output_blob), (N, C, H, W)


# ********************************************************************************************* #
"""
    1. Assumes image is in BGR format
    2. Returns processed image and result
"""
def infer_image(model_name=None, image=None):
    """
        Processing ...
    """

    network, (input_blob, output_blob), (N, C, H, W) = get_model(model_name=model_name)
    disp_image = image.copy()
    disp_image = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2RGB)
    dh, dw, _ = disp_image.shape
    image = cv2.resize(src=image, dsize=(W, H), interpolation=cv2.INTER_AREA).transpose(2, 0, 1)
    result = network.infer({input_blob : image})[output_blob]

    """
        Processing ....
    """
    processed_result = ___________
    return disp_image, processed_result


# ********************************************************************************************* #

VIDEO_WIDTH, VIDEO_HEIGHT, VIDEO_FPS, VIDEO_DELAY = 640, 480, 30, 5
def infer_realtime(device_id=0, model_name=None):
    network, (input_blob, output_blob), (N, C, H, W) = get_model(model_name=model_name)

    if platform.system() != "Windows":
        vid = cv2.VideoCapture(device_id)
    else:
        vid = cv2.VideoCapture(device_id, cv2.CAP_DSHOW)
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_HEIGHT)
    vid.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_WIDTH)
    vid.set(cv2.CAP_PROP_FPS, VIDEO_FPS)

    while vid.isOpened():
        _, frame = vid.read()
        disp_frame = frame.copy()
        dh, dw, _ = disp_frame.shape

        frame = cv2.resize(src=cv2.cvtColor(src=frame, code=cv2.COLOR_BAYER_BG2BGR), dsize=(W, H), interpolation=cv2.INTER_AREA).transpose(2, 0, 1)
        result = network.infer({input_blob : frame})[output_blob]

        """
            Processing ....
            Get Processed Frame
            disp_frame is now Processed Frame
        """
        
        cv2.imshow("Feed", disp_frame)
        if cv2.waitKey(VIDEO_DELAY) == ord("q"):
            break

    vid.release()
    cv2.destroyAllWindows()


# ********************************************************************************************* #

"""
    path = PATH_TO_VIDEO_FILE/Video.mp4
"""
def infer_video(device_id=0, model_name=None, path=None):
    network, (input_blob, output_blob), (N, C, H, W) = get_model(model_name=model_name)

    vid = cv2.VideoCapture(path)

    while vid.isOpened():
        ret, frame = vid.read()

        if ret:
            disp_frame = frame.copy()
            dh, dw, _ = disp_frame.shape

            frame = cv2.resize(src=cv2.cvtColor(src=frame, code=cv2.COLOR_BAYER_BG2BGR), dsize=(W, H), interpolation=cv2.INTER_AREA).transpose(2, 0, 1)
            result = network.infer({input_blob : frame})[output_blob]

            """
                Processing ....
                Get Processed Frame
                disp_frame is now Processed Frame
            """
            
            cv2.imshow("Feed", disp_frame)
            if cv2.waitKey(15) == ord("q"):
                break
        else:
            vid.set(cv2.CAP_PROP_POS_FRAMES, 0)


    vid.release()
    cv2.destroyAllWindows()

