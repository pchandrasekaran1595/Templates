import cv2
import numpy as np
import pyrealsense2 as rs

def simple_processing(alpha=0.05):
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    pipeline.start()

    try:
        while True:
            frames = pipeline.wait_for_frames()
            depth_frame, color_frame = frames.get_depth_frame(), frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue
            depth_frame, color_frame = np.array(depth_frame.get_data()), np.array(color_frame.get_data())

            color_frame = cv2.cvtColor(src=color_frame, code=cv2.COLOR_BGR2RGB)
            depth_frame = cv2.applyColorMap(src=cv2.convertScaleAbs(src=depth_frame, alpha=alpha), colormap=cv2.COLORMAP_JET)

            dh, dw, _ = depth_frame.shape
            if depth_frame.shape != color_frame.shape:
                color_frame = cv2.resize(src=color_frame, dsize=(dh, dw), interpolation=cv2.INTER_AREA)
            disp_frame = np.hstack((color_frame, depth_frame))
            cv2.imshow("Feed", disp_frame)
            if cv2.waitKey(1) == ord("q"):
                break
        cv2.destroyAllWindows()
    
    except Exception as e:
        print(e)
    
    finally:
        pipeline.stop()
    