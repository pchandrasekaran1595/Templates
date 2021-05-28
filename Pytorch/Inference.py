import cv2
import torch
import numpy as np
from torchvision import transforms, models, ops

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABELS_USED = []  # List containing labels of respective models

"""
    Detector Transform
        transform = transforms.Compose([transforms.ToTensor(), ])
    All other Transform
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225]),
                                        ])
"""


def breaker(num=50, char="*"):
    print("\n" + num*char + "\n")

# ********************************************************************************************* #

"""
    1. Assumes image to be in BGR Format
    2. Assumes Model is in .eval() mode
    3. Present Script only gives most confident detection; modify if more are required
    4. Returns image with boudling box drawn on it.
"""
def infer_detector(image=None, model=None, transform=None, size=224, iou=0.5):
    disp_image = image.copy()
    dh, dw, _ = disp_image.shape
    image = cv2.cvtColor(src=cv2.resize(src=image, dsize=(size, size), interpolation=cv2.INTER_AREA), 
                         code=cv2.COLOR_BGR2RGB)

    with torch.no_grad():
        output = model(transform(image).to(device).unsqueeze(dim=0))
    cnts, scrs, lbls = output[0]["scores"], output[0]["scores"], output[0]["labels"]

    if len(cnts) != 0:
        cnts = ops.clip_boxes_to_image(cnts, (size, size))
        best_index = ops.nms(cnts, scrs, iou)
        x1, y1, x2, y2 = int(cnts[best_index][0] * dw), int(cnts[best_index][1] * dh), \
                         int(cnts[best_index][2] * dw), int(cnts[best_index][3] * dh)
        cv2.rectangle(img=disp_image, pt1=(x1, y1), pt2=(x2, y2), color=(0, 0, 255), thickness=2)
        # cv2.putText(img=disp_image, text=LABELS_USED[int(lbls[best_index].item())], org=(x1-10, y1-10),
        #             fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
        #             color=(0, 0, 255), thickness=1)
    else:
        # breaker()
        # print("No Objects Detected")
        # breaker()
        cv2.putText(img=disp_image, text="No Objects Detected", org=(15, 15),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                    color=(0, 0, 255), thickness=1)
    return disp_image


# ********************************************************************************************* #

"""
    1. Assumes image to be in BGR Format
    2. Assumes Model is in .eval() mode
    3. Returns a tuple 
        i.  Image with class label present
        ii. Class Label Index
"""
def infer_classifier(image=None, model=None, transform=None, size=224):
    disp_image = image.copy()
    dh, dw, _ = disp_image.shape
    image = cv2.cvtColor(src=cv2.resize(src=image, dsize=(size, size), interpolation=cv2.INTER_AREA), 
                         code=cv2.COLOR_BGR2RGB)

    with torch.no_grad():
        output = torch.argmax(model(transform(image).to(device).unsqueeze(dim=0)), dim=1)

    # Could be output[0].item()
    cv2.putText(img=disp_image, text=LABELS_USED[int(output[0][0].item())], org=(25, 75),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                color=(0, 0, 255), thickness=1)
    return disp_image, int(output[0][0].item())


# ********************************************************************************************* #

"""
    1. Assumes image to be in BGR Format
    2. Assumes Model is in .eval() mode
    3. Returns a tuple 
        i.  Segmented Image
        ii. Unique Class Index present in class_index_image
"""
def image_segment(image=None, model=None, transform=None):

    with torch.no_grad():
        output = model(transform(image).to(device).unsqueeze(0))["out"]

    class_index_image = torch.argmax(output[0], dim=0)
    return decode(class_index_image=class_index_image), np.unique(class_index_image.detach().cpu().numpy())


def decode(class_index_image=None):
    colors = np.array([(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                       (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
                       (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                       (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])

    r, g, b = np.zeros(class_index_image.shape, dtype=np.uint8), \
              np.zeros(class_index_image.shape, dtype=np.uint8), \
              np.zeros(class_index_image.shape, dtype=np.uint8)

    for i in range(21):
        indexes = (class_index_image == i)
        r[indexes] = colors[i][0]
        g[indexes] = colors[i][1]
        b[indexes] = colors[i][2]
    return np.stack([r, g, b], axis=2)

# ********************************************************************************************* #
