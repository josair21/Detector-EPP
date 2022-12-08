import argparse
import os
import platform
import sys
from pathlib import Path
import torch
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode
detecciones = 0
zzi = 0
source = '0'
device = ''
weights = 'best.pt'
dnn = False
data='data/coco128.yaml'
half = False
imgsz = (640, 640)
vid_stride = 1
conf_thres = 0.5
# Load model
device = select_device(device)
model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
stride, names, pt = model.stride, model.names, model.pt
imgsz = check_img_size(imgsz, s=stride)  # check image size

dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)

model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
seen, windows, dt = 0, [], (Profile(), Profile(), Profile())

for path, im, im0s, vid_cap, s in dataset:
    casco = False
    chaleco = False
    guantes = False
    lentes = False
    persona = False
    zapatos = False
    with dt[0]:
        im = torch.from_numpy(im).to(model.device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

    # Inference
    with dt[1]:
        im = cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE)
        pred = model(im, augment=False, visualize=False)

    # NMS
    with dt[2]:
        pred = non_max_suppression(pred, conf_thres, 0.45, None, False, max_det=1000)

    # Second-stage classifier (optional)
    # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

    # Process predictions
    for i, det in enumerate(pred):  # per image
        seen += 1
        p, im0, frame = path[i], im0s[i].copy(), dataset.count
        s += f'{i}: '
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
            # Print results
            for c in det[:, 5].unique():
                print(c)
                n = (det[:, 5] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                if c == torch.tensor(0):
                    casco = True
                if c == torch.tensor(1):
                    chaleco = True
                if c == torch.tensor(2):
                    guantes = True
                if c == torch.tensor(3):
                    lentes = True
                if c == torch.tensor(4):
                    persona = True 
                if c == torch.tensor(5):
                    zapatos = True

                '''casco = True if c == torch.tensor(0) else None
                chaleco = True if c == torch.tensor(1) else None
                guantes = True if c == torch.tensor(2) else None
                lentes = True if c == torch.tensor(3) else None
                persona = True if c == torch.tensor(4) else None
                zapatos = True if c == torch.tensor(5) else None
                casco = False
                chaleco = False
                guantes = False
                lentes = False
                persona = False
                zapatos = False  (casco == True) and'''
            if  casco and lentes and chaleco and guantes:
                zzi += 1
            else:
                zzi = 0
        # Stream results
            cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
            cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
            im0 = cv2.circle(im0, (30*(zzi+1),30), radius=5, color=(0,0,255), thickness=5)
            cv2.imshow(str(p), im0)
            if zzi > 10:
                detecciones += 1
                zzi = 0
                font                   = cv2.FONT_HERSHEY_SIMPLEX
                bottomLeftCornerOfText = (300,500)
                fontScale              = 5
                fontColor              = (0,0,255)
                thickness              = 5
                lineType               = 2

                cv2.putText(im0,'ADELANTE', 
                    bottomLeftCornerOfText, 
                    font, 
                    fontScale,
                    fontColor,
                    thickness,
                    lineType)
                cv2.imshow(str(p), im0)
                cv2.imwrite('detecciones/'+str(detecciones)+'.jpg', im0)
                cv2.waitKey(2000)
            else:
                cv2.waitKey(1)  # 1 millisecond
    # Print time (inference-only)
    LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")
# Print results
t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)