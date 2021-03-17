import argparse
import time
import math
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random, argmin

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box_and_dot, plot_line
from utils.torch_utils import select_device, load_classifier, time_synchronized

class Position:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def score(self, position:Position) -> float:
        return math.sqrt((self.x - position.x)**2 + (self.y - position.y)**2)

class Track:
    def __init__(self, color, id, isUsed, position_list:[Position]):
        self.id = id
        self.color = color
        self.isActive = True
        self.position_list = position_list
        self.isUsed = isUsed

    def add_position(self, position:Position):
        self.position_list.append(position)

    def last_position(self) -> Position:
        return self.position_list[-1]

    def getIsActive(self) -> bool:
        return self.isActive

    def setIsActive(self, isActive):
        self.isActive = isActive

    def getId(self) -> int:
        return self.id
    
    def setIsUsed(self, value:bool):
        self.isUsed = value

    def getIsUsed(self) -> bool:
        return self.isUsed

    def getColor(self):
        return self.color

    def getPositionList(self) -> [Position]:
        return self.position_list


class Tracking:
    def __init__(self, track_list):
        self.tracking_list = track_list

    def add_track(self, track:Track):
        self.tracking_list.append(track)
    
    def add_to_track(self, id, position:Position):
        for track in self.tracking_list:
            if(track.getId == id and track.getIsActive(self)):
                track.add_position(self, position)

    def reset_tracks(self):
        for track in self.tracking_list:
            track.setIsUsed(self, False)

    def getTrack(self, id) -> Track:
        for track in self.tracking_list:
            if track.getId(self) == id:
                return track
        return None
    
    def getBestTrackID(self, position:Position) -> int:
        id_list = []
        score_list = []
        for track in self.tracking_list:
            if track.getIsUsed(self) == False:
                last = track.last_position(self)
                id_list.append(track.getId)
                score_list.append(position.score(last))
                track.setIsUsed(self, True)

        if(len(id_list) == 0):
            return 0

        index = argmin(score_list)
        return id_list[index]


def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    #Tracking structure
    track_list = []
    tracking_struct = Tracking(track_list)
    counter = 0

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):

                    # Tracking code
                    pos = Position(xyxy[0], xyxy[1])
                    id = tracking_struct.getBestTrackID(pos)
                    color = []
                    track = None
                    if (id == 0):
                        counter += 1
                        color = [random.randint(0, 255) for _ in range(3)]
                        new_track = Track(color, counter, True, [])
                        tracking_struct.add_track(new_track)
                        track = new_track
                    else:
                        track = tracking_struct.getTrack(id)
                        color = track.getColor()
                        #color = tracking_struct.getTrack(id).getColor()
                        track.add_position(pos)
                        track.setIsUsed(True)

                    position_list = track.getPositionList()
                    if (len(position_list) >= 2) :
                        for i in range(len(position_list) - 1):
                            initial = position_list[i]
                            final = position_list[i+1]
                            plot_line((initial.x, initial.y), (final.x, final.y), img, color, 1)

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box_and_dot(xyxy, im0, id, color, label=label, line_thickness=3)

                tracking_struct.reset_tracks()

            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)') # remove to see prints

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)
    check_requirements()

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
