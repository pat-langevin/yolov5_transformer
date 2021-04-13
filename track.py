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

# class for a positon
class Position:
    # initialization of a position
    def __init__(self, x, y):
        self.x = x
        self.y = y

    # find the score from a point to this point
    # based on the pytagore
    def score(self, position:'Position') -> float:
        return math.sqrt((self.x - position.x)**2 + (self.y - position.y)**2)

# class of a track
# is used to keep all the values needed to draw a track on a frame
# keeps the statuses
class Track:
    # initialization of a track
    def __init__(self, color, track_id, isUsed, position_list:[Position]):
        self.track_id = track_id # id
        self.color = color # the color
        self.isActive = True # set the track to active
        self.position_list = position_list # create an empty list of postion
        self.isUsed = isUsed # set the used status
        self.step = 0 # step to 0

    # add a point/position to a track
    def add_position(self, position:Position):
        self.position_list.append(position)

    # return latest point added to the track, otherwise none
    def last_position(self) -> Position:
        if(len(self.position_list) == 0):
            return None
        else:
            return self.position_list[-1]
    
    # get if a track is active
    def getIsActive(self) -> bool:
        return self.isActive

    # set active status
    def setIsActive(self, isActive):
        self.isActive = isActive
    
    # return the id
    def getId(self) -> int:
        return self.track_id
    
    # set the status for a frame
    def setIsUsed(self, value:bool):
        self.isUsed = value
    
    # return the status
    def getIsUsed(self) -> bool:
        return self.isUsed
    
    # return the color
    def getColor(self):
        return self.color

    # return all the position of the track
    def getPositionList(self) -> [Position]:
        return self.position_list

    # return the steps, used to set a track
    # to inactive
    def increase_step(self):
        self.step += 1

    # set the step to 0, to keep it active
    def reset_step(self):
        self.step = 0
        
    # return number of steps
    def getStep(self) -> int:
        return self.step

# The algorithm to manage the tracks to be drawn on a frame
class Tracking:
    # empty list of tracks
    def __init__(self, track_list):
        self.tracking_list = track_list

    # add a track to the track list
    def add_track(self, track:Track):
        self.tracking_list.append(track)
    
    # return the track  list
    def get_tracking_list(self) -> []:
        return self.tracking_list
    
    # add a position to a track
    def add_to_track(self, track_id, position:Position):
        for track in self.tracking_list:
            if(track.getId() == track_id and track.getIsActive()):
                track.add_position(position)

    # reset the used status of all the tracks to false
    # to be used for the next frame
    def reset_tracks(self):
        for track in self.tracking_list:
            track.setIsUsed(False)

    # return a track based on an id
    def getTrack(self, track_id) -> Track:
        for track in self.tracking_list:
            if track.getId() == track_id:
                return track
        return None

    # increase the steps off all the tracks
    # if a track has 100 step, it is inactive
    # and cannot be used again
    def increase_step_all_track(self):
        for track in self.tracking_list:
            if track.getIsActive():    
                track.increase_step()
                #print(track.getStep())
                if track.getStep() >= 100:
                    #print("False")
                    track.setIsActive(False)
    
    # return the id of a track based on the best score
    # of a position and tyhe lastest point of the not
    # used track during a frame
    def getBestTrackID(self, position:Position) -> int:
        id_list = []
        score_list = []
        for track in self.tracking_list:
            if track.getIsUsed() == False and track.getIsActive() == True:
                last = track.last_position()
                id_list.append(track.getId())
                score_list.append(position.score(last))
                #print(score_list[-1])
                track.setIsUsed(True)

        if(len(id_list) == 0):
            return 0

        if(min(score_list) >= 15):
            return 0
        #print("----get best track----")
        #print(id_list)
        #print(score_list)
        #print("----end get best track----")
        index = argmin(score_list)
        return id_list[index]

# based on the detect from detection.py
def track(save_img=False):
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
                    # get corners of the bonding box
                    x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                    # find x center
                    x = int(x1 + (x2 - x1)/2)
                    # find y center
                    y = int(y1 + (y2 - y1)/2)
                    # create position with center values
                    pos = Position(x, y)
                    # find best track based on the position
                    track_id = tracking_struct.getBestTrackID(pos)
                    # empty color
                    color = []
                    # null color
                    track = None

                    # if no track were found
                    if (track_id == 0):
                        # increase counter
                        counter += 1
                        # randomize color
                        color = [random.randint(0, 255) for _ in range(3)]
                        # empty new list of positon
                        list_position = []
                        # create new track
                        new_track = Track(color, counter, True, list_position)
                        # add postion to track
                        new_track.add_position(pos)
                        # add track to track list
                        tracking_struct.add_track(new_track)
                        # point track to new track
                        track = new_track
                    # if a track was found
                    else:
                        # find track
                        track = tracking_struct.getTrack(track_id)
                        # get color
                        color = track.getColor()
                        # add position
                        track.add_position(pos)
                        # reset the step
                        track.reset_step()
                        # used for this frame
                        track.setIsUsed(True)

                    # get all the point of the track (new of found)
                    position_track_list = track.getPositionList()
                    # if more than 2 positon, then draw lines
                    if (len(position_track_list) >= 2) :
                        # for each position present in position list of the track
                        for i in range(len(position_track_list) - 1):
                            # initial point
                            initial = position_track_list[i]
                            # final point
                            final = position_track_list[i+1]
                            # draw line from initial to final point on the frame
                            plot_line((initial.x, initial.y), (final.x, final.y), im0, color, 1)

                    if save_img or view_img:  # Add bbox to image
                        # save image to video
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box_and_dot(xyxy, (x, y), im0, track_id, color, label=label, line_thickness=3)
                    
                    # reset all the track in track list
                    tracking_struct.reset_tracks()
                # increase step in all track to find inactive
                tracking_struct.increase_step_all_track()

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
    parser.add_argument('--project', default='runs/track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    check_requirements()

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                track()
                strip_optimizer(opt.weights)
        else:
            track()
