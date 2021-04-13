## yolov5_transformer
yolov5 applying transformer layer

### how to train the detector
```
python train.py --data data.yaml --cfg config.yaml --weights '' --batch-size 64
```

### how to run the detector
```
python detect.py --source video.mp4 --weights weights.pt --conf 0.25
```

### how to run the tracker
```
python track.py --source video.mp4 --weights weights.pt --conf 0.25
```
