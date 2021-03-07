from models.experimental import attempt_load
from models.yolo import Model
from utils.autoanchor import check_anchors
from utils.general import check_file
from utils.torch_utils import select_device
import yaml


cfg = check_file('models/yolov5s_transformers.yaml')
hyp = check_file('data/hyp.scratch.yaml')
data = 'data/coco128.yaml'
device = select_device('', batch_size=32)

with open(data) as f:
    data_dict = yaml.load(f, Loader=yaml.SafeLoader)  # data dict
nc = int(data_dict['nc'])  # number of classes

with open(hyp) as f:
    hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps
model = Model(cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
print(model)