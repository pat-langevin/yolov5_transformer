# This file contains experimental modules

import numpy as np
import torch
import torch.nn as nn

from models.common import Conv, DWConv, CSPBottleneck3Conv
from utils.google_utils import attempt_download


class CrossConv(nn.Module):
    # Cross Convolution Downsample
    def __init__(self, c1, c2, k=3, s=1, g=1, e=1.0, shortcut=False):
        # ch_in, ch_out, kernel, stride, groups, expansion, shortcut
        super(CrossConv, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, (1, k), (1, s))
        self.cv2 = Conv(c_, c2, (k, 1), (s, 1), g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class Sum(nn.Module):
    # Weighted sum of 2 or more layers https://arxiv.org/abs/1911.09070
    def __init__(self, n, weight=False):  # n: number of inputs
        super(Sum, self).__init__()
        self.weight = weight  # apply weights boolean
        self.iter = range(n - 1)  # iter object
        if weight:
            self.w = nn.Parameter(-torch.arange(1., n) / 2, requires_grad=True)  # layer weights

    def forward(self, x):
        y = x[0]  # no weight
        if self.weight:
            w = torch.sigmoid(self.w) * 2
            for i in self.iter:
                y = y + x[i + 1] * w[i]
        else:
            for i in self.iter:
                y = y + x[i + 1]
        return y


class GhostConv(nn.Module):
    # Ghost Convolution https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):  # ch_in, ch_out, kernel, stride, groups
        super(GhostConv, self).__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act)

    def forward(self, x):
        y = self.cv1(x)
        return torch.cat([y, self.cv2(y)], 1)


class GhostBottleneck(nn.Module):
    # Ghost Bottleneck https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=3, s=1):  # ch_in, ch_out, kernel, stride
        super(GhostBottleneck, self).__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(GhostConv(c1, c_, 1, 1),  # pw
                                  DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
                                  GhostConv(c_, c2, 1, 1, act=False))  # pw-linear
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False),
                                      Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class MixConv2d(nn.Module):
    # Mixed Depthwise Conv https://arxiv.org/abs/1907.09595
    def __init__(self, c1, c2, k=(1, 3), s=1, equal_ch=True):
        super(MixConv2d, self).__init__()
        groups = len(k)
        if equal_ch:  # equal c_ per group
            i = torch.linspace(0, groups - 1E-6, c2).floor()  # c2 indices
            c_ = [(i == g).sum() for g in range(groups)]  # intermediate channels
        else:  # equal weight.numel() per group
            b = [c2] + [0] * groups
            a = np.eye(groups + 1, groups, k=-1)
            a -= np.roll(a, 1, axis=1)
            a *= np.array(k) ** 2
            a[0] = 1
            c_ = np.linalg.lstsq(a, b, rcond=None)[0].round()  # solve for equal weight indices, ax = b

        self.m = nn.ModuleList([nn.Conv2d(c1, int(c_[g]), k[g], s, k[g] // 2, bias=False) for g in range(groups)])
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return x + self.act(self.bn(torch.cat([m(x) for m in self.m], 1)))


class Ensemble(nn.ModuleList):
    # Ensemble of models
    def __init__(self):
        super(Ensemble, self).__init__()

    def forward(self, x, augment=False):
        y = []
        for module in self:
            y.append(module(x, augment)[0])
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.stack(y).mean(0)  # mean ensemble
        y = torch.cat(y, 1)  # nms ensemble
        return y, None  # inference, train output


def attempt_load(weights, map_location=None):
    # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        attempt_download(w)
        ckpt = torch.load(w, map_location=map_location)  # load
        model.append(ckpt['ema' if ckpt.get('ema') else 'model'].float().fuse().eval())  # FP32 model

    # Compatibility updates
    for m in model.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True  # pytorch 1.7.0 compatibility
        elif type(m) is Conv:
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility

    if len(model) == 1:
        return model[-1]  # return model
    else:
        print('Ensemble created with %s\n' % weights)
        for k in ['names', 'stride']:
            setattr(model, k, getattr(model[-1], k))
        return model  # return ensemble

# Experiements for project
# initialize transformer block with a CSP block as a backbone
class TF(CSPBottleneck3Conv):
    # initialize transformer block
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        # added a CSP layer
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        # create transforemer block
        self.m = TFBlock(c_, c_, 4, n)

# transformer layer class
class TFLayer(nn.Module):
    # initialize layer
    def __init__(self, c, num_heads):
        super().__init__()
        # transformer parameters and components
        self.q = nn.Linear(c, c, bias=False) #query target
        self.k = nn.Linear(c, c, bias=False) #keys source
        self.v = nn.Linear(c, c, bias=False) #values source
        
        # from the paper
        # multihead structure
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        # normalization structure
        self.ln1 = nn.LayerNorm(c)
        self.ln2 = nn.LayerNorm(c)
        # linear structures
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, input):
        # normalisation of input
        ln1_output = self.ln1(input)
        # using the keys, query, and values as input to the 
        # multihead attention machanism
        ma_output = self.ma(self.q(ln1_output), self.k(ln1_output), self.v(ln1_output))[0] + input
        # normalisation of multihead output
        ln2_output = self.ln2(ma_output)
        # linearization of normalisation output
        output = self.fc2(self.fc1(ln2_output)) + ln2_output
        # return output
        return output

# transformer block class
# create a block of transformer based on the layers
class TFBlock(nn.Module):
    def __init__(self, c1, c2, num_heads, num_layers):
        super().__init__()

        # standard convolution if input
        # and output are not equal
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        # linearisation of input
        self.linear = nn.Linear(c2, c2)
        # create sequence of multiple layer of transformer
        self.tr = nn.Sequential(*[TFLayer(c2, num_heads) for _ in range(num_layers)])
        # set output size
        self.c2 = c2

    def forward(self, input):
        if self.conv is not None:
            input = self.conv(input)
        # rshape of tensor
        b, _, w, h = input.shape
        p = input.flatten(2)
        # reformat tensor
        p = p.unsqueeze(0)
        # transpose the tensor
        p = p.transpose(0, 3)
        # remove dimensition of size 3
        p = p.squeeze(3)
        # linear
        e = self.linear(p)
        # add linear value
        input = p + e

        # sequential
        input = self.tr(input)
        # reformat tensor
        input = input.unsqueeze(3)
        # transpose
        input = input.transpose(0, 3)
        # reshape
        input = input.reshape(b, self.c2, w, h)
        # return output
        return input