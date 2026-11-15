#!/usr/bin/env python3
"""YOLOPX (LIMO fine-tuned) -> ONNX export for TensorRT.

Exported graph
  input : "image"          float32 [1,3,H,W], RGB, 0~255 (normalization is inside the graph)
  output: "drivable_mask"  uint8   [1,H,W]  (1 = drivable area)
          "lane_mask"      uint8   [1,H,W]  (1 = lane line)
The detection head is untrained, so it is dropped from the graph.

Usage:
  python3 tools/export_onnx.py --weights models/epoch-80.pth --output models/yolopx_480x640.onnx \
      [--image some_frame.jpg]
"""
import argparse
import os
import sys
import types

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
YOLOPX_DIR = os.path.join(HERE, 'YOLOPX')
sys.path.insert(0, YOLOPX_DIR)

# lib/ imports training-only packages at module level. They are not used to build
# the model, so stub them instead of pip-installing (albumentations would pull
# opencv-python and clobber the Jetson OpenCV build).
def _stub_getattr(attr):
    if attr.startswith('__'):
        raise AttributeError(attr)
    return type(attr, (), {})  # any public name -> dummy class


for _name in ['prefetch_generator', 'albumentations', 'tensorboardX', 'imageio', 'loguru', 'yacs',
              'yacs.config']:
    try:
        __import__(_name)
    except ImportError:
        _m = types.ModuleType(_name)
        _m.__path__ = []
        _m.__getattr__ = _stub_getattr
        sys.modules[_name] = _m

from lib.models import get_net  # noqa: E402

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class YolopxSeg(nn.Module):
    """Normalization + segmentation heads + argmax, without the detection head."""

    def __init__(self, net):
        super().__init__()
        self.net = net
        self.register_buffer('mean', torch.tensor(MEAN).view(1, 3, 1, 1) * 255.0)
        self.register_buffer('std', torch.tensor(STD).view(1, 3, 1, 1) * 255.0)

    def forward(self, x):
        x = (x - self.mean) / self.std
        _, da, ll = self.net(x)
        da_mask = (da[:, 1] > da[:, 0]).to(torch.uint8)  # == argmax over 2 classes
        ll_mask = (ll[:, 1] > ll[:, 0]).to(torch.uint8)
        return da_mask, ll_mask


def load_model(weights):
    net = get_net(None)  # cfg is unused by get_net
    ckpt = torch.load(weights, map_location='cpu', weights_only=True)
    net.load_state_dict(ckpt['state_dict'])
    net.eval()
    return YolopxSeg(net).eval()


def load_input(image, h, w):
    if image is None:
        rng = np.random.default_rng(0)
        return torch.from_numpy(rng.uniform(0, 255, (1, 3, h, w)).astype(np.float32))
    import cv2
    img = cv2.imread(image, cv2.IMREAD_COLOR)
    assert img is not None, f'cannot read {image}'
    assert img.shape[:2] == (h, w), f'image is {img.shape[:2]}, expected {(h, w)}'
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(img.transpose(2, 0, 1)[None].astype(np.float32).copy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--height', type=int, default=480)
    ap.add_argument('--width', type=int, default=640)
    ap.add_argument('--opset', type=int, default=17)
    ap.add_argument('--image', default=None, help='640x480 frame used for the ORT vs PyTorch check')
    args = ap.parse_args()
    assert args.height % 32 == 0 and args.width % 32 == 0, 'H and W must be multiples of 32'

    model = load_model(args.weights)
    x = load_input(args.image, args.height, args.width)

    with torch.no_grad():
        ref_da, ref_ll = model(x)
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        torch.onnx.export(
            model, (x,), args.output,
            input_names=['image'], output_names=['drivable_mask', 'lane_mask'],
            opset_version=args.opset, do_constant_folding=True, dynamo=False)
    print(f'[export] saved {args.output} ({os.path.getsize(args.output) / 1e6:.1f} MB)')

    import onnx
    onnx.checker.check_model(onnx.load(args.output))
    print('[check] onnx.checker OK')

    import onnxruntime as ort
    sess = ort.InferenceSession(args.output, providers=['CPUExecutionProvider'])
    da, ll = sess.run(None, {'image': x.numpy()})
    for name, ref, out in [('drivable', ref_da.numpy(), da), ('lane', ref_ll.numpy(), ll)]:
        agree = (ref == out).mean() * 100
        print(f'[check] {name:8s} mask: PyTorch vs ORT agreement {agree:.3f}%  '
              f'(positive px torch={int(ref.sum())}, ort={int(out.sum())})')


if __name__ == '__main__':
    main()
