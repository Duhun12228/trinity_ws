#!/usr/bin/env python3
"""YOLOPX (LIMO multiclass, limo_test_2~) -> ONNX export for TensorRT.

Exported graph
  input : "image"           float32 [1,3,H,W], RGB, 0~255 (normalization is inside the graph)
  output: "drivable_class"  uint8   [1,H,W]   0 bg / 1 drivable (my lane) / 2 left / 3 right
          "lane_class"      uint8   [1,H,W]   0 bg / 1 lane / 2 stop_line
          "det"             float32 [1,N,6]   cx, cy, w, h (input px), score, class id
                            score = objectness * best class prob; NMS is done in the node.
The v1 (2-class, no detection) exporter is kept in tools/v1/.

Usage (tools/YOLOPX must hold the matching multiclass code):
  python3 tools/export_onnx.py --weights models/epoch-95.pth \
      --output models/yolopx_mc_480x640.onnx [--image some_frame.jpg]
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

# Same values as YOLOPX/lib/config/default.py (classes.json). yacs is not installed on the
# Jetson, so the config module is not imported; get_net() only needs these three fields.
DET_CLASSES = ['traffic_light_red', 'traffic_light_yellow', 'traffic_light_green',
               'sign_pedestrian', 'sign_slow', 'sign_A', 'sign_B', 'sign_parking',
               'cone', 'road_slow_text', 'crosswalk']
NUM_DA = 4
NUM_LL = 3


class YolopxMC(nn.Module):
    """Normalization + all heads; segmentation argmax and detection score on the GPU."""

    def __init__(self, net):
        super().__init__()
        self.net = net
        self.register_buffer('mean', torch.tensor(MEAN).view(1, 3, 1, 1) * 255.0)
        self.register_buffer('std', torch.tensor(STD).view(1, 3, 1, 1) * 255.0)

    def forward(self, x):
        x = (x - self.mean) / self.std
        det_out, da, ll = self.net(x)
        da_cls = torch.argmax(da, dim=1).to(torch.uint8)
        ll_cls = torch.argmax(ll, dim=1).to(torch.uint8)
        pred = det_out[0]                     # [1,N,5+nc]: cx,cy,w,h, obj, cls probs (decoded)
        cls_prob, cls_id = pred[..., 5:].max(dim=-1, keepdim=True)
        score = pred[..., 4:5] * cls_prob
        det = torch.cat([pred[..., :4], score, cls_id.to(pred.dtype)], dim=-1)
        return da_cls, ll_cls, det


def load_model(weights):
    cfg = types.SimpleNamespace(DET_CLASSES=DET_CLASSES, num_seg_class=NUM_DA,
                                num_ll_class=NUM_LL)
    net = get_net(cfg)
    ckpt = torch.load(weights, map_location='cpu', weights_only=True)
    net.load_state_dict(ckpt['state_dict'])
    net.eval()
    return YolopxMC(net).eval()


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
    ap.add_argument('--image', default=None,
                    help='640x480 frame used for the ORT vs PyTorch check')
    args = ap.parse_args()
    assert args.height % 32 == 0 and args.width % 32 == 0, 'H and W must be multiples of 32'

    model = load_model(args.weights)
    x = load_input(args.image, args.height, args.width)

    with torch.no_grad():
        ref = [t.numpy() for t in model(x)]
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        torch.onnx.export(
            model, (x,), args.output,
            input_names=['image'], output_names=['drivable_class', 'lane_class', 'det'],
            opset_version=args.opset, do_constant_folding=True, dynamo=False)
    print(f'[export] saved {args.output} ({os.path.getsize(args.output) / 1e6:.1f} MB), '
          f'det {tuple(ref[2].shape)}')

    import onnx
    onnx.checker.check_model(onnx.load(args.output))
    print('[check] onnx.checker OK')

    import onnxruntime as ort
    sess = ort.InferenceSession(args.output, providers=['CPUExecutionProvider'])
    out = sess.run(None, {'image': x.numpy()})
    for name, r, o, n in [('drivable', ref[0], out[0], NUM_DA), ('lane', ref[1], out[1], NUM_LL)]:
        agree = (r == o).mean() * 100
        px = ' '.join(f'{c}:{int((o == c).sum())}' for c in range(1, n))
        print(f'[check] {name:8s} class map: PyTorch vs ORT agreement {agree:.3f}%  (ORT px {px})')
    d = np.abs(ref[2][..., 4] - out[2][..., 4]).max()
    top = out[2][0][out[2][0][:, 4].argsort()[::-1][:5]]
    print(f'[check] det score max |PyTorch - ORT| = {d:.2e}; top-5 (cls, score): '
          + ', '.join(f'({DET_CLASSES[int(c)]}, {s:.2f})' for s, c in top[:, 4:6]))


if __name__ == '__main__':
    main()
