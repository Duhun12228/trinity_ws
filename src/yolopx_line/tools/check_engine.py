#!/usr/bin/env python3
"""Compare a multiclass TensorRT engine against its ONNX model (onnxruntime, FP32 reference).

Per image: class-map agreement and per-class IoU (drivable 1~3, lane 1~2), and the
detection score difference on the boxes the reference keeps (score >= --conf).
The v1 (2-class) checker is kept in tools/v1/.

Usage:
  python3 tools/check_engine.py --engine models/yolopx_mc_480x640_fp16.engine \
      --onnx models/yolopx_mc_480x640.onnx --images frame1.jpg frame2.jpg ... [--save-dir out/]
"""
import argparse
import os
import time

import cv2
import numpy as np
import pycuda.autoinit  # noqa: F401  (creates the CUDA context)
import pycuda.driver as cuda
import tensorrt as trt

OUTPUTS = ('drivable_class', 'lane_class', 'det')
DA_NAMES = {1: 'mylane', 2: 'left', 3: 'right'}
LL_NAMES = {1: 'lane', 2: 'stop'}


class TrtRunner:
    def __init__(self, engine_path):
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        self.ctx = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        self.host, self.dev = {}, {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            self.host[name] = cuda.pagelocked_empty(shape, dtype)
            self.dev[name] = cuda.mem_alloc(self.host[name].nbytes)
            self.ctx.set_tensor_address(name, int(self.dev[name]))

    def __call__(self, image_nchw):
        np.copyto(self.host['image'], image_nchw)
        cuda.memcpy_htod_async(self.dev['image'], self.host['image'], self.stream)
        self.ctx.execute_async_v3(self.stream.handle)
        for name in OUTPUTS:
            cuda.memcpy_dtoh_async(self.host[name], self.dev[name], self.stream)
        self.stream.synchronize()
        return [self.host[name][0].copy() for name in OUTPUTS]


def preprocess(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb.transpose(2, 0, 1)[None].astype(np.float32)


def iou(a, b):
    union = np.logical_or(a, b).sum()
    return float('nan') if union == 0 else np.logical_and(a, b).sum() / union


def overlay(bgr, da, ll, det, conf):
    out = bgr.copy()
    for m, c in ((da == 1, (0, 255, 0)), (da == 2, (235, 206, 135)), (da == 3, (200, 50, 160)),
                 (ll == 1, (0, 0, 255)), (ll == 2, (203, 192, 255))):
        out[m] = (out[m] * 0.5 + np.array(c) * 0.5).astype(np.uint8)
    for cx, cy, w, h, s, _ in det[det[:, 4] >= conf]:
        cv2.rectangle(out, (int(cx - w / 2), int(cy - h / 2)), (int(cx + w / 2), int(cy + h / 2)),
                      (0, 255, 255), 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--engine', required=True)
    ap.add_argument('--onnx', required=True)
    ap.add_argument('--images', nargs='+', required=True)
    ap.add_argument('--conf', type=float, default=0.3)
    ap.add_argument('--save-dir', default=None, help='save TRT overlays here (no NMS)')
    args = ap.parse_args()

    import onnxruntime as ort
    sess = ort.InferenceSession(args.onnx, providers=['CPUExecutionProvider'])
    trt_run = TrtRunner(args.engine)
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    cols = [f'DA {n}' for n in DA_NAMES.values()] + [f'LL {n}' for n in LL_NAMES.values()]
    print(f'{"image":26s} {"DA agr%":>8s} {"LL agr%":>8s} '
          + ' '.join(f'{c:>9s}' for c in cols) + f' {"det dS":>7s} {"trt ms":>7s}')
    for path in args.images:
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        x = preprocess(bgr)
        r_da, r_ll, r_det = (m[0] for m in sess.run(list(OUTPUTS), {'image': x}))
        t0 = time.perf_counter()
        da, ll, det = trt_run(x)
        ms = (time.perf_counter() - t0) * 1000
        ious = [iou(r_da == c, da == c) for c in DA_NAMES] + [iou(r_ll == c, ll == c)
                                                              for c in LL_NAMES]
        keep = r_det[:, 4] >= args.conf
        ds = np.abs(r_det[keep, 4] - det[keep, 4]).max() if keep.any() else 0.0
        print(f'{os.path.basename(path)[:26]:26s} {(r_da == da).mean() * 100:8.3f} '
              f'{(r_ll == ll).mean() * 100:8.3f} ' + ' '.join(f'{v:9.4f}' for v in ious)
              + f' {ds:7.4f} {ms:7.1f}')
        if args.save_dir:
            cv2.imwrite(os.path.join(args.save_dir, os.path.basename(path)),
                        overlay(bgr, da, ll, det, args.conf))


if __name__ == '__main__':
    main()
