#!/usr/bin/env python3
"""Compare TensorRT engine masks against the ONNX model (onnxruntime, FP32 reference).

Usage:
  python3 tools/check_engine.py --engine models/yolopx_480x640_fp16.engine \
      --onnx models/yolopx_480x640.onnx --images frame1.jpg frame2.jpg ... [--save-dir out/]
"""
import argparse
import os
import time

import cv2
import numpy as np
import pycuda.autoinit  # noqa: F401  (creates the CUDA context)
import pycuda.driver as cuda
import tensorrt as trt


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
        for name in ('drivable_mask', 'lane_mask'):
            cuda.memcpy_dtoh_async(self.host[name], self.dev[name], self.stream)
        self.stream.synchronize()
        return self.host['drivable_mask'][0].copy(), self.host['lane_mask'][0].copy()


def preprocess(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb.transpose(2, 0, 1)[None].astype(np.float32)


def iou(a, b):
    union = np.logical_or(a, b).sum()
    return 1.0 if union == 0 else np.logical_and(a, b).sum() / union


def overlay(bgr, da, ll):
    out = bgr.copy()
    da = (da == 1) & (ll == 0)
    out[da] = (out[da] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)
    out[ll == 1] = (out[ll == 1] * 0.5 + np.array([0, 0, 255]) * 0.5).astype(np.uint8)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--engine', required=True)
    ap.add_argument('--onnx', required=True)
    ap.add_argument('--images', nargs='+', required=True)
    ap.add_argument('--save-dir', default=None, help='save TRT overlays here')
    args = ap.parse_args()

    import onnxruntime as ort
    sess = ort.InferenceSession(args.onnx, providers=['CPUExecutionProvider'])
    trt_run = TrtRunner(args.engine)
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    print(f'{"image":30s} {"DA IoU":>8s} {"LL IoU":>8s} {"LL px ref/trt":>15s} {"trt ms":>7s}')
    for path in args.images:
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        x = preprocess(bgr)
        ref_da, ref_ll = (m[0] for m in sess.run(None, {'image': x}))
        t0 = time.perf_counter()
        da, ll = trt_run(x)
        ms = (time.perf_counter() - t0) * 1000
        print(f'{os.path.basename(path):30s} {iou(ref_da, da):8.4f} {iou(ref_ll, ll):8.4f} '
              f'{int(ref_ll.sum()):>7d}/{int(ll.sum()):<7d} {ms:7.1f}')
        if args.save_dir:
            cv2.imwrite(os.path.join(args.save_dir, os.path.basename(path)), overlay(bgr, da, ll))


if __name__ == '__main__':
    main()
