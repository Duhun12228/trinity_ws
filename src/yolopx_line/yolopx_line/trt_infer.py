"""
TensorRT runner for the exported YOLOPX engine.

Engine I/O (see tools/export_onnx.py):
  input  image            float32 [1,3,H,W]  RGB 0~255
  output multiclass (v2)  drivable_class uint8 [1,H,W], lane_class uint8 [1,H,W],
                          det float32 [1,N,6]
         v1 (tools/v1)    drivable_mask uint8 [1,H,W], lane_mask uint8 [1,H,W]
Every tensor other than "image" is treated as an output and returned by name.
"""
import numpy as np
import pycuda.driver as cuda
import tensorrt as trt


class YolopxTrt:

    def __init__(self, engine_path):
        cuda.init()
        # Own context, kept current on the creating thread (rclpy callbacks run there too).
        self.cuda_ctx = cuda.Device(0).make_context()
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f:
            self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f'failed to deserialize engine: {engine_path}')
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
        self.outputs = [n for n in self.host if n != 'image']

        _, _, self.height, self.width = self.host['image'].shape

    def start(self, rgb):
        """
        Enqueue inference on rgb (uint8 HxWx3, engine size) and return immediately.

        The previous result must have been collected with wait() first.
        """
        np.copyto(self.host['image'][0], rgb.transpose(2, 0, 1), casting='unsafe')
        cuda.memcpy_htod_async(self.dev['image'], self.host['image'], self.stream)
        self.ctx.execute_async_v3(self.stream.handle)
        for name in self.outputs:
            cuda.memcpy_dtoh_async(self.host[name], self.dev[name], self.stream)

    def wait(self):
        """
        Block until the enqueued inference finishes.

        Returns {output name: array} with the batch dim dropped.

        The returned arrays are copies, safe to keep after the next start().
        """
        self.stream.synchronize()
        return {name: self.host[name][0].copy() for name in self.outputs}

    def infer(self, rgb):
        self.start(rgb)
        return self.wait()

    def close(self):
        if self.cuda_ctx is None:
            return
        # Release TRT objects and device buffers before popping the context.
        self.ctx = None
        self.engine = None
        self.dev.clear()
        self.host.clear()
        self.stream = None
        self.cuda_ctx.pop()
        self.cuda_ctx.detach()
        self.cuda_ctx = None
