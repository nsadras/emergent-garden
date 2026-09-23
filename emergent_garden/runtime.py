"""Execution settings and shutdown at complete ecological transactions."""

import os
import signal

import torch


def enable_cuda_replay(device, ecology_version):
    """V10+ CUDA worlds opt this process into deterministic PyTorch kernels.

    Seeds alone do not control the order of parallel reductions. Set the cuBLAS
    workspace before the world's first CUDA allocation. As with PyTorch's thread
    count, these are process settings; older worlds do not turn them back off.
    Hardware/library changes can still change results.
    """
    if ecology_version >= 10 and torch.device(device).type == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False


class StopFlag:
    def __init__(self):
        self.requested = False
        self.previous = {}
        for sig in (signal.SIGINT, signal.SIGTERM):
            self.previous[sig] = signal.signal(sig, self._request)

    def _request(self, signum, frame):
        self.requested = True

    def close(self):
        for sig, handler in self.previous.items():
            signal.signal(sig, handler)
