"""Peak GPU memory, measured the same way for in-process (HF) and subprocess (llama-server) models."""
import os
import threading
import time

import pynvml


def physical_gpu():
    """NVML and nvidia-smi ignore CUDA_VISIBLE_DEVICES, so map it ourselves: CUDA_VISIBLE_DEVICES=1 -> GPU 1."""
    return int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])


class PeakGpuMemory:
    """Samples used memory on this process's GPU every 50 ms. `peak_gb` = highest level above the level at start.

        with PeakGpuMemory() as mem:
            ...load and run a model...
        print(mem.peak_gb)
    """

    def __enter__(self):
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(physical_gpu())
        self.start = self.peak = self._used()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def _used(self):
        return pynvml.nvmlDeviceGetMemoryInfo(self.handle).used

    def _sample(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, self._used())
            time.sleep(0.05)

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()

    @property
    def start_gb(self):
        return round(self.start / 1e9, 3)

    @property
    def peak_gb(self):
        return round((self.peak - self.start) / 1e9, 3)
