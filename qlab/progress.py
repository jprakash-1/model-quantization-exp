"""One tiny progress printer used everywhere. Plain lines (no \\r), so it reads the same in the notebook
and in the GPU-1 job's log file.

    bar = Progress(250, "GSM8K")
    for ...: bar.update(16, info="412 tok/s")
    bar.done()
    ->  GSM8K  96/250  38%  0:01:12 elapsed  ~0:01:56 left  412 tok/s
"""
import time


def hms(seconds):
    s = int(seconds)
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


class Progress:
    def __init__(self, total, label, every=15):
        self.total, self.label, self.every = total, label, every
        self.n, self.start, self.last, self.printed = 0, time.time(), 0.0, None
        self.info = ""

    def update(self, n=1, info=None):
        self.set(self.n + n, info)

    def set(self, n, info=None):
        self.n = min(n, self.total)
        self.info = info if info is not None else self.info
        if time.time() - self.last >= self.every:
            self._print()

    def done(self, info=None):
        if self.printed == self.total and info is None:
            return  # the 100% line is already out
        self.n = self.total
        self.info = info if info is not None else self.info
        self._print()

    def _print(self):
        self.last, self.printed = time.time(), self.n
        elapsed = self.last - self.start
        pct = 100 * self.n / self.total if self.total else 100
        left = f"  ~{hms(elapsed * (self.total - self.n) / self.n)} left" if 0 < self.n < self.total else ""
        print(f"    {self.label}  {self.n}/{self.total}  {pct:3.0f}%  {hms(elapsed)} elapsed{left}  {self.info}".rstrip(),
              flush=True)
