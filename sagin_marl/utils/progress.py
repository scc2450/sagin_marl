from __future__ import annotations

import sys
import time


class Progress:
    def __init__(
        self,
        total: int,
        desc: str = "",
        width: int = 30,
        min_interval: float = 0.2,
        file=None,
    ) -> None:
        self.total = max(0, int(total))
        self.desc = desc
        self.width = max(10, int(width))
        self.min_interval = max(0.0, float(min_interval))
        self.file = file or sys.stdout
        self.inline = self._supports_inline_refresh(self.file)
        self.start = time.perf_counter()
        self.last = 0.0

    @staticmethod
    def _supports_inline_refresh(file) -> bool:
        try:
            return bool(file.isatty())
        except Exception:
            return False

    def update(self, current: int) -> None:
        now = time.perf_counter()
        if self.total <= 0:
            return
        if current < self.total and (now - self.last) < self.min_interval:
            return
        self.last = now
        frac = min(1.0, max(0.0, current / self.total))
        filled = int(self.width * frac)
        bar = "#" * filled + "-" * (self.width - filled)
        elapsed = max(1e-9, now - self.start)
        rate = current / elapsed
        eta = (self.total - current) / rate if rate > 0 else 0.0
        if self.inline:
            msg = f"\r{self.desc} [{bar}] {current}/{self.total} ({rate:.2f}/s, ETA {eta:.1f}s)"
        else:
            msg = f"{self.desc} [{bar}] {current}/{self.total} ({rate:.2f}/s, ETA {eta:.1f}s)\n"
        self.file.write(msg)
        self.file.flush()

    def close(self) -> None:
        self.update(self.total)
        if self.inline:
            self.file.write("\n")
            self.file.flush()
