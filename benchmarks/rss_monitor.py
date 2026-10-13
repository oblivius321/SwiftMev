"""
SwiftMev — Monitor de Memória por Amostragem Contínua (psutil)
Amostra o RSS do processo em alta frequência (ex: a cada 2 ms) em thread em segundo plano,
registrando o pico de memória observado e metadados de coleta para transparência científica.
"""
from __future__ import annotations
import threading
import time
import psutil


class RSSSamplingMonitor:
    def __init__(self, interval_ms: float = 2.0):
        self.interval_s = interval_ms / 1000.0
        self.proc = psutil.Process()
        self._running = False
        self._thread: threading.Thread | None = None
        self.samples: list[float] = []
        self.initial_rss_mb: float = 0.0
        self.final_rss_mb: float = 0.0

    def _sample_loop(self):
        while self._running:
            try:
                rss_mb = self.proc.memory_info().rss / (1024 * 1024)
                self.samples.append(rss_mb)
            except Exception:
                pass
            time.sleep(self.interval_s)

    def __enter__(self):
        self.samples.clear()
        self.initial_rss_mb = round(self.proc.memory_info().rss / (1024 * 1024), 2)
        self._running = True
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=0.2)
        self.final_rss_mb = round(self.proc.memory_info().rss / (1024 * 1024), 2)

    def get_summary(self) -> dict:
        peak_mb = max(self.samples) if self.samples else max(self.initial_rss_mb, self.final_rss_mb)
        return {
            "rss_initial_mb": self.initial_rss_mb,
            "rss_final_mb": self.final_rss_mb,
            "rss_peak_observed_mb": round(peak_mb, 2),
            "rss_delta_mb": round(self.final_rss_mb - self.initial_rss_mb, 2),
            "sample_interval_ms": round(self.interval_s * 1000.0, 1),
            "sample_count": len(self.samples)
        }
