"""Spectrum analyzer display: a standalone window with its own background
thread (spectrum_worker.SpectrumWorker) -- a separate instance from
fm_waterfall.py's, so the two features are fully independent: closing one,
or a slowdown in one, has no effect on the other. See spectrum_worker.py
for why this computation runs off the audio-critical thread at all.
"""
import queue

import numpy as np
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QColor, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import QSizePolicy, QVBoxLayout, QWidget

from spectrum_worker import SpectrumWorker

NFFT = 2048
DISPLAY_SIZE = (760, 200)
REFRESH_MS = 200
MIN_DB = -20
MAX_DB = 60


class SpectrumBarWidget(QWidget):
    """Instantaneous FFT magnitude spectrum, drawn as a filled area graph --
    the classic "spectrum analyzer" view."""

    def __init__(self, min_db, max_db, parent=None):
        super().__init__(parent)
        self.min_db = min_db
        self.max_db = max_db
        self.spectrum_db = None
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumHeight(120)

    def set_spectrum(self, spectrum_db):
        self.spectrum_db = spectrum_db
        self.update()  # schedule a repaint

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(15, 15, 20))

        if self.spectrum_db is not None and self.width() > 0:
            w, h = self.width(), self.height()
            norm = np.clip((self.spectrum_db - self.min_db) / (self.max_db - self.min_db), 0.0, 1.0)
            xs = np.linspace(0, w, len(norm))
            ys = h - norm * h

            path = QPainterPath()
            path.moveTo(0, h)
            for x, y in zip(xs, ys):
                path.lineTo(x, y)
            path.lineTo(w, h)
            path.closeSubpath()

            painter.setPen(QPen(QColor(120, 210, 255), 1))
            painter.setBrush(QColor(40, 130, 200, 160))
            painter.drawPath(path)

        painter.end()


class SpectrumAnalyzerWindow(QWidget):
    """Standalone window showing the instantaneous RF spectrum. Owns its
    own SpectrumWorker thread; feed it raw IQ blocks from wherever the SDR
    is actually being read via feed_samples()."""

    def __init__(self, sample_rate, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Spectrum Analyzer")

        self.bar = SpectrumBarWidget(MIN_DB, MAX_DB)
        half_span_khz = sample_rate / 2 / 1000
        self.bar.setToolTip(
            f"Instantaneous signal power across the ±{half_span_khz:.0f} kHz "
            f"being sampled around the tuned frequency."
        )

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.bar)
        self.setLayout(layout)
        self.resize(*DISPLAY_SIZE)

        self.worker = SpectrumWorker(nfft=NFFT)
        self.worker.start()

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._redraw)
        self._timer.start(REFRESH_MS)

    def feed_samples(self, samples):
        self.worker.feed_samples(samples)

    def _redraw(self):
        latest_spectrum = None
        while True:
            try:
                latest_spectrum = self.worker.spectrum_queue.get_nowait()
            except queue.Empty:
                break

        if latest_spectrum is not None:
            self.bar.set_spectrum(latest_spectrum)

    def closeEvent(self, event):
        self._timer.stop()
        self.worker.stop()
        self.worker.wait(2000)
        event.accept()
