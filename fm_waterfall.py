"""Waterfall spectrum display: a standalone window with its own background
thread (spectrum_worker.SpectrumWorker), entirely decoupled from the
audio-critical SDR reader thread. See spectrum_worker.py for why.
"""
import queue

import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QSizeGrip, QSizePolicy, QVBoxLayout, QWidget

from fm_visuals import make_colormap_lut, Waterfall
from spectrum_worker import SpectrumWorker

NFFT = 2048
WATERFALL_ROWS = 200
DISPLAY_SIZE = (760, 260)
# fm_gui.py's SdrWorker now feeds samples in sub-chunks every ~0.1s (see
# VISUAL_CHUNKS_PER_BLOCK there), so new rows arrive at ~10Hz; match the
# redraw rate to that instead of drawing at the old 5Hz and just letting
# rows queue up between redraws.
REFRESH_MS = 100
MIN_DB = -20
MAX_DB = 60


class WaterfallWindow(QWidget):
    """Standalone window showing a scrolling RF spectrum. Owns its own
    SpectrumWorker thread; feed it raw IQ blocks from wherever the SDR is
    actually being read via feed_samples()."""

    def __init__(self, sample_rate, parent=None):
        super().__init__(parent)
        self.setWindowTitle("RF Waterfall")

        self.lut = make_colormap_lut()
        self.waterfall = Waterfall(NFFT, WATERFALL_ROWS, min_db=MIN_DB, max_db=MAX_DB)

        self.image_label = QLabel()
        self.image_label.setScaledContents(True)
        self.image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        # setScaledContents only affects how the pixmap is *painted* --
        # QLabel still reports the pixmap's native resolution (NFFT=2048px
        # wide) as its minimumSizeHint(), which the layout would otherwise
        # enforce as the window's hard minimum width. An explicit small
        # minimum overrides that fallback so the window can actually shrink
        # (and a WM can snap/tile it, which it refuses to do for a window
        # it thinks has a fixed multi-thousand-pixel minimum width).
        self.image_label.setMinimumSize(1, 1)

        half_span_khz = sample_rate / 2 / 1000
        self.image_label.setToolTip(
            f"Time flows downward. Shows signal power across the "
            f"±{half_span_khz:.0f} kHz being sampled around the tuned "
            f"frequency — bright bands are stations/signals nearby."
        )

        # Some window managers (notably under Wayland) don't register a Qt
        # window's side edges as resize-drag zones, even though top/bottom
        # work fine. QSizeGrip is handled entirely by Qt itself rather than
        # relying on the WM's own border hit-testing, so it resizes reliably
        # regardless of that.
        grip_row = QHBoxLayout()
        grip_row.addStretch()
        grip_row.addWidget(QSizeGrip(self), 0, Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignRight)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.image_label)
        layout.addLayout(grip_row)
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
        got_row = False
        while True:
            try:
                spectrum_db = self.worker.spectrum_queue.get_nowait()
            except queue.Empty:
                break
            self.waterfall.push(spectrum_db)
            got_row = True

        if not got_row:
            return

        image = np.ascontiguousarray(self.waterfall.to_rgb_image(self.lut))
        height, width, _ = image.shape
        qimage = QImage(
            image.data, width, height, 3 * width, QImage.Format.Format_RGB888
        ).copy()  # copy: detach from `image`'s buffer before it's garbage collected
        self.image_label.setPixmap(QPixmap.fromImage(qimage))

    def closeEvent(self, event):
        self._timer.stop()
        self.worker.stop()
        self.worker.wait(2000)
        event.accept()
