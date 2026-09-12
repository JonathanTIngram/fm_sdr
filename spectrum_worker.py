"""Generic background-thread spectrum computation, shared by any window
that needs one (fm_waterfall.py, fm_spectrum.py) without those sibling
window modules depending on each other.

Only one thing can have the RTL-SDR device open at a time, so this never
touches the device itself -- whatever *is* reading it (fm_gui.py's
SdrWorker) hands raw IQ blocks to feed_samples(), which just does a cheap
queue put and returns immediately. The actual FFT/averaging work
(fm_visuals.compute_spectrum_db) runs on this worker's own thread, so it
never adds to the audio-critical thread's own loop iteration time -- that's
what caused audible stuttering when this ran inline with the audio path.
"""
import queue

from PyQt6.QtCore import QThread

from fm_visuals import compute_spectrum_db

DEFAULT_NFFT = 2048
SAMPLE_QUEUE_MAXSIZE = 2  # a little slack; drop blocks rather than lag behind


class SpectrumWorker(QThread):
    """Pulls raw IQ blocks off a queue and computes one spectrum row per
    block, entirely off the audio-critical thread."""

    def __init__(self, nfft=DEFAULT_NFFT):
        super().__init__()
        self.nfft = nfft
        self.sample_queue = queue.Queue(maxsize=SAMPLE_QUEUE_MAXSIZE)
        self.spectrum_queue = queue.Queue(maxsize=SAMPLE_QUEUE_MAXSIZE)
        self._running = True

    def feed_samples(self, samples):
        """Call from the SDR reader thread. Cheap: just a queue put; drops
        the block instead of blocking if this thread hasn't kept up."""
        try:
            self.sample_queue.put_nowait(samples)
        except queue.Full:
            pass

    def stop(self):
        self._running = False

    def run(self):
        while self._running:
            try:
                samples = self.sample_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            spectrum_db = compute_spectrum_db(samples, self.nfft)
            try:
                self.spectrum_queue.put_nowait(spectrum_db)
            except queue.Full:
                pass
