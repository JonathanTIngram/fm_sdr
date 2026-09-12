"""Background-thread RDS (station name / RadioText / raw packet log)
decoding, decoupled from the audio-critical SDR reader thread the same way
spectrum_worker.py is -- see that module's docstring for why.
"""
import queue

import numpy as np
from PyQt6.QtCore import QThread

from fm_listen import wideband_fm_demod, SDR_SAMPLE_RATE, SAMPLES_PER_BLOCK
from rds import RdsDecoder

SAMPLE_QUEUE_MAXSIZE = 8

# fm_gui.py feeds samples in small ~0.1s sub-chunks (for the
# waterfall/spectrum windows' sake), but rds.py's own subcarrier-mixing
# resample has no filter state carried between calls -- processing many
# small chunks independently means many more edge transients per second
# than processing fewer, larger ones, which measurably degrades bit
# recovery. Buffer sub-chunks back up to roughly the original block size
# before running the RDS pipeline, decoupling how often we're fed from how
# much we process at once.
ACCUMULATE_SAMPLES = SAMPLES_PER_BLOCK


class RdsWorker(QThread):
    """Pulls raw IQ blocks off a queue and feeds them through an
    RdsDecoder, entirely off the audio-critical thread. Read .rds directly
    for current station name/RadioText/sync state from another thread --
    plain attribute reads are safe/atomic under the GIL, matching the
    pattern used for AudioPlaybackBuffer.volume elsewhere."""

    def __init__(self):
        super().__init__()
        self.rds = RdsDecoder()
        self.sample_queue = queue.Queue(maxsize=SAMPLE_QUEUE_MAXSIZE)
        self._running = True
        self._buffer = []
        self._buffered_len = 0

    def feed_samples(self, samples):
        """Call from the SDR reader thread. Cheap: just a queue put; drops
        the block instead of blocking if this thread hasn't kept up."""
        try:
            self.sample_queue.put_nowait(samples)
        except queue.Full:
            pass

    def reset(self):
        """Call when the tuned frequency changes -- the old station's
        PS/RadioText no longer applies, and any partially-buffered chunk
        from it must be dropped, not concatenated with the new station's."""
        self.rds = RdsDecoder()
        self._buffer = []
        self._buffered_len = 0

    def stop(self):
        self._running = False

    def run(self):
        while self._running:
            try:
                samples = self.sample_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            self._buffer.append(samples)
            self._buffered_len += len(samples)
            if self._buffered_len < ACCUMULATE_SAMPLES:
                continue

            combined = np.concatenate(self._buffer)
            self._buffer = []
            self._buffered_len = 0

            demod = wideband_fm_demod(combined)
            self.rds.process_block(demod, SDR_SAMPLE_RATE)
