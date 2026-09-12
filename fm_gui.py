"""PyQt6 FM tuner: turn the frequency and volume knobs, audio plays continuously.

Usage:
    python3 fm_gui.py
"""
import sys
import queue
import threading

import numpy as np
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import QApplication, QDial, QHBoxLayout, QLabel, QVBoxLayout, QWidget
import sounddevice as sd
from rtlsdr import RtlSdr

from fm_listen import (
    fm_demodulate,
    AudioPlaybackBuffer,
    DEFAULT_FREQ_MHZ,
    SDR_SAMPLE_RATE,
    AUDIO_SAMPLE_RATE,
    DECIM,
    SAMPLES_PER_BLOCK,
    QUEUE_MAXSIZE,
    OUTPUT_LATENCY_S,
)
from fm_waterfall import WaterfallWindow
from fm_spectrum import SpectrumAnalyzerWindow
from fm_rds import RdsWindow

FREQ_MIN_MHZ = 88.0
FREQ_MAX_MHZ = 108.0
FREQ_STEP_MHZ = 0.1
DEFAULT_VOLUME_PCT = 80

# on_samples (waterfall/spectrum feed) only gets called once per full block
# read -- once every SECONDS_PER_BLOCK (0.5s) -- which visibly bottlenecks
# those windows' update rate no matter how fast their own threads are.
# Reading in smaller sub-chunks and calling on_samples after each one gets
# them updates 5x more often, while still concatenating back into the same
# full block for fm_demodulate(), so the audio path is byte-for-byte
# unchanged. read_samples() streams via rtlsdr_read_sync with no meaningful
# per-call overhead, so this is essentially free.
VISUAL_CHUNKS_PER_BLOCK = 5
VISUAL_CHUNK_SAMPLES = SAMPLES_PER_BLOCK // VISUAL_CHUNKS_PER_BLOCK


class SdrWorker(QThread):
    """Owns the RTL-SDR device and audio stream, running the read/demod/play
    loop on its own thread so the GUI never blocks. Frequency changes from
    the GUI thread are applied under a lock since read_samples() and
    center_freq both touch the same USB device handle. Volume changes go
    straight to the AudioPlaybackBuffer, which applies them on PortAudio's
    thread with no lock needed (see AudioPlaybackBuffer.volume).

    Deliberately minimal: just read -> demodulate -> enqueue. Anything
    extra done per block (spectrum analysis, RDS decoding, etc.) holds
    Python's GIL long enough to occasionally starve the audio callback
    thread of scheduling, which is audible as stuttering even with a
    generous output buffer. `on_samples`, if given, is called once per
    sub-chunk right after reading it (before demodulation) -- keep whatever
    it does cheap (e.g. a queue put for some other thread to consume),
    since it still runs inline on this thread. Reading in
    VISUAL_CHUNKS_PER_BLOCK sub-chunks instead of one read_samples() call
    per full block means on_samples fires that many times more often,
    without changing the block fm_demodulate() ever sees."""

    error = pyqtSignal(str)

    def __init__(self, initial_freq_hz, initial_volume, on_samples=None):
        super().__init__()
        self._lock = threading.Lock()
        self._freq_hz = initial_freq_hz
        self._volume = initial_volume
        self._running = True
        self._on_samples = on_samples
        self.sdr = None
        self.player = None

    def set_freq(self, freq_hz):
        with self._lock:
            self._freq_hz = freq_hz
            if self.sdr is not None:
                self.sdr.center_freq = freq_hz

    def set_volume(self, volume):
        self._volume = volume
        if self.player is not None:
            self.player.volume = volume

    def stop(self):
        self._running = False

    def run(self):
        stream = None
        try:
            self.sdr = RtlSdr()
            self.sdr.sample_rate = SDR_SAMPLE_RATE
            with self._lock:
                self.sdr.center_freq = self._freq_hz
            self.sdr.gain = 'auto'

            audio_queue = queue.Queue(maxsize=QUEUE_MAXSIZE)
            self.player = AudioPlaybackBuffer(audio_queue, volume=self._volume)
            stream = sd.OutputStream(
                samplerate=AUDIO_SAMPLE_RATE, channels=1, dtype='float32',
                callback=self.player.callback, latency=OUTPUT_LATENCY_S,
            )
            stream.start()

            while self._running:
                chunks = []
                for _ in range(VISUAL_CHUNKS_PER_BLOCK):
                    with self._lock:
                        chunk = self.sdr.read_samples(VISUAL_CHUNK_SAMPLES)
                    if self._on_samples is not None:
                        self._on_samples(chunk)
                    chunks.append(chunk)
                samples = np.concatenate(chunks)

                audio = fm_demodulate(samples, DECIM)
                try:
                    audio_queue.put(audio, timeout=1)
                except queue.Full:
                    pass  # consumer stalled; drop this block rather than build up latency
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            if stream is not None:
                stream.stop()
                stream.close()
            if self.sdr is not None:
                self.sdr.close()


def _make_knob_panel(title, dial, value_label):
    value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    font = value_label.font()
    font.setPointSize(20)
    value_label.setFont(font)

    dial.setNotchesVisible(True)
    dial.setFixedSize(150, 150)

    name_label = QLabel(title)
    name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

    panel = QVBoxLayout()
    panel.addWidget(value_label)
    panel.addWidget(dial, alignment=Qt.AlignmentFlag.AlignCenter)
    panel.addWidget(name_label)
    return panel


class TunerWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FM Tuner")

        # Frequency knob
        steps_min = round(FREQ_MIN_MHZ / FREQ_STEP_MHZ)
        steps_max = round(FREQ_MAX_MHZ / FREQ_STEP_MHZ)

        self.freq_label = QLabel(f"{DEFAULT_FREQ_MHZ:.1f} MHz")
        self.freq_dial = QDial()
        self.freq_dial.setMinimum(steps_min)
        self.freq_dial.setMaximum(steps_max)
        self.freq_dial.setValue(round(DEFAULT_FREQ_MHZ / FREQ_STEP_MHZ))
        self.freq_dial.valueChanged.connect(self._on_freq_changed)
        freq_panel = _make_knob_panel("Frequency", self.freq_dial, self.freq_label)

        # Volume knob
        self.volume_label = QLabel(f"{DEFAULT_VOLUME_PCT}%")
        self.volume_dial = QDial()
        self.volume_dial.setMinimum(0)
        self.volume_dial.setMaximum(100)
        self.volume_dial.setValue(DEFAULT_VOLUME_PCT)
        self.volume_dial.valueChanged.connect(self._on_volume_changed)
        volume_panel = _make_knob_panel("Volume", self.volume_dial, self.volume_label)

        knobs_row = QHBoxLayout()
        knobs_row.addLayout(freq_panel)
        knobs_row.addLayout(volume_panel)

        self.setLayout(knobs_row)
        self.resize(420, 260)

        # Separate windows, separate threads (fm_waterfall.py, fm_spectrum.py,
        # fm_rds.py) -- each fed raw IQ blocks via on_samples below, never
        # sharing the audio thread's own loop with any of their processing,
        # and fully independent of each other too (own worker thread each).
        self.waterfall_window = WaterfallWindow(SDR_SAMPLE_RATE)
        self.waterfall_window.show()

        self.spectrum_window = SpectrumAnalyzerWindow(SDR_SAMPLE_RATE)
        self.spectrum_window.show()

        self.rds_window = RdsWindow()
        self.rds_window.show()

        def on_samples(samples):
            self.waterfall_window.feed_samples(samples)
            self.spectrum_window.feed_samples(samples)
            self.rds_window.feed_samples(samples)

        self.worker = SdrWorker(
            DEFAULT_FREQ_MHZ * 1e6, DEFAULT_VOLUME_PCT / 100,
            on_samples=on_samples,
        )
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_freq_changed(self, steps):
        freq_mhz = steps * FREQ_STEP_MHZ
        self.freq_label.setText(f"{freq_mhz:.1f} MHz")
        self.worker.set_freq(freq_mhz * 1e6)
        self.rds_window.reset()  # station changed; old PS/RadioText no longer applies

    def _on_volume_changed(self, pct):
        self.volume_label.setText(f"{pct}%")
        self.worker.set_volume(pct / 100)

    def _on_error(self, message):
        self.freq_label.setText(f"Error: {message}")

    def closeEvent(self, event):
        self.worker.stop()
        self.worker.wait(2000)
        self.waterfall_window.close()
        self.spectrum_window.close()
        self.rds_window.close()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = TunerWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
