"""PyQt6 FM tuner: turn the frequency and volume knobs, audio plays continuously.

Usage:
    python3 fm_gui.py
"""
import sys
import queue
import threading

import numpy as np
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication, QDial, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget,
)
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
from fm_visuals import compute_spectrum_db, estimate_channel_prominence_db
from fm_waterfall import WaterfallWindow
from fm_spectrum import SpectrumAnalyzerWindow
from fm_rds import RdsWindow

FREQ_MIN_MHZ = 88.0
FREQ_MAX_MHZ = 108.0
FREQ_STEP_MHZ = 0.1
DEFAULT_VOLUME_PCT = 80

# Seek scans one FREQ_STEP_MHZ at a time, reading a short burst at each
# candidate frequency and checking for a broad power hump at the tuned
# center (see fm_visuals.estimate_channel_prominence_db) rather than
# demodulating -- much cheaper per step, and works regardless of AGC gain.
SEEK_NFFT = 1024
SEEK_SETTLE_SAMPLES = 4096   # discarded: lets the tuner's PLL settle after retuning
SEEK_MEASURE_SAMPLES = SEEK_NFFT * 16
SEEK_PROMINENCE_DB = 10.0    # empirical: real broadcast signals sit well above this

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
    seek_progress = pyqtSignal(float)  # MHz, emitted as each candidate frequency is checked
    seek_finished = pyqtSignal(float)  # MHz, the frequency seeking landed on

    def __init__(self, initial_freq_hz, initial_volume, on_samples=None):
        super().__init__()
        self._lock = threading.Lock()
        self._freq_hz = initial_freq_hz
        self._volume = initial_volume
        self._running = True
        self._on_samples = on_samples
        self.sdr = None
        self.player = None

        self._seeking = False
        self._seek_direction = 0
        self._seek_start_freq_hz = None
        self._seek_steps_taken = 0
        self._seek_total_steps = 0

    def set_freq(self, freq_hz):
        with self._lock:
            self._freq_hz = freq_hz
            if self.sdr is not None:
                self.sdr.center_freq = freq_hz

    def set_volume(self, volume):
        self._volume = volume
        if self.player is not None:
            self.player.volume = volume

    def seek(self, direction):
        """Start scanning up (+1) or down (-1) from the current frequency
        for the next frequency with a real signal on it. Muted (nothing
        gets enqueued to play) until it lands, since scanning necessarily
        passes over a lot of static. No-op if already seeking."""
        with self._lock:
            if self._seeking:
                return
            self._seeking = True
            self._seek_direction = direction
            self._seek_start_freq_hz = self._freq_hz
            self._seek_steps_taken = 0
            self._seek_total_steps = round((FREQ_MAX_MHZ - FREQ_MIN_MHZ) / FREQ_STEP_MHZ) + 1

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
                if self._seeking:
                    self._seek_step()
                    continue

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

    def _seek_step(self):
        """Check one candidate frequency, then either stop (found a signal,
        or scanned the whole band and gave up -- back to where it started)
        or leave self._seeking set so the next run() iteration checks the
        next one. Runs entirely on this thread; no audio gets enqueued
        while seeking, so playback drains to silence on its own."""
        step_hz = FREQ_STEP_MHZ * 1e6
        min_hz, max_hz = FREQ_MIN_MHZ * 1e6, FREQ_MAX_MHZ * 1e6

        with self._lock:
            next_freq = self._freq_hz + self._seek_direction * step_hz
            if next_freq > max_hz + 1:
                next_freq = min_hz
            elif next_freq < min_hz - 1:
                next_freq = max_hz
            self._freq_hz = next_freq
            self.sdr.center_freq = next_freq
            self.sdr.read_samples(SEEK_SETTLE_SAMPLES)
            measured = self.sdr.read_samples(SEEK_MEASURE_SAMPLES)

        self.seek_progress.emit(next_freq / 1e6)
        if self._on_samples is not None:
            self._on_samples(measured)  # keep spectrum/waterfall/RDS windows live during the scan

        self._seek_steps_taken += 1
        spectrum_db = compute_spectrum_db(measured, SEEK_NFFT)
        prominence = estimate_channel_prominence_db(spectrum_db, SDR_SAMPLE_RATE)
        found = prominence > SEEK_PROMINENCE_DB
        exhausted = self._seek_steps_taken >= self._seek_total_steps

        if found or exhausted:
            if not found:
                with self._lock:
                    self._freq_hz = self._seek_start_freq_hz
                    self.sdr.center_freq = self._freq_hz
            self._seeking = False
            self.seek_finished.emit(self._freq_hz / 1e6)


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

        self.seek_down_btn = QPushButton("◀ Seek")
        self.seek_up_btn = QPushButton("Seek ▶")
        self.seek_down_btn.clicked.connect(lambda: self._start_seek(-1))
        self.seek_up_btn.clicked.connect(lambda: self._start_seek(1))
        seek_row = QHBoxLayout()
        seek_row.addWidget(self.seek_down_btn)
        seek_row.addWidget(self.seek_up_btn)
        freq_column = QVBoxLayout()
        freq_column.addLayout(freq_panel)
        freq_column.addLayout(seek_row)

        # Volume knob
        self.volume_label = QLabel(f"{DEFAULT_VOLUME_PCT}%")
        self.volume_dial = QDial()
        self.volume_dial.setMinimum(0)
        self.volume_dial.setMaximum(100)
        self.volume_dial.setValue(DEFAULT_VOLUME_PCT)
        self.volume_dial.valueChanged.connect(self._on_volume_changed)
        volume_panel = _make_knob_panel("Volume", self.volume_dial, self.volume_label)

        knobs_row = QHBoxLayout()
        knobs_row.addLayout(freq_column)
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
        self.worker.seek_progress.connect(self._on_seek_progress)
        self.worker.seek_finished.connect(self._on_seek_finished)
        self.worker.start()

    def _start_seek(self, direction):
        self.seek_down_btn.setEnabled(False)
        self.seek_up_btn.setEnabled(False)
        self.freq_dial.setEnabled(False)
        self.worker.seek(direction)

    def _on_seek_progress(self, freq_mhz):
        self.freq_label.setText(f"{freq_mhz:.1f} MHz")

    def _on_seek_finished(self, freq_mhz):
        self.seek_down_btn.setEnabled(True)
        self.seek_up_btn.setEnabled(True)
        self.freq_dial.setEnabled(True)
        self.freq_label.setText(f"{freq_mhz:.1f} MHz")
        self.freq_dial.blockSignals(True)
        self.freq_dial.setValue(round(freq_mhz / FREQ_STEP_MHZ))
        self.freq_dial.blockSignals(False)
        self.rds_window.reset()  # scanning fed the RDS decoder noise from every skipped frequency

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
