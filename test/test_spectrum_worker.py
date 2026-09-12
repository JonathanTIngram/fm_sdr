"""Test for SpectrumWorker's queue plumbing in spectrum_worker.py.

The actual DSP (compute_spectrum_db, Waterfall) is already covered by
test_fm_visuals.py; this just checks the thread correctly drains samples
fed to it and produces spectrum rows, decoupled from whoever feeds it.
Shared by both fm_waterfall.py and fm_spectrum.py, each of which owns its
own separate SpectrumWorker instance/thread.
"""
import queue

import numpy as np
import pytest

from PyQt6.QtWidgets import QApplication

from spectrum_worker import SpectrumWorker


@pytest.fixture(scope="module", autouse=True)
def qapp():
    # QThread needs a QCoreApplication/QApplication instance to exist.
    app = QApplication.instance() or QApplication([])
    yield app


def test_spectrum_worker_produces_a_row_per_fed_block():
    worker = SpectrumWorker(nfft=256)
    worker.start()
    try:
        samples = (np.random.randn(4096) + 1j * np.random.randn(4096)).astype(np.complex64)
        worker.feed_samples(samples)
        spectrum_db = worker.spectrum_queue.get(timeout=2)
        assert len(spectrum_db) == 256
        assert np.all(np.isfinite(spectrum_db))
    finally:
        worker.stop()
        worker.wait(2000)


def test_spectrum_worker_drops_blocks_when_queue_full_instead_of_blocking():
    worker = SpectrumWorker(nfft=256)
    # Don't start() the thread -- nothing drains sample_queue, so we can
    # test feed_samples()'s drop-on-full behavior deterministically.
    samples = np.zeros(4096, dtype=np.complex64)
    for _ in range(10):
        worker.feed_samples(samples)  # must never raise/block even once full
    assert worker.sample_queue.qsize() <= worker.sample_queue.maxsize
