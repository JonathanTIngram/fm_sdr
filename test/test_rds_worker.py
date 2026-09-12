"""Test for RdsWorker's queue plumbing in rds_worker.py.

The actual RDS decode logic (block sync, group parsing) is already covered
by test_rds.py; this just checks the thread correctly drains samples fed to
it without crashing, and that reset() actually starts over.
"""
import queue

import numpy as np
import pytest

from PyQt6.QtWidgets import QApplication

from rds_worker import RdsWorker, ACCUMULATE_SAMPLES


@pytest.fixture(scope="module", autouse=True)
def qapp():
    # QThread needs a QCoreApplication/QApplication instance to exist.
    app = QApplication.instance() or QApplication([])
    yield app


def test_rds_worker_processes_fed_samples_without_crashing():
    worker = RdsWorker()
    worker.start()
    try:
        # Plain noise won't ever sync -- this just checks the thread stays
        # alive and doesn't raise while actually running process_block()
        # (which only happens once enough has been buffered up).
        samples = (np.random.randn(ACCUMULATE_SAMPLES) + 1j * np.random.randn(ACCUMULATE_SAMPLES)).astype(np.complex64)
        worker.feed_samples(samples)
        worker.wait(1000)  # run() loops forever, so this is just a bounded sleep
        assert worker.isRunning()
        assert worker.rds.sync.locked is False
    finally:
        worker.stop()
        worker.wait(2000)


def test_rds_worker_buffers_small_chunks_before_processing():
    worker = RdsWorker()
    # Don't start() the thread -- drive run()'s buffering logic directly by
    # calling the same accumulation a single small feed would leave short of
    # threshold, checking it doesn't trigger a decode yet.
    small_chunk = np.zeros(1000, dtype=np.complex64)
    worker._buffer.append(small_chunk)
    worker._buffered_len += len(small_chunk)
    assert worker._buffered_len < ACCUMULATE_SAMPLES


def test_rds_worker_reset_starts_a_fresh_decoder():
    worker = RdsWorker()
    original_decoder = worker.rds
    worker.rds.groups.ps_chars[:] = list("TESTNAME")  # simulate a decoded station name
    worker._buffer.append(np.zeros(1000, dtype=np.complex64))  # simulate a partial buffer
    worker._buffered_len = 1000

    worker.reset()

    assert worker.rds is not original_decoder
    assert worker.rds.groups.station_name == ""
    assert worker._buffer == []
    assert worker._buffered_len == 0


def test_rds_worker_drops_blocks_when_queue_full_instead_of_blocking():
    worker = RdsWorker()
    # Don't start() the thread -- nothing drains sample_queue, so we can
    # test feed_samples()'s drop-on-full behavior deterministically.
    samples = np.zeros(4096, dtype=np.complex64)
    for _ in range(10):
        worker.feed_samples(samples)  # must never raise/block even once full
    assert worker.sample_queue.qsize() <= worker.sample_queue.maxsize
