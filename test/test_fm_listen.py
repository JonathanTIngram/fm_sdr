"""Unit tests for the FM demodulation logic in fm_listen.py.

These only exercise the pure fm_demodulate() function with synthetic IQ
signals -- no RTL-SDR hardware or audio device is touched.
"""
import queue

import numpy as np
import pytest

from fm_listen import fm_demodulate, AudioPlaybackBuffer, SDR_SAMPLE_RATE, DECIM, MAX_DEVIATION_HZ


def make_fm_signal(fs, duration, message_freq, deviation, n_samples=None):
    """Build a synthetic complex-baseband FM signal.

    message_freq: frequency (Hz) of the sinusoidal modulating tone.
    deviation: peak frequency deviation (Hz) imparted on the carrier.
    """
    n = n_samples if n_samples is not None else int(fs * duration)
    t = np.arange(n) / fs
    inst_freq = deviation * np.sin(2 * np.pi * message_freq * t)
    phase = 2 * np.pi * np.cumsum(inst_freq) / fs
    return np.exp(1j * phase)


def test_output_is_float32():
    samples = make_fm_signal(SDR_SAMPLE_RATE, 0.05, message_freq=1000, deviation=5000)
    audio = fm_demodulate(samples, DECIM)
    assert audio.dtype == np.float32


def test_output_length_matches_decimation():
    samples = make_fm_signal(SDR_SAMPLE_RATE, 0.05, message_freq=1000, deviation=5000)
    audio = fm_demodulate(samples, DECIM)
    expected_len = len(samples) // DECIM
    # scipy.signal.decimate's exact output length can be off by a sample or two
    assert abs(len(audio) - expected_len) <= 2


def test_amplitude_scales_with_deviation_not_block_peak():
    # A signal at a fraction of the standard 75kHz max deviation should map
    # to roughly that same fraction of full scale -- not get rescaled up to
    # a fixed peak regardless of this block's own loudness.
    deviation = 5000
    samples = make_fm_signal(SDR_SAMPLE_RATE, 0.05, message_freq=1000, deviation=deviation)
    audio = fm_demodulate(samples, DECIM)
    expected_peak = deviation / MAX_DEVIATION_HZ
    assert np.max(np.abs(audio)) == pytest.approx(expected_peak, rel=0.05)


def test_full_deviation_maps_to_full_scale():
    samples = make_fm_signal(SDR_SAMPLE_RATE, 0.05, message_freq=1000, deviation=MAX_DEVIATION_HZ)
    audio = fm_demodulate(samples, DECIM)
    assert np.max(np.abs(audio)) == pytest.approx(1.0, rel=0.05)


def test_two_blocks_with_different_loudness_get_consistent_gain():
    # This is the actual bug being guarded against: independently
    # normalizing each block to the same fixed peak would make a quiet
    # block just as loud as an adjacent loud one, causing an audible gain
    # jump at every block boundary. Gain must stay proportional across
    # blocks instead.
    quiet = fm_demodulate(make_fm_signal(SDR_SAMPLE_RATE, 0.05, 1000, deviation=2000), DECIM)
    loud = fm_demodulate(make_fm_signal(SDR_SAMPLE_RATE, 0.05, 1000, deviation=20000), DECIM)
    ratio = np.max(np.abs(loud)) / np.max(np.abs(quiet))
    assert ratio == pytest.approx(10.0, rel=0.05)


def test_output_clipped_to_valid_audio_range():
    samples = make_fm_signal(SDR_SAMPLE_RATE, 0.05, message_freq=1000, deviation=200_000)
    audio = fm_demodulate(samples, DECIM)
    assert np.max(np.abs(audio)) <= 1.0


def test_constant_frequency_input_has_no_nan_or_inf():
    n = int(SDR_SAMPLE_RATE * 0.05)
    t = np.arange(n) / SDR_SAMPLE_RATE
    samples = np.exp(1j * 2 * np.pi * 1000 * t)  # fixed frequency offset, no modulation
    audio = fm_demodulate(samples, DECIM)
    assert np.all(np.isfinite(audio))


def test_recovers_modulating_tone_frequency():
    message_freq = 1000  # Hz, within audio range
    samples = make_fm_signal(SDR_SAMPLE_RATE, 0.2, message_freq=message_freq, deviation=5000)
    audio = fm_demodulate(samples, DECIM)

    audio_rate = SDR_SAMPLE_RATE / DECIM
    spectrum = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(len(audio), d=1 / audio_rate)

    # Ignore the DC bin when looking for the dominant tone.
    peak_freq = freqs[1:][np.argmax(spectrum[1:])]

    assert peak_freq == pytest.approx(message_freq, abs=50)


def make_outdata(frames):
    return np.zeros((frames, 1), dtype=np.float32)


def test_playback_buffer_exact_block():
    q = queue.Queue()
    q.put(np.array([1, 2, 3, 4], dtype=np.float32))
    player = AudioPlaybackBuffer(q, volume=1.0)

    outdata = make_outdata(4)
    player.callback(outdata, 4, None, None)

    assert np.array_equal(outdata[:, 0], [1, 2, 3, 4])
    assert q.empty()


def test_playback_buffer_leftover_carries_over_to_next_callback():
    q = queue.Queue()
    q.put(np.array([1, 2, 3, 4], dtype=np.float32))
    player = AudioPlaybackBuffer(q, volume=1.0)

    first = make_outdata(2)
    player.callback(first, 2, None, None)
    assert np.array_equal(first[:, 0], [1, 2])
    assert q.empty()  # the block was already pulled off the queue

    second = make_outdata(2)
    player.callback(second, 2, None, None)
    assert np.array_equal(second[:, 0], [3, 4])


def test_playback_buffer_spans_multiple_queued_blocks():
    q = queue.Queue()
    q.put(np.array([1, 2], dtype=np.float32))
    q.put(np.array([3, 4, 5], dtype=np.float32))
    player = AudioPlaybackBuffer(q, volume=1.0)

    outdata = make_outdata(5)
    player.callback(outdata, 5, None, None)

    assert np.array_equal(outdata[:, 0], [1, 2, 3, 4, 5])
    assert q.empty()


def test_playback_buffer_underrun_pads_with_silence():
    q = queue.Queue()  # empty: simulates the SDR/demod side falling behind
    player = AudioPlaybackBuffer(q, volume=1.0)

    outdata = make_outdata(4)
    player.callback(outdata, 4, None, None)

    assert np.array_equal(outdata[:, 0], [0, 0, 0, 0])


def test_playback_buffer_applies_volume():
    q = queue.Queue()
    q.put(np.array([1, 2, 3, 4], dtype=np.float32))
    player = AudioPlaybackBuffer(q, volume=0.5)

    outdata = make_outdata(4)
    player.callback(outdata, 4, None, None)

    assert np.array_equal(outdata[:, 0], [0.5, 1.0, 1.5, 2.0])


def test_playback_buffer_volume_change_applies_immediately():
    # A block already sitting in _leftover should still be scaled by whatever
    # volume is current at write time, not the volume when it was queued.
    q = queue.Queue()
    q.put(np.array([1, 2, 3, 4], dtype=np.float32))
    player = AudioPlaybackBuffer(q, volume=1.0)

    first = make_outdata(2)
    player.callback(first, 2, None, None)  # pulls the block into _leftover, consumes [1, 2]

    player.volume = 0.0  # e.g. user drags the volume knob to zero

    second = make_outdata(2)
    player.callback(second, 2, None, None)  # remaining [3, 4] from the same block
    assert np.array_equal(second[:, 0], [0.0, 0.0])
