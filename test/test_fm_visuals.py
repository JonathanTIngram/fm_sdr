"""Unit tests for the waterfall's DSP/rendering logic in fm_visuals.py.

Pure numpy -- no Qt, SDR, or audio device involved.
"""
import numpy as np
import pytest

from fm_visuals import (
    compute_spectrum_db, estimate_channel_prominence_db, make_colormap_lut, Waterfall,
)


def test_spectrum_output_length_is_nfft():
    samples = np.random.randn(20000) + 1j * np.random.randn(20000)
    spectrum = compute_spectrum_db(samples, nfft=2048)
    assert len(spectrum) == 2048


def test_spectrum_has_no_nan_or_inf_on_silence():
    samples = np.zeros(4096, dtype=np.complex64)
    spectrum = compute_spectrum_db(samples, nfft=2048)
    assert np.all(np.isfinite(spectrum))


def test_spectrum_peak_locates_tone_frequency():
    fs = 1_024_000
    nfft = 2048
    tone_freq = 100_000  # Hz offset from center
    n = nfft * 50  # enough segments for Welch averaging
    t = np.arange(n) / fs
    samples = np.exp(1j * 2 * np.pi * tone_freq * t)

    spectrum = compute_spectrum_db(samples, nfft=nfft)
    freq_bins = np.fft.fftshift(np.fft.fftfreq(nfft, d=1 / fs))
    peak_freq = freq_bins[np.argmax(spectrum)]

    assert peak_freq == pytest.approx(tone_freq, abs=fs / nfft * 2)


def test_channel_prominence_detects_elevated_center():
    fs = 1_024_000
    nfft = 1024
    bin_hz = fs / nfft
    center = nfft // 2
    half_bw_bins = int(100_000 / bin_hz)

    spectrum = np.full(nfft, -50.0, dtype=np.float32)  # flat noise floor
    spectrum[center - half_bw_bins:center + half_bw_bins] = -10.0  # broad hump at the tuned center

    prominence = estimate_channel_prominence_db(spectrum, fs)
    assert prominence == pytest.approx(40, abs=1)


def test_channel_prominence_near_zero_for_flat_spectrum():
    fs = 1_024_000
    spectrum = np.full(1024, -30.0, dtype=np.float32)

    prominence = estimate_channel_prominence_db(spectrum, fs)
    assert prominence == pytest.approx(0, abs=0.5)


def test_channel_prominence_ignores_hump_outside_channel_and_floor_bands():
    fs = 1_024_000
    nfft = 1024
    bin_hz = fs / nfft
    center = nfft // 2

    spectrum = np.full(nfft, -50.0, dtype=np.float32)
    # a strong hump between the channel window and the floor bands -- not
    # the tuned frequency, so it shouldn't move the prominence estimate
    lo = center + int(150_000 / bin_hz)
    hi = center + int(250_000 / bin_hz)
    spectrum[lo:hi] = 0.0

    prominence = estimate_channel_prominence_db(spectrum, fs)
    assert abs(prominence) < 1


def test_colormap_lut_shape_and_range():
    lut = make_colormap_lut(n=256)
    assert lut.shape == (256, 3)
    assert lut.dtype == np.uint8
    assert lut.min() >= 0 and lut.max() <= 255


def test_waterfall_push_scrolls_and_places_newest_row_last():
    wf = Waterfall(nfft=4, num_rows=3, min_db=-20, max_db=60)
    wf.push(np.array([1, 1, 1, 1], dtype=np.float32))
    wf.push(np.array([2, 2, 2, 2], dtype=np.float32))

    assert np.array_equal(wf.rows[-1], [2, 2, 2, 2])
    assert np.array_equal(wf.rows[-2], [1, 1, 1, 1])


def test_waterfall_discards_oldest_row_once_full():
    wf = Waterfall(nfft=2, num_rows=2, min_db=-20, max_db=60)
    wf.push(np.array([1, 1], dtype=np.float32))
    wf.push(np.array([2, 2], dtype=np.float32))
    wf.push(np.array([3, 3], dtype=np.float32))  # the "1" row should fall off

    assert np.array_equal(wf.rows[0], [2, 2])
    assert np.array_equal(wf.rows[1], [3, 3])


def test_waterfall_image_clips_extreme_values_to_lut_ends():
    wf = Waterfall(nfft=2, num_rows=1, min_db=-20, max_db=60)
    wf.push(np.array([-1000, 1000], dtype=np.float32))  # far below/above range
    lut = make_colormap_lut(n=256)

    image = wf.to_rgb_image(lut)
    assert np.array_equal(image[0, 0], lut[0])
    assert np.array_equal(image[0, 1], lut[-1])


def test_waterfall_image_shape_and_dtype():
    wf = Waterfall(nfft=8, num_rows=5)
    wf.push(np.zeros(8, dtype=np.float32))
    lut = make_colormap_lut(n=256)

    image = wf.to_rgb_image(lut)
    assert image.shape == (5, 8, 3)
    assert image.dtype == np.uint8
