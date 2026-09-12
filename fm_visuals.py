"""Pure DSP/rendering helpers for the waterfall display.

Kept separate from fm_gui.py so the actual math (spectrum estimation, the
scrolling image buffer, color mapping) can be unit tested without touching
Qt, the SDR, or the audio device.
"""
import numpy as np
import matplotlib.cm


def compute_spectrum_db(samples, nfft=2048, window=None):
    """Estimate the power spectrum (in dB) of a block of complex IQ samples.

    Splits the block into non-overlapping nfft-length segments, FFTs each,
    and averages the power across segments (Welch's method) -- this is what
    smooths out the noisy, single-snapshot look a single raw FFT would have,
    since a 0.5s block contains hundreds of segments to average over.
    Returns a `nfft`-length array, DC in the middle (frequency increasing
    left to right), in dB.
    """
    if window is None:
        window = np.hanning(nfft)

    n_segments = max(1, len(samples) // nfft)
    power_acc = np.zeros(nfft)
    for i in range(n_segments):
        segment = samples[i * nfft:(i + 1) * nfft]
        spectrum = np.fft.fftshift(np.fft.fft(segment * window))
        power_acc += np.abs(spectrum) ** 2
    power_acc /= n_segments

    return (10 * np.log10(power_acc + 1e-12)).astype(np.float32)


def make_colormap_lut(name='viridis', n=256):
    """Build an (n, 3) uint8 lookup table from a matplotlib colormap, used
    to turn normalized dB values into RGB pixels without needing matplotlib
    to render anything itself."""
    cmap = matplotlib.colormaps[name]
    lut = (np.array([cmap(i / (n - 1))[:3] for i in range(n)]) * 255)
    return lut.astype(np.uint8)


class Waterfall:
    """A scrolling 2D buffer of spectrum rows (oldest at the top, newest at
    the bottom), convertible to an RGB image via a colormap LUT."""

    def __init__(self, nfft, num_rows, min_db=-20, max_db=60):
        self.nfft = nfft
        self.num_rows = num_rows
        self.min_db = min_db
        self.max_db = max_db
        self.rows = np.full((num_rows, nfft), min_db, dtype=np.float32)

    def push(self, spectrum_db):
        self.rows = np.roll(self.rows, -1, axis=0)
        self.rows[-1] = spectrum_db

    def to_rgb_image(self, lut):
        norm = (self.rows - self.min_db) / (self.max_db - self.min_db)
        norm = np.clip(norm, 0.0, 1.0)
        idx = (norm * (len(lut) - 1)).astype(np.int32)
        return lut[idx]
