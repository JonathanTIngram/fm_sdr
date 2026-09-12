"""Listen to a wideband FM radio station with the RTL-SDR.

Usage:
    python3 fm_listen.py 100.7    # tune to 100.7 MHz and play audio
"""
import sys
import queue
import threading

import numpy as np
from scipy.signal import decimate
import sounddevice as sd
from rtlsdr import RtlSdr

DEFAULT_FREQ_MHZ = 100.7
SDR_SAMPLE_RATE = 1_024_000   # Hz, wide enough for a 200 kHz FM channel
AUDIO_SAMPLE_RATE = 48_000    # Hz, standard audio playback rate
DECIM = SDR_SAMPLE_RATE // AUDIO_SAMPLE_RATE  # ~21
SECONDS_PER_BLOCK = 0.5
SAMPLES_PER_BLOCK = int(SDR_SAMPLE_RATE * SECONDS_PER_BLOCK)
QUEUE_MAXSIZE = 6  # ~3 seconds of buffered audio blocks
OUTPUT_LATENCY_S = 0.3  # PortAudio output buffer target; see main() for why

MAX_DEVIATION_HZ = 75_000  # standard FM broadcast max frequency deviation
FULL_SCALE_RAD_PER_SAMPLE = 2 * np.pi * MAX_DEVIATION_HZ / SDR_SAMPLE_RATE


def wideband_fm_demod(samples):
    """Polar-discriminator FM demodulation at the *input* sample rate (no
    decimation). This still contains everything above the audio band --
    the 19kHz stereo pilot and 57kHz RDS subcarrier included -- which
    fm_demodulate()'s decimation filters out. rds.py needs this exact
    signal to recover RDS data."""
    return np.angle(samples[1:] * np.conj(samples[:-1]))


def fm_demodulate(samples, decim):
    """Convert a block of complex IQ samples into normalized mono audio.

    Uses wideband_fm_demod (instantaneous frequency = the phase derivative
    of the IQ signal), then decimates down to audio rate and scales by a
    *fixed* conversion factor derived from the standard 75kHz max FM
    broadcast deviation -- not each block's own peak. Real audio's loudness
    varies moment to moment; rescaling every 0.5s block independently to
    the same fixed peak would apply wildly different gain to a quiet block
    versus a loud one, producing an audible gain jump at every block
    boundary. Actual volume is applied later by AudioPlaybackBuffer.
    """
    demod = wideband_fm_demod(samples)
    audio = decimate(demod, decim, ftype='fir')
    audio = audio - np.mean(audio)
    audio = audio / FULL_SCALE_RAD_PER_SAMPLE
    audio = np.clip(audio, -1.0, 1.0)  # guard against noise/over-deviation spikes
    return audio.astype(np.float32)


class AudioPlaybackBuffer:
    """Feeds a callback-driven sounddevice OutputStream from a queue of
    audio blocks.

    PortAudio's callback frame size rarely matches our block size, so this
    keeps a leftover slice between calls. Runs entirely on PortAudio's own
    thread and never blocks: reading/demodulating (which take real time to
    acquire hardware samples) happen on a separate producer thread, so the
    audio device is never starved waiting on the SDR, and slow blocks pad
    with silence instead of stalling the whole loop.
    """

    def __init__(self, audio_queue, volume=0.8):
        self.queue = audio_queue
        self.volume = volume  # 0.0-1.0; read/written from different threads,
        # but plain float attribute swaps are atomic under the GIL so no lock is needed
        self._leftover = np.zeros(0, dtype=np.float32)

    def callback(self, outdata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        filled = 0
        while filled < frames:
            if len(self._leftover) == 0:
                try:
                    self._leftover = self.queue.get_nowait()
                except queue.Empty:
                    outdata[filled:, 0] = 0.0  # underrun: pad with silence
                    return
            take = min(frames - filled, len(self._leftover))
            outdata[filled:filled + take, 0] = self._leftover[:take] * self.volume
            self._leftover = self._leftover[take:]
            filled += take


def main():
    freq_mhz = float(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_FREQ_MHZ

    sdr = RtlSdr()
    sdr.sample_rate = SDR_SAMPLE_RATE
    sdr.center_freq = freq_mhz * 1e6
    sdr.gain = 'auto'

    print(f"Tuned to {freq_mhz} MHz. Ctrl+C to stop.")

    audio_queue = queue.Queue(maxsize=QUEUE_MAXSIZE)
    player = AudioPlaybackBuffer(audio_queue, volume=0.8)
    # OUTPUT_LATENCY_S: gives PortAudio's own buffer real slack. Reading and
    # demodulating hold Python's GIL for tens of ms at a time (worse with
    # RDS's bit-sync search running too), which can briefly delay this
    # callback being scheduled even though data is ready; the device's
    # 'high' preset is often only ~35ms here, not enough margin against
    # that. A live-radio use case can easily afford the extra latency this
    # buys in exchange for not audibly glitching on every such delay.
    stream = sd.OutputStream(
        samplerate=AUDIO_SAMPLE_RATE, channels=1, dtype='float32',
        callback=player.callback, latency=OUTPUT_LATENCY_S,
    )

    running = threading.Event()
    running.set()

    def reader_loop():
        while running.is_set():
            samples = sdr.read_samples(SAMPLES_PER_BLOCK)
            audio = fm_demodulate(samples, DECIM)
            try:
                audio_queue.put(audio, timeout=1)
            except queue.Full:
                pass  # consumer stalled; drop this block rather than build up latency

    reader_thread = threading.Thread(target=reader_loop, daemon=True)

    try:
        stream.start()
        reader_thread.start()
        reader_thread.join()
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        running.clear()
        reader_thread.join(timeout=2)
        stream.stop()
        stream.close()
        sdr.close()


if __name__ == "__main__":
    main()
