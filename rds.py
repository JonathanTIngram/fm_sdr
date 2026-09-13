"""RDS (Radio Data System) decoder.

Extracts the 57kHz RDS subcarrier from the wideband FM-demodulated signal
(the signal *before* fm_listen.fm_demodulate()'s audio low-pass/decimation
step -- that step filters out everything above ~24kHz, but RDS lives at
57kHz), recovers its data bits, and parses them into groups to build up the
station name (PS) and RadioText (RT -- typically artist/song, as free text;
classic FM RDS carries no images).

Pipeline, all pure Python/NumPy so it's unit-testable without hardware:

  1. mix_rds_to_baseband() shifts the 57kHz subcarrier down to 0Hz and
     resamples straight to 8 samples/bit.
  2. differential_bits() turns those baseband samples into 0/1 bits. RDS's
     biphase coding plus differential precoding means comparing the first
     half vs. second half of each bit period, then comparing *that* against
     the previous bit period, recovers the original bits without ever
     needing to track absolute carrier phase -- only phase *within* one
     ~0.84ms bit period has to be roughly stable, which a few Hz of leftover
     frequency error (from the tuner's own clock) satisfies easily. No PLL.
  3. BlockSynchronizer finds 26-bit block boundaries: each block (16 data +
     10 check bits) has a fixed "offset word" XORed into its check bits
     before transmission based on its type (A/B/C/C'/D), so computing the
     syndrome of a candidate 26-bit window and matching it against the 5
     known offset words both finds alignment and identifies block type.
  4. GroupParser assembles validated A+B+C/C'+D blocks into groups and
     decodes the two group types needed here: type 0 (station name) and
     type 2 (RadioText).
"""
import numpy as np
from fractions import Fraction
from scipy.signal import resample_poly
from collections import deque

RDS_SUBCARRIER_HZ = 57000.0
BIT_RATE = 1187.5  # bits/sec
SAMPLES_PER_BIT = 8
RDS_BASEBAND_RATE = BIT_RATE * SAMPLES_PER_BIT  # 9500 Hz

# GF(2) generator polynomial for the RDS block check code, x^10+x^8+x^7+x^5+x^4+x^3+1,
# as coefficients from x^10 down to x^0.
GENERATOR_BITS = [1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1]

OFFSET_WORDS = {
    'A': 0b0011111100,
    'B': 0b0110011000,
    'C': 0b0101101000,
    "C'": 0b1101010000,
    'D': 0b0110110100,
}
BLOCK_SEQUENCE = ['A', 'B', 'C', 'D']

PTY_NAMES = [
    'None', 'News', 'Current Affairs', 'Information', 'Sport', 'Education', 'Drama',
    'Culture', 'Science', 'Varied', 'Pop Music', 'Rock Music', 'Easy Listening',
    'Light Classical', 'Serious Classical', 'Other Music', 'Weather', 'Finance',
    "Children's Programmes", 'Social Affairs', 'Religion', 'Phone-In', 'Travel',
    'Leisure & Hobby', 'Jazz Music', 'Country Music', 'National Music', 'Oldies Music',
    'Folk Music', 'Documentary', 'Alarm Test', 'Alarm',
]


def mix_rds_to_baseband(wideband_demod, fs_in):
    """Shift the 57kHz RDS subcarrier down to 0Hz and resample to exactly
    SAMPLES_PER_BIT samples per bit. `fs_in` is the sample rate of
    `wideband_demod` (i.e. the SDR sample rate the FM demodulation ran at,
    before any audio decimation)."""
    n = np.arange(len(wideband_demod))
    mixed = wideband_demod * np.exp(-1j * 2 * np.pi * RDS_SUBCARRIER_HZ * n / fs_in)
    ratio = Fraction(int(RDS_BASEBAND_RATE), int(fs_in)).limit_denominator(10000)
    return resample_poly(mixed, ratio.numerator, ratio.denominator)


def differential_bits(baseband, phase_offset, samples_per_bit=SAMPLES_PER_BIT):
    """Convert baseband RDS samples to a 0/1 bit array using a fixed
    within-symbol phase offset (0..samples_per_bit-1: which sample the bit
    period starts on -- unknown a priori, so the caller tries several).

    Each bit period is compared half-vs-half (a matched filter against the
    biphase/Manchester basis shape) to get a complex "bit metric" per bit;
    RDS's differential precoding is then undone by comparing consecutive
    bit metrics' relative phase, which cancels slowly-varying residual
    carrier phase instead of requiring a phase-locked local oscillator.
    """
    trimmed = baseband[phase_offset:]
    n_bits = len(trimmed) // samples_per_bit
    if n_bits < 2:
        return np.zeros(0, dtype=np.uint8)
    trimmed = trimmed[:n_bits * samples_per_bit]
    chunks = trimmed.reshape(n_bits, samples_per_bit)
    half = samples_per_bit // 2
    metric = chunks[:, half:].mean(axis=1) - chunks[:, :half].mean(axis=1)
    prod = metric[1:] * np.conj(metric[:-1])
    return (prod.real < 0).astype(np.uint8)


def poly_remainder(bits):
    """GF(2) remainder of `bits` (an iterable of 0/1, MSB first) divided by
    the RDS generator polynomial."""
    bits = list(bits)
    glen = len(GENERATOR_BITS)
    for i in range(len(bits) - glen + 1):
        if bits[i]:
            for j in range(glen):
                bits[i + j] ^= GENERATOR_BITS[j]
    value = 0
    for b in bits[-(glen - 1):]:
        value = (value << 1) | b
    return value


def block_syndrome(block_bits26):
    return poly_remainder(block_bits26)


def bits_to_int(bits):
    value = 0
    for b in bits:
        value = (value << 1) | int(b)
    return value


def int_to_bits(value, width):
    return [(value >> i) & 1 for i in range(width - 1, -1, -1)]


def encode_block(data_value16, offset_key):
    """Encode a 16-bit data word into a full 26-bit RDS block (list of
    0/1). Used only to build synthetic signals for tests -- real decoding
    never calls this."""
    data_bits = int_to_bits(data_value16, 16)
    check = poly_remainder(data_bits + [0] * 10)
    check_bits = int_to_bits(check, 10)
    offset_bits = int_to_bits(OFFSET_WORDS[offset_key], 10)
    transmitted_check = [c ^ o for c, o in zip(check_bits, offset_bits)]
    return data_bits + transmitted_check


class BlockSynchronizer:
    """Finds 26-bit block boundaries in a continuous bit stream and
    classifies each block as A/B/C/C'/D using the RDS check code, resyncing
    automatically if too many consecutive blocks fail to validate.

    While unsynced, each feed() call searches only the bits given in that
    call (a fresh window) rather than an accumulated buffer, since a caller
    doing a bit-timing-phase search needs each phase's attempt isolated.
    Once locked, leftover bits *are* carried over between calls, since real
    block boundaries won't line up with 0.5s SDR block boundaries.
    """

    def __init__(self, max_consecutive_errors=3):
        self.bits = []
        self.locked = False
        self.expected_index = 0
        self.consecutive_errors = 0
        self.max_consecutive_errors = max_consecutive_errors
        self.blocks_ok = 0
        self.blocks_error = 0

    def feed(self, new_bits):
        if self.locked:
            self.bits.extend(int(b) for b in new_bits)
        else:
            self.bits = [int(b) for b in new_bits]

        results = []
        while len(self.bits) >= 26:
            if not self.locked:
                if len(self.bits) < 52:
                    break  # need a second block to confirm before committing to a lock
                block_type = self._match_any_offset(self.bits[:26])
                if block_type is None:
                    del self.bits[0]
                    continue
                # Require the *next* block to also validate, in the correct
                # sequence position, before committing -- a single matching
                # block isn't enough evidence: with 5 offset words out of
                # 1024 possible syndromes, checked at every bit position,
                # random noise finds one by chance well within a second.
                seq_pos = BLOCK_SEQUENCE.index('C' if block_type == "C'" else block_type)
                next_expected = BLOCK_SEQUENCE[(seq_pos + 1) % 4]
                next_candidates = ['C', "C'"] if next_expected == 'C' else [next_expected]
                next_syndrome = block_syndrome(self.bits[26:52])
                next_matched = next(
                    (c for c in next_candidates if OFFSET_WORDS[c] == next_syndrome), None
                )
                if next_matched is None:
                    del self.bits[0]
                    continue

                results.append(self._accept(block_type, self.bits[:26]))
                results.append(self._accept(next_matched, self.bits[26:52]))
                del self.bits[:52]
                self.locked = True
                seq_pos2 = BLOCK_SEQUENCE.index('C' if next_matched == "C'" else next_matched)
                self.expected_index = (seq_pos2 + 1) % 4
                self.consecutive_errors = 0
            else:
                expected = BLOCK_SEQUENCE[self.expected_index]
                candidates = ['C', "C'"] if expected == 'C' else [expected]
                syndrome = block_syndrome(self.bits[:26])
                matched = next((c for c in candidates if OFFSET_WORDS[c] == syndrome), None)
                if matched is not None:
                    results.append(self._accept(matched, self.bits[:26]))
                    self.consecutive_errors = 0
                else:
                    self.blocks_error += 1
                    self.consecutive_errors += 1
                del self.bits[:26]
                self.expected_index = (self.expected_index + 1) % 4
                if self.consecutive_errors > self.max_consecutive_errors:
                    self.locked = False
                    self.consecutive_errors = 0
        return results

    def _match_any_offset(self, block_bits26):
        syndrome = block_syndrome(block_bits26)
        for key, val in OFFSET_WORDS.items():
            if syndrome == val:
                return key
        return None

    def _accept(self, block_type, block_bits26):
        self.blocks_ok += 1
        return block_type, bits_to_int(block_bits26[:16])


def _clean_text(chars):
    return ''.join(c if c.isprintable() else ' ' for c in chars).rstrip()


class GroupParser:
    """Assembles synced A/B/C/D blocks into groups and decodes group types
    0 (station name / PS) and 2 (RadioText / RT)."""

    def __init__(self, log_maxlen=30):
        self.pi_code = None
        self.pty = None
        self.ps_chars = [' '] * 8
        self.rt_chars = [' '] * 64
        self.rt_ab_flag = None
        self.log = deque(maxlen=log_maxlen)
        self.group_count = 0  # monotonic; log itself evicts old entries
        self._pending = {}

    def feed_block(self, block_type, data_value):
        key = 'C' if block_type == "C'" else block_type
        if block_type == 'A':
            self._pending = {'A': data_value}
            return None
        if 'A' not in self._pending:
            return None  # stray block before we've seen this group's A block
        self._pending[key] = data_value
        if not {'A', 'B', 'C', 'D'} <= self._pending.keys():
            return None

        group = self._decode_group(self._pending)
        self._pending = {}
        self.log.append(group)
        self.group_count += 1
        return group

    def _decode_group(self, blocks):
        a, b, c, d = blocks['A'], blocks['B'], blocks['C'], blocks['D']
        self.pi_code = a
        group_type = (b >> 12) & 0xF
        version_b = (b >> 11) & 0x1
        tp = (b >> 10) & 0x1
        self.pty = (b >> 5) & 0x1F

        entry = {
            'pi': a, 'group_type': group_type,
            'version': 'B' if version_b else 'A', 'tp': tp, 'pty': self.pty,
            'fields': {},
        }

        if group_type == 0:
            ta = (b >> 4) & 0x1
            ms = (b >> 3) & 0x1
            di = (b >> 2) & 0x1
            segment = b & 0x3
            chars = chr((d >> 8) & 0xFF) + chr(d & 0xFF)
            self.ps_chars[segment * 2:segment * 2 + 2] = list(chars)
            entry['fields'] = {
                'name': 'PS', 'segment': segment, 'chars': chars,
                'ta': ta, 'ms': ms, 'di': di,
            }
        elif group_type == 2:
            # Bit 4 is the text A/B flag: stations toggle it whenever a
            # brand-new RadioText message starts, so receivers know to wipe
            # any leftover characters from the previous message instead of
            # splicing the new one into the old buffer -- without this, a
            # shorter new message leaves stale tail characters behind.
            text_ab = (b >> 4) & 0x1
            if self.rt_ab_flag is not None and text_ab != self.rt_ab_flag:
                self.rt_chars = [' '] * 64
            self.rt_ab_flag = text_ab

            segment = b & 0xF
            if version_b == 0:
                chars = chr((c >> 8) & 0xFF) + chr(c & 0xFF) + chr((d >> 8) & 0xFF) + chr(d & 0xFF)
                base = segment * 4
            else:
                chars = chr((d >> 8) & 0xFF) + chr(d & 0xFF)
                base = segment * 2
            end = min(base + len(chars), 64)
            self.rt_chars[base:end] = list(chars[:end - base])
            entry['fields'] = {
                'name': 'RT', 'segment': segment, 'chars': chars, 'text_ab': text_ab,
            }

        return entry

    @property
    def station_name(self):
        return _clean_text(self.ps_chars)

    @property
    def radiotext(self):
        return _clean_text(self.rt_chars)

    @property
    def pty_name(self):
        if self.pty is None or not (0 <= self.pty < len(PTY_NAMES)):
            return None
        return PTY_NAMES[self.pty]


class RdsDecoder:
    """Top-level facade: feed it wideband FM-demodulated blocks, read back
    the decoded station info at any time."""

    def __init__(self):
        self.sync = BlockSynchronizer()
        self.groups = GroupParser()
        self._phase_offset = 0

    def process_block(self, wideband_demod, fs_in):
        baseband = mix_rds_to_baseband(wideband_demod, fs_in)
        new_groups = []

        if not self.sync.locked:
            for phase in range(SAMPLES_PER_BIT):
                bits = differential_bits(baseband, phase)
                results = self.sync.feed(bits)
                if self.sync.locked:
                    self._phase_offset = phase
                    new_groups.extend(self._apply(results))
                    break
        else:
            bits = differential_bits(baseband, self._phase_offset)
            results = self.sync.feed(bits)
            new_groups.extend(self._apply(results))

        return new_groups

    def _apply(self, block_results):
        out = []
        for block_type, data in block_results:
            group = self.groups.feed_block(block_type, data)
            if group is not None:
                out.append(group)
        return out
