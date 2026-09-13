"""Unit tests for the RDS decoder in rds.py.

Two layers get tested:
  - Protocol logic (block check-code sync, group field decoding) via
    synthetic bitstreams built with encode_block() -- exact, deterministic.
  - The bit-recovery DSP (differential_bits) via a synthetic biphase
    baseband signal built independently of encode_block.
mix_rds_to_baseband() (the actual 57kHz RF mixing) only gets a light
sanity check here; the real proof is the live smoke test against hardware.
"""
import numpy as np
import pytest

from rds import (
    encode_block,
    block_syndrome,
    bits_to_int,
    differential_bits,
    mix_rds_to_baseband,
    BlockSynchronizer,
    GroupParser,
    RDS_SUBCARRIER_HZ,
    OFFSET_WORDS,
    SAMPLES_PER_BIT,
)


def test_encode_block_syndrome_matches_its_offset_word():
    for offset_key in OFFSET_WORDS:
        block = encode_block(0b1010110011000101, offset_key)
        assert len(block) == 26
        assert block_syndrome(block) == OFFSET_WORDS[offset_key]


def bits_for_group(pi, b_value, c_value, d_value):
    return (
        encode_block(pi, 'A')
        + encode_block(b_value, 'B')
        + encode_block(c_value, 'C')
        + encode_block(d_value, 'D')
    )


def test_block_synchronizer_locks_and_decodes_clean_stream():
    bits = bits_for_group(0x1234, 0x0000, 0xAAAA, 0x4D59)  # 'M','Y' packed
    bits += bits_for_group(0x1234, 0x0001, 0xAAAA, 0x5354)  # 'S','T'

    sync = BlockSynchronizer()
    results = sync.feed(bits)

    assert sync.locked
    assert sync.blocks_ok == 8
    assert sync.blocks_error == 0
    assert [r[0] for r in results] == ['A', 'B', 'C', 'D', 'A', 'B', 'C', 'D']
    assert results[0] == ('A', 0x1234)
    assert results[3] == ('D', 0x4D59)


def test_block_synchronizer_finds_sync_after_garbage_prefix():
    rng = np.random.default_rng(42)
    garbage = list(rng.integers(0, 2, size=137))
    bits = garbage + bits_for_group(0x1234, 0x0000, 0xAAAA, 0x4D59)

    sync = BlockSynchronizer()
    results = sync.feed(bits)

    assert sync.locked
    assert ('A', 0x1234) in results


def test_group_parser_decodes_ps_name():
    pi = 0x1234
    segments = ['MY', 'ST', 'AT', 'IO']
    bits = []
    for i, seg in enumerate(segments):
        b_value = (0 << 12) | (0 << 11) | i  # group 0, version A, segment i
        d_value = (ord(seg[0]) << 8) | ord(seg[1])
        bits += bits_for_group(pi, b_value, 0xAAAA, d_value)

    sync = BlockSynchronizer()
    parser = GroupParser()
    for block_type, data in sync.feed(bits):
        parser.feed_block(block_type, data)

    assert parser.station_name == 'MYSTATIO'
    assert parser.pi_code == pi


def test_group_parser_decodes_radiotext_version_a():
    pi = 0xABCD
    message = "TEST ARTIST - TEST SONG TITLE".ljust(64)[:64]
    bits = []
    for segment in range(16):
        chunk = message[segment * 4:segment * 4 + 4]
        b_value = (2 << 12) | (0 << 11) | segment  # group 2, version A
        c_value = (ord(chunk[0]) << 8) | ord(chunk[1])
        d_value = (ord(chunk[2]) << 8) | ord(chunk[3])
        bits += bits_for_group(pi, b_value, c_value, d_value)

    sync = BlockSynchronizer()
    parser = GroupParser()
    for block_type, data in sync.feed(bits):
        parser.feed_block(block_type, data)

    assert parser.radiotext == message.rstrip()


def test_group_parser_decodes_radiotext_version_b():
    pi = 0xABCD
    message = "SHORT RT B".ljust(32)[:32]
    bits = []
    for segment in range(16):
        chunk = message[segment * 2:segment * 2 + 2]
        b_value = (2 << 12) | (1 << 11) | segment  # group 2, version B
        d_value = (ord(chunk[0]) << 8) | ord(chunk[1])
        bits += bits_for_group(pi, b_value, pi, d_value)  # block C repeats PI in version B

    sync = BlockSynchronizer()
    parser = GroupParser()
    for block_type, data in sync.feed(bits):
        parser.feed_block(block_type, data)

    assert parser.radiotext == message.rstrip()


def test_group_parser_clears_radiotext_on_text_ab_toggle():
    pi = 0xABCD
    old_message = "OLD ARTIST - LONG SONG TITLE HERE".ljust(64)[:64]
    bits = []
    for segment in range(16):
        chunk = old_message[segment * 4:segment * 4 + 4]
        b_value = (2 << 12) | (0 << 11) | (0 << 4) | segment  # text A/B flag = 0
        c_value = (ord(chunk[0]) << 8) | ord(chunk[1])
        d_value = (ord(chunk[2]) << 8) | ord(chunk[3])
        bits += bits_for_group(pi, b_value, c_value, d_value)

    sync = BlockSynchronizer()
    parser = GroupParser()
    for block_type, data in sync.feed(bits):
        parser.feed_block(block_type, data)
    assert parser.radiotext == old_message.rstrip()

    # New, shorter message with the text A/B flag toggled -- only the first
    # segment arrives so far, but the flag flip must wipe the rest of the
    # old message's leftover tail rather than splicing onto it.
    new_first_chunk = "NEW "
    b_value = (2 << 12) | (0 << 11) | (1 << 4) | 0  # text A/B flag flipped to 1
    c_value = (ord(new_first_chunk[0]) << 8) | ord(new_first_chunk[1])
    d_value = (ord(new_first_chunk[2]) << 8) | ord(new_first_chunk[3])
    bits = bits_for_group(pi, b_value, c_value, d_value)
    for block_type, data in sync.feed(bits):
        parser.feed_block(block_type, data)

    assert parser.radiotext == "NEW"


def test_differential_bits_recovers_known_pattern():
    target_bits = [1, 0, 1, 1, 0, 0, 1]
    samples_per_bit = SAMPLES_PER_BIT
    half = samples_per_bit // 2

    metrics = [1 + 0j]
    for b in target_bits:
        metrics.append(-metrics[-1] if b else metrics[-1])

    chunks = []
    for val in metrics:
        first_half = np.full(half, -val / 2, dtype=complex)
        second_half = np.full(half, val / 2, dtype=complex)
        chunks.append(np.concatenate([first_half, second_half]))
    baseband = np.concatenate(chunks)

    decoded = differential_bits(baseband, phase_offset=0, samples_per_bit=samples_per_bit)
    assert list(decoded) == target_bits


def test_mix_rds_to_baseband_isolates_offset_tone():
    fs_in = 1_024_000
    offset_hz = 300.0
    n = int(fs_in * 0.2)
    t = np.arange(n) / fs_in
    tone = np.cos(2 * np.pi * (RDS_SUBCARRIER_HZ + offset_hz) * t)

    baseband = mix_rds_to_baseband(tone, fs_in)
    assert np.all(np.isfinite(baseband))

    fs_out = len(baseband) / 0.2
    spectrum = np.abs(np.fft.fftshift(np.fft.fft(baseband)))
    freqs = np.fft.fftshift(np.fft.fftfreq(len(baseband), d=1 / fs_out))
    peak_freq = freqs[np.argmax(spectrum)]

    assert peak_freq == pytest.approx(offset_hz, abs=50)
