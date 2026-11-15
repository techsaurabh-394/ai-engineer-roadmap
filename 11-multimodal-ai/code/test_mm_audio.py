import math

import pytest

import mm_audio as a


def test_sine_length_and_values():
    s = a.sine_wave(100, 800, 1.0)
    assert len(s) == 800
    assert s[0] == 0.0
    assert s[2] == pytest.approx(math.sin(2 * math.pi * 100 * 2 / 800))
    assert max(s) <= 1.0 and min(s) >= -1.0


def test_frame_count_formula_matches_framing():
    sig = list(range(1000))
    for frame_len, hop in [(200, 100), (400, 160), (50, 50)]:
        assert len(a.frame_signal(sig, frame_len, hop)) == a.num_frames(1000, frame_len, hop)


def test_speech_frame_arithmetic():
    assert a.num_frames(16000, 400, 160) == 98
    assert a.num_frames(16000 * 30, 400, 160) == 2998
    assert a.num_frames(100, 400, 160) == 0


def test_bin_frequency():
    assert a.bin_frequency(10, 16000, 400) == 400.0


@pytest.mark.parametrize("f", [50, 100, 200, 300])
def test_dominant_frequency_matches_tone(f):
    sr, n = 800, 200
    frame = a.sine_wave(f, sr, 0.25)[:n]
    k, hz = a.dominant_frequency(frame, sr)
    assert hz == pytest.approx(f, abs=sr / n)  # within one bin (4 Hz)


def test_two_tones_found():
    sr, n = 800, 200
    sig = a.mix(a.sine_wave(100, sr, 0.25), a.sine_wave(300, sr, 0.25, 0.5))
    mags = a.dft_magnitude([x * w for x, w in zip(sig[:n], a.hann(n))])
    top = sorted(sorted(range(len(mags)), key=lambda i: -mags[i])[:2])
    assert [a.bin_frequency(k, sr, n) for k in top] == [100.0, 300.0]


def test_mel_is_increasing_and_compressive():
    assert a.mel(0) == 0
    assert a.mel(2000) - a.mel(1000) < a.mel(1000) - a.mel(0)


def test_wer_basic():
    assert a.word_error_rate("a b c", "a b c") == 0.0
    assert a.word_error_rate("a b c d", "a x c d") == 0.25  # 1 substitution
    assert a.word_error_rate("a b c d", "a b c") == 0.25    # 1 deletion
    assert a.word_error_rate("a b", "a b c d") == 1.0       # 2 insertions
    assert a.word_error_rate("a b", "") == 1.0


def test_wer_can_exceed_one_and_ignores_case():
    assert a.word_error_rate("a", "x y z") == 3.0
    assert a.word_error_rate("Hello World", "hello world") == 0.0


def test_wer_empty_reference():
    with pytest.raises(ValueError):
        a.word_error_rate("", "x")
