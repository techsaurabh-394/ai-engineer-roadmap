"""Audio basics for speech models, stdlib only.

Synthesise sine waves, cut the signal into frames, find the dominant frequency
with a pure-Python DFT, count frames for a sample rate and hop, and score a
transcript with word error rate (Levenshtein distance on words).

Run:  python mm_audio.py
"""
import math


def sine_wave(freq_hz, sample_rate, duration_s, amplitude=1.0):
    n = int(round(sample_rate * duration_s))
    return [amplitude * math.sin(2 * math.pi * freq_hz * i / sample_rate) for i in range(n)]


def mix(*signals):
    n = min(len(s) for s in signals)
    return [sum(s[i] for s in signals) for i in range(n)]


def num_frames(num_samples, frame_len, hop):
    """Full frames only (no padding): 1 + (N - frame_len) // hop."""
    if num_samples < frame_len:
        return 0
    return 1 + (num_samples - frame_len) // hop


def frame_signal(signal, frame_len, hop):
    return [signal[s:s + frame_len] for s in range(0, len(signal) - frame_len + 1, hop)]


def hann(n):
    return [0.5 - 0.5 * math.cos(2 * math.pi * i / n) for i in range(n)]


def dft_magnitude(frame):
    """Magnitude of the first N/2 + 1 DFT bins, written out the slow, clear way."""
    n = len(frame)
    mags = []
    for k in range(n // 2 + 1):
        re = sum(frame[t] * math.cos(2 * math.pi * k * t / n) for t in range(n))
        im = -sum(frame[t] * math.sin(2 * math.pi * k * t / n) for t in range(n))
        mags.append(math.hypot(re, im))
    return mags


def bin_frequency(k, sample_rate, frame_len):
    """Bin k sits at k * sample_rate / frame_len Hz."""
    return k * sample_rate / frame_len


def dominant_frequency(frame, sample_rate):
    windowed = [x * w for x, w in zip(frame, hann(len(frame)))]
    mags = dft_magnitude(windowed)
    k = max(range(len(mags)), key=lambda i: mags[i])
    return k, bin_frequency(k, sample_rate, len(frame))


def mel(hz):
    """Common mel-scale formula: 2595 * log10(1 + hz / 700)."""
    return 2595 * math.log10(1 + hz / 700)


def word_error_rate(reference, hypothesis):
    """(substitutions + deletions + insertions) / words in the reference."""
    ref, hyp = reference.lower().split(), hypothesis.lower().split()
    if not ref:
        raise ValueError("reference must contain at least one word")
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cost = 0 if r == h else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1] / len(ref)


def main():
    sr = 800
    print(f"sample rate {sr} Hz; 1 second = {sr} numbers; Nyquist limit = {sr // 2} Hz")
    tone = sine_wave(100, sr, 1.0)
    print("first 8 samples of a 100 Hz tone:", [round(x, 3) for x in tone[:8]])

    frame_len, hop = 200, 100
    print(f"frames: {len(frame_signal(tone, frame_len, hop))} "
          f"(formula {num_frames(len(tone), frame_len, hop)}); "
          f"frame = {1000 * frame_len / sr:.0f} ms, hop = {1000 * hop / sr:.0f} ms")

    for f in (50, 100, 200):
        k, hz = dominant_frequency(sine_wave(f, sr, 0.25)[:frame_len], sr)
        print(f"tone {f} Hz -> dominant bin {k} -> {hz:.0f} Hz")

    two = mix(sine_wave(100, sr, 0.25), sine_wave(300, sr, 0.25, 0.5))
    mags = dft_magnitude([x * w for x, w in zip(two[:frame_len], hann(frame_len))])
    top = sorted(range(len(mags)), key=lambda i: -mags[i])[:2]
    print("two tones 100 Hz + 300 Hz -> top bins at", sorted(bin_frequency(k, sr, frame_len) for k in top), "Hz")

    print("\nSpeech-style framing at 16000 Hz, 25 ms window (400), 10 ms hop (160):")
    print("30 s of audio ->", num_frames(16000 * 30, 400, 160), "frames")
    print("1 s of audio  ->", num_frames(16000, 400, 160), "frames")
    print("mel(1000 Hz) =", round(mel(1000), 1), "; mel(2000 Hz) =", round(mel(2000), 1))

    ref = "the quick brown fox jumps"
    hyp = "the quick brown box jumps over"
    print(f"\nWER('{ref}', '{hyp}') = {word_error_rate(ref, hyp):.2f}")
    print("WER identical =", word_error_rate(ref, ref))


if __name__ == "__main__":
    main()
