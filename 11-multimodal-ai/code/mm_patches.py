"""How a vision transformer turns an image into tokens, stdlib only.

Cut an image (nested lists) into non-overlapping square patches, flatten each
patch into one vector, count tokens for different resolutions, and build the
2D position grid.

Run:  python mm_patches.py
"""
import random


def make_image(height, width, channels=1, seed=0):
    """A small synthetic image as nested lists: image[row][col] is a number
    (grayscale) or a list of `channels` numbers (RGB)."""
    rng = random.Random(seed)
    if channels == 1:
        return [[rng.randint(0, 255) for _ in range(width)] for _ in range(height)]
    return [[[rng.randint(0, 255) for _ in range(channels)] for _ in range(width)]
            for _ in range(height)]


def _pixel_values(px):
    return list(px) if isinstance(px, (list, tuple)) else [px]


def split_into_patches(image, patch):
    """Return a list of patches in row-major order. Each patch is a list of rows."""
    h, w = len(image), len(image[0])
    if h % patch or w % patch:
        raise ValueError(f"image {h}x{w} is not divisible by patch size {patch}")
    patches = []
    for top in range(0, h, patch):
        for left in range(0, w, patch):
            patches.append([row[left:left + patch] for row in image[top:top + patch]])
    return patches


def flatten_patch(patch):
    """One patch -> one flat list of numbers (row by row, channels together)."""
    out = []
    for row in patch:
        for px in row:
            out.extend(_pixel_values(px))
    return out


def patch_vector_length(patch_size, channels):
    return patch_size * patch_size * channels


def count_patches(resolution, patch):
    """Patches for a square image: (resolution / patch) squared."""
    if resolution % patch:
        raise ValueError("resolution must be divisible by patch size")
    side = resolution // patch
    return side * side


def count_tokens(resolution, patch, cls_token=True):
    return count_patches(resolution, patch) + (1 if cls_token else 0)


def position_grid(rows, cols):
    """(row, col) index of every patch, in the same order as split_into_patches."""
    return [(r, c) for r in range(rows) for c in range(cols)]


def project(vector, weights):
    """Linear patch embedding: vector (length P) times weights (P x D) -> length D."""
    d = len(weights[0])
    return [sum(vector[i] * weights[i][j] for i in range(len(vector))) for j in range(d)]


def attention_pairs(tokens):
    """Self-attention compares every token with every token: tokens squared."""
    return tokens * tokens


def video_tokens(seconds, fps, resolution, patch=16, cls_token=True):
    """Tokens if every sampled frame is encoded as its own image."""
    frames = int(seconds * fps)
    return frames, frames * count_tokens(resolution, patch, cls_token)


def main():
    image = make_image(8, 8, channels=1, seed=7)
    patches = split_into_patches(image, 4)
    print(f"8x8 grayscale image, patch 4 -> {len(patches)} patches")
    first = flatten_patch(patches[0])
    print("patch 0 flattened:", first, f"(length {len(first)})")
    print("position grid:", position_grid(2, 2))

    rgb = make_image(8, 8, channels=3, seed=7)
    rgb_patches = split_into_patches(rgb, 4)
    print(f"8x8 RGB image, patch 4 -> {len(rgb_patches)} patches, "
          f"vector length {len(flatten_patch(rgb_patches[0]))}")

    print("\nResolution vs tokens (patch 16, plus 1 CLS token):")
    print("res  patches  tokens  attention_pairs")
    for res in (224, 336, 448, 672, 896):
        t = count_tokens(res, 16)
        print(f"{res:<4} {count_patches(res, 16):<8} {t:<7} {attention_pairs(t)}")

    print("\nPatch size 14 vs 16 at 224:", count_patches(224, 14), "vs", count_patches(224, 16))
    print("Doubling 224 -> 448 multiplies patches by",
          count_patches(448, 16) / count_patches(224, 16))
    print("Patch vector length for 16x16 RGB:", patch_vector_length(16, 3))
    print("Patch embedding weights for 16x16 RGB -> 768:", patch_vector_length(16, 3) * 768)

    print("\nVideo token budget (224 px, patch 16, 197 tokens per frame):")
    for seconds, fps in ((10, 1), (60, 1), (60, 2), (600, 1)):
        frames, toks = video_tokens(seconds, fps, 224)
        print(f"{seconds:>4} s at {fps} fps -> {frames:>4} frames -> {toks:>6} tokens")
    print("8 uniformly spaced frames of a 60 s clip:", [round(i * 60 / 8, 1) for i in range(8)])


if __name__ == "__main__":
    main()
