"""Q4(b): count-min sketch of the word stream, and its relative error."""

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

DELTA = math.exp(-5)
EPSILON = math.e * 1e-4
P = 123457
BLOCK = 1 << 26


def read_ints(filename, columns=1):
    """Read a whitespace-separated integer file in blocks, yielding one array per block."""
    with open(filename, "rb") as f:
        rest = b""
        while True:
            block = f.read(BLOCK)
            if not block:
                break
            block = rest + block
            cut = block.rfind(b"\n") + 1
            block, rest = block[:cut], block[cut:]
            yield np.array(block.split(), dtype=np.int64).reshape(-1, columns)
        if rest.strip():
            yield np.array(rest.split(), dtype=np.int64).reshape(-1, columns)


def sketch(stream_file, params, n_buckets):
    counts = np.zeros((len(params), n_buckets), dtype=np.int64)
    total = 0
    for block in read_ints(stream_file):
        words = block[:, 0]
        total += len(words)
        y = words % P
        for j, (a, b) in enumerate(params):
            counts[j] += np.bincount((a * y + b) % P % n_buckets, minlength=n_buckets)
    return counts, total


def estimate(words, counts, params, n_buckets):
    y = words % P
    return np.min([counts[j][(a * y + b) % P % n_buckets]
                   for j, (a, b) in enumerate(params)], axis=0)


def plot_errors(frequencies, errors, filename):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure()
    plt.loglog(frequencies, errors, ".", markersize=1)
    plt.axhline(1.0, color="red", linewidth=1)
    plt.xlabel("exact word frequency F[i] / t")
    plt.ylabel("relative error E_r[i]")
    plt.title("Count-min sketch relative error (delta=e^-5, epsilon=e*10^-4)")
    plt.grid(True, which="both")
    plt.savefig(os.path.join(HERE, filename), bbox_inches="tight")
    plt.close()


def main():
    tiny = len(sys.argv) > 1 and sys.argv[1] == "tiny"
    suffix = "_tiny" if tiny else ""
    n_buckets = math.ceil(math.e / EPSILON)
    params = np.loadtxt(os.path.join(DATA, "hash_params.txt"), dtype=np.int64)
    print("%d hash functions, %d buckets" % (len(params), n_buckets))

    counts, total = sketch(os.path.join(DATA, "words_stream%s.txt" % suffix),
                           params, n_buckets)
    print("stream length t = %d" % total)

    exact = np.concatenate(list(read_ints(os.path.join(DATA, "counts%s.txt" % suffix), 2)))
    words, F = exact[:, 0], exact[:, 1]
    approx = estimate(words, counts, params, n_buckets)

    errors = (approx - F) / F
    print("%d distinct words, median error %.6f, max error %.2f"
          % (len(words), np.median(errors), errors.max()))
    print("words with error below 1: %.2f%%" % (100.0 * np.mean(errors < 1.0)))

    plot_errors(F / total, errors, "relative-error%s.png" % suffix)


if __name__ == "__main__":
    main()
