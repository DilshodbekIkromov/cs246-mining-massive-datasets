import hashlib
from collections import defaultdict
from itertools import combinations

import numpy as np

PRIME = (1 << 31) - 1
PRIME_U = np.uint64(PRIME)


def shingles(text, k=5, word_level=False):
    if word_level:
        tokens = text.lower().split()
        if len(tokens) < k:
            return {" ".join(tokens)} if tokens else set()
        return {" ".join(tokens[i:i + k]) for i in range(len(tokens) - k + 1)}
    t = " ".join(text.lower().split())
    if len(t) < k:
        return {t} if t else set()
    return {t[i:i + k] for i in range(len(t) - k + 1)}


def hash_shingles(shingle_set):
    if not shingle_set:
        return np.empty(0, dtype=np.uint64)
    digest = hashlib.blake2b
    return np.fromiter(
        (int.from_bytes(digest(s.encode("utf-8"), digest_size=8).digest(), "big") % PRIME
         for s in shingle_set),
        dtype=np.uint64,
        count=len(shingle_set),
    )


def make_hash_params(n, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.integers(1, PRIME, size=n, dtype=np.uint64)
    b = rng.integers(0, PRIME, size=n, dtype=np.uint64)
    return a, b


def minhash(hashed, a, b, chunk=4096):
    if hashed.size == 0:
        return np.full(a.size, PRIME_U, dtype=np.uint64)
    sig = np.full(a.size, PRIME_U, dtype=np.uint64)
    for i in range(0, hashed.size, chunk):
        block = hashed[i:i + chunk, None] * a[None, :] + b[None, :]
        np.minimum(sig, (block % PRIME_U).min(axis=0), out=sig)
    return sig


def choose_bands(n, threshold):
    best = None
    for r in range(1, n + 1):
        b = n // r
        if b < 1:
            break
        err = abs((1.0 / b) ** (1.0 / r) - threshold)
        if best is None or err < best[0]:
            best = (err, b, r)
    return best[1], best[2]


def detection_prob(s, b, r):
    return 1.0 - (1.0 - s ** r) ** b


def candidate_pairs(sigs, b, r, max_bucket=2000):
    pairs = set()
    for band in range(b):
        buckets = defaultdict(list)
        lo, hi = band * r, (band + 1) * r
        for doc_id in range(sigs.shape[0]):
            buckets[sigs[doc_id, lo:hi].tobytes()].append(doc_id)
        for ids in buckets.values():
            if 1 < len(ids) <= max_bucket:
                pairs.update(combinations(ids, 2))
    return pairs


def jaccard(s1, s2):
    union = len(s1 | s2)
    return len(s1 & s2) / union if union else 1.0


def find_similar(docs, threshold=0.8, n=128, k=5, word_level=False, seed=0, exact_check=True):
    sets = [shingles(d, k, word_level) for d in docs]
    a, hb = make_hash_params(n, seed)
    sigs = np.stack([minhash(hash_shingles(s), a, hb) for s in sets])
    b, r = choose_bands(n, threshold)
    out = []
    for i, j in candidate_pairs(sigs, b, r):
        sim = jaccard(sets[i], sets[j]) if exact_check else float((sigs[i] == sigs[j]).mean())
        if sim >= threshold:
            out.append((i, j, sim))
    out.sort(key=lambda x: -x[2])
    return out, (b, r)


if __name__ == "__main__":
    base = "the quick brown fox jumps over the lazy dog near the old river bank at dawn"
    docs = [
        base,
        base.replace("lazy dog", "lazy hound"),
        base + " and then it disappeared into the misty forest",
        "completely unrelated text about distributed systems and consensus protocols",
        "a totally different sentence discussing database indexing strategies",
        base,
    ]
    matches, (b, r) = find_similar(docs, threshold=0.8, n=128, k=5)
    print(f"bands={b} rows={r} est_threshold={(1/b)**(1/r):.3f}")
    for i, j, s in matches:
        print(f"{i} ~ {j}  jaccard={s:.3f}")
    for s in (0.5, 0.7, 0.8, 0.9):
        print(f"P(candidate | J={s}) = {detection_prob(s, b, r):.3f}")