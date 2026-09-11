"""Q4(d): user-user and item-item collaborative filtering on the TV show data.

Using the matrices derived in part (c):

    user-user:  Gamma = P^(-1/2) R R^T P^(-1/2) R
    item-item:  Gamma = R Q^(-1/2) R^T R Q^(-1/2)

where P and Q are the diagonal user-degree and item-degree matrices.  Because
they are diagonal, the products are computed by row/column scaling rather than
by building the full diagonal matrices.
"""

import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RATINGS = os.path.join(HERE, "data", "user-shows.txt")
SHOWS = os.path.join(HERE, "data", "shows.txt")

ALEX = 499          # the 500th user, 0-indexed
NUM_HIDDEN = 100    # the first 100 shows whose entries were erased
TOP_K = 5


def load_shows(filename):
    with open(filename) as f:
        return [line.strip().strip('"') for line in f if line.strip()]


def inverse_sqrt_degrees(degrees):
    """1/sqrt(d) for nonzero degrees, 0 elsewhere (the diagonal of P or Q)."""
    out = np.zeros_like(degrees, dtype=float)
    nonzero = degrees > 0
    out[nonzero] = 1.0 / np.sqrt(degrees[nonzero])
    return out


def user_user_gamma(R):
    p = inverse_sqrt_degrees(R.sum(axis=1))     # diagonal of P^(-1/2)
    # P^(-1/2) R R^T P^(-1/2) R, scaling rows/columns instead of matrix products.
    S_U = (p[:, None] * R) @ (p[:, None] * R).T
    return S_U @ R


def item_item_gamma(R):
    q = inverse_sqrt_degrees(R.sum(axis=0))     # diagonal of Q^(-1/2)
    S_I = (R * q[None, :]).T @ (R * q[None, :])
    return R @ S_I


def top_recommendations(gamma, shows, k=TOP_K):
    """Highest-scoring of the first 100 shows for Alex, ties by smaller index."""
    scores = gamma[ALEX, :NUM_HIDDEN]
    # argsort is stable, so negating the scores breaks ties by smaller index.
    order = np.argsort(-scores, kind="stable")[:k]
    return [(int(i), shows[i], float(scores[i])) for i in order]


def show(title, recommendations):
    print(title)
    for index, name, score in recommendations:
        print("  %-40s score = %.4f  (show %d)" % (name, score, index + 1))
    print()


def main():
    R = np.loadtxt(RATINGS)
    shows = load_shows(SHOWS)
    print("R: %d users x %d shows" % R.shape)

    show("User-user collaborative filtering, top %d:" % TOP_K,
         top_recommendations(user_user_gamma(R), shows))
    show("Item-item collaborative filtering, top %d:" % TOP_K,
         top_recommendations(item_item_gamma(R), shows))


if __name__ == "__main__":
    main()

