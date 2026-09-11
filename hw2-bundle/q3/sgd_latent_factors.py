"""Q3(b): latent factor recommender trained with Stochastic Gradient Descent.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RATINGS = os.path.join(HERE, "data", "ratings.train.txt")

K = 20
LAMBDA = 0.1
ITERATIONS = 40
ETA = 0.02

NUM_MOVIES = 1682
NUM_USERS = 943


def stream_ratings(filename):
    """Yield (movie index, user index, rating) one entry at a time."""
    with open(filename) as f:
        for line in f:
            if not line.strip():
                continue
            user, movie, rating = line.split()
            yield int(movie) - 1, int(user) - 1, float(rating)


def objective(Q, P, filename, lam=LAMBDA):
    """E = sum of squared errors over known ratings + L2 regularization."""
    error = 0.0
    for i, u, rating in stream_ratings(filename):
        error += (rating - Q[i] @ P[u]) ** 2
    return error + lam * (np.sum(P ** 2) + np.sum(Q ** 2))


def train(filename, k=K, lam=LAMBDA, eta=ETA, iterations=ITERATIONS, seed=0):
    rng = np.random.default_rng(seed)
    # q_i . p_u then falls in [0, 5] as required.
    high = np.sqrt(5.0 / k)
    Q = rng.uniform(0.0, high, size=(NUM_MOVIES, k))
    P = rng.uniform(0.0, high, size=(NUM_USERS, k))

    errors = []
    for iteration in range(1, iterations + 1):
        for i, u, rating in stream_ratings(filename):
            q_i, p_u = Q[i].copy(), P[u]
            eps = 2.0 * (rating - q_i @ p_u)
            Q[i] = q_i + eta * (eps * p_u - 2.0 * lam * q_i)
            P[u] = p_u + eta * (eps * q_i - 2.0 * lam * p_u)

        errors.append(objective(Q, P, filename, lam))
        print("iteration %2d: E = %.2f" % (iteration, errors[-1]))

    return Q, P, errors


def plot_errors(errors, eta, filename):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(range(1, len(errors) + 1), errors, marker="o")
    plt.xlabel("iteration")
    plt.ylabel("E")
    plt.title("Objective E vs. iteration (k=%d, lambda=%s, eta=%s)"
              % (K, LAMBDA, eta))
    plt.grid(True)
    plt.savefig(os.path.join(HERE, filename), bbox_inches="tight")
    plt.close()


def main():
    eta = float(sys.argv[1]) if len(sys.argv) > 1 else ETA
    print("eta = %s" % eta)
    _, _, errors = train(RATINGS, eta=eta)
    print("\nE after %d iterations: %.2f" % (ITERATIONS, errors[-1]))
    plot_errors(errors, eta, "error-vs-iteration.png")


if __name__ == "__main__":
    main()
