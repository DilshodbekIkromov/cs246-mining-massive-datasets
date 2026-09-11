"""Q1(b): soft margin SVM trained by batch, stochastic and mini batch gradient descent."""

import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FEATURES = os.path.join(HERE, "data", "features.txt")
TARGET = os.path.join(HERE, "data", "target.txt")

C = 100.0
BATCH_SIZE = 20
SEED = 0


def cost(X, y, w, b):
    return 0.5 * w @ w + C * np.maximum(0.0, 1.0 - y * (X @ w + b)).sum()


def gradient(X, y, w, b):
    """Gradient of f restricted to the given samples."""
    violating = y * (X @ w + b) < 1.0
    yv, Xv = y[violating], X[violating]
    return w - C * (yv @ Xv), -C * yv.sum()


def descend(X, y, eta, epsilon, batch_size, averaged):
    """Gradient descent over batches of batch_size; returns costs and elapsed time."""
    rng = np.random.default_rng(SEED)
    n = len(y)
    if batch_size < n:
        order = rng.permutation(n)
        X, y = X[order], y[order]

    w, b = np.zeros(X.shape[1]), 0.0
    costs = [cost(X, y, w, b)]
    delta, l = 0.0, 0
    start = time.time()

    while True:
        rows = slice(l * batch_size, (l + 1) * batch_size)
        grad_w, grad_b = gradient(X[rows], y[rows], w, b)
        w, b = w - eta * grad_w, b - eta * grad_b
        l = (l + 1) % int(np.ceil(n / batch_size))

        costs.append(cost(X, y, w, b))
        percent = abs(costs[-2] - costs[-1]) * 100.0 / costs[-2]
        delta = 0.5 * delta + 0.5 * percent if averaged else percent
        if delta < epsilon:
            break

    return costs, time.time() - start


def plot_costs(results, filename):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure()
    for name, (costs, _) in results.items():
        plt.plot(range(len(costs)), costs, label=name)
    plt.xlabel("iteration k")
    plt.ylabel("f_k(w, b)")
    plt.title("SVM cost vs. iteration (C=%g)" % C)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(HERE, filename), bbox_inches="tight")
    plt.close()


def main():
    X = np.loadtxt(FEATURES, delimiter=",")
    y = np.loadtxt(TARGET)
    print("X: %d x %d\n" % X.shape)

    results = {}
    for name, eta, epsilon, batch_size, averaged in [
            ("batch GD", 0.0000003, 0.25, len(y), False),
            ("stochastic GD", 0.0001, 0.001, 1, True),
            ("mini batch GD", 0.00001, 0.01, BATCH_SIZE, True)]:
        results[name] = descend(X, y, eta, epsilon, batch_size, averaged)
        costs, seconds = results[name]
        print("%-14s %5d iterations, %6.2f s, final cost %.4f"
              % (name, len(costs) - 1, seconds, costs[-1]))

    plot_costs(results, "cost-vs-iteration.png")


if __name__ == "__main__":
    main()
