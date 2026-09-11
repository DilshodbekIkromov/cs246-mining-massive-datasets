"""Q2: iterative k-means on Spark with Euclidean and Manhattan distance.

Each iteration maps every point to (nearest centroid index, (point, 1, cost)),
reduces those by key to get the per-cluster sums, and divides to obtain the new
centroids.  The cost of the iteration comes for free from the same pass, since
the assignment step already computes the distance to the closest centroid.
"""

import os
import sys

import numpy as np

os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
from pyspark import SparkContext

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "data.txt")
INITS = {"c1": os.path.join(HERE, "data", "c1.txt"),
         "c2": os.path.join(HERE, "data", "c2.txt")}

MAX_ITER = 20
K = 10


def load_vectors(filename):
    with open(filename) as f:
        return [np.array([float(x) for x in line.split()]) for line in f if line.strip()]


def euclidean_cost(point, centroids):
    """Squared Euclidean distance to the closest centroid (phi's summand)."""
    d = np.sum((centroids - point) ** 2, axis=1)
    index = int(np.argmin(d))
    return index, float(d[index])


def manhattan_cost(point, centroids):
    """Manhattan distance to the closest centroid (psi's summand)."""
    d = np.sum(np.abs(centroids - point), axis=1)
    index = int(np.argmin(d))
    return index, float(d[index])


def kmeans(points, initial_centroids, assign, max_iter=MAX_ITER):
    """Run iterative k-means, returning the cost of each iteration."""
    centroids = np.array(initial_centroids)
    costs = []

    for _ in range(max_iter):
        def to_cluster(p, c=centroids):
            index, cost = assign(p, c)
            return index, (p, 1, cost)

        summed = (points
                  .map(to_cluster)
                  .reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1], a[2] + b[2]))
                  .collect())

        # The cost is measured against the centroids used for this assignment.
        costs.append(sum(total for _, (_, _, total) in summed))

        for index, (vector_sum, count, _) in summed:
            centroids[index] = vector_sum / count

    return costs


def plot_costs(costs_by_init, ylabel, title, filename):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure()
    for name, costs in costs_by_init.items():
        plt.plot(range(1, len(costs) + 1), costs, marker="o", label="%s.txt" % name)
    plt.xlabel("iteration")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(range(1, MAX_ITER + 1, 2))
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(HERE, filename), bbox_inches="tight")
    plt.close()


def percentage_change(costs):
    """Percentage drop in cost between iteration 1 and iteration 10."""
    return 100.0 * (costs[0] - costs[9]) / costs[0]


def main():
    sc = SparkContext("local[*]", "KMeans")
    sc.setLogLevel("WARN")

    points = (sc.textFile(DATA)
              .map(lambda line: np.array([float(x) for x in line.split()]))
              .cache())

    for label, ylabel, assign, filename in [
            ("Euclidean", "cost phi", euclidean_cost, "cost-euclidean.png"),
            ("Manhattan", "cost psi", manhattan_cost, "cost-manhattan.png")]:
        costs_by_init = {}
        for name, path in INITS.items():
            costs = kmeans(points, load_vectors(path), assign)
            costs_by_init[name] = costs
            print("\n%s distance, %s.txt" % (label, name))
            for i, cost in enumerate(costs, start=1):
                print("  iteration %2d: %.4f" % (i, cost))
            print("  percentage change after 10 iterations: %.2f%%"
                  % percentage_change(costs))
        plot_costs(costs_by_init, ylabel, "%s k-means cost" % label, filename)

    sc.stop()


if __name__ == "__main__":
    main()
