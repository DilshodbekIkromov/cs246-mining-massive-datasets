"""Q2: PageRank and HITS on Spark."""

import os
import sys

import numpy as np

os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
from pyspark import SparkContext

HERE = os.path.dirname(os.path.abspath(__file__))

BETA = 0.8
ITERATIONS = 40
TOP_K = 5


def load_edges(sc, filename):
    return (sc.textFile(os.path.join(HERE, "data", filename))
            .map(lambda line: tuple(int(x) for x in line.split()))
            .distinct()
            .cache())


def pagerank(edges, n, beta=BETA, iterations=ITERATIONS):
    links = edges.groupByKey().mapValues(list).cache()
    r = np.full(n, 1.0 / n)

    for _ in range(iterations):
        contributions = (links
                         .flatMap(lambda kv: [(j, r[kv[0] - 1] / len(kv[1])) for j in kv[1]])
                         .reduceByKey(lambda a, b: a + b)
                         .collect())
        new_r = np.full(n, (1.0 - beta) / n)
        for j, value in contributions:
            new_r[j - 1] += beta * value
        r = new_r

    return r


def hits(edges, n, iterations=ITERATIONS):
    outgoing = edges.groupByKey().mapValues(list).cache()
    incoming = edges.map(lambda e: (e[1], e[0])).groupByKey().mapValues(list).cache()
    h = np.ones(n)

    for _ in range(iterations):
        a = accumulate(incoming, h, n)
        a /= a.max()
        h = accumulate(outgoing, a, n)
        h /= h.max()

    return h, a


def accumulate(adjacency, vector, n):
    """Sum vector[j] over the neighbours j of each node."""
    totals = (adjacency
              .map(lambda kv: (kv[0], sum(vector[j - 1] for j in kv[1])))
              .collect())
    out = np.zeros(n)
    for i, value in totals:
        out[i - 1] = value
    return out


def show(title, scores, k=TOP_K):
    order = np.argsort(-scores, kind="stable")
    print("%s, highest %d:" % (title, k))
    for i in order[:k]:
        print("  node %4d: %.10f" % (i + 1, scores[i]))
    print("%s, lowest %d:" % (title, k))
    for i in order[::-1][:k]:
        print("  node %4d: %.10f" % (i + 1, scores[i]))
    print()


def main():
    filename = sys.argv[1] if len(sys.argv) > 1 else "graph-full.txt"
    sc = SparkContext("local[*]", "PageRankHITS")
    sc.setLogLevel("WARN")

    edges = load_edges(sc, filename)
    n = edges.flatMap(lambda e: e).max()
    print("%s: %d nodes, %d distinct edges\n" % (filename, n, edges.count()))

    show("PageRank", pagerank(edges, n))
    h, a = hits(edges, n)
    show("Hubbiness", h)
    show("Authority", a)

    sc.stop()


if __name__ == "__main__":
    main()
