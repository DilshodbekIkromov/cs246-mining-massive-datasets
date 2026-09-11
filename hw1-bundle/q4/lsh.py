# Authors: Jessica Su, Wanzi Zhou, Pratyaksh Sharma, Dylan Liu, Ansh Shukla

import os
import time
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))


def l1(u, v):
    return np.sum(np.abs(u - v))


def load_data(filename):
    """Rows are 20x20 image patches flattened to 400 columns."""
    return np.genfromtxt(filename, delimiter=',')


def create_function(dimensions, thresholds):
    """One hash: bit i is 1 iff v[dimensions[i]] >= thresholds[i]."""
    def f(v):
        bits = [v[d] >= t for d, t in zip(dimensions, thresholds)]
        return "".join(str(int(b)) for b in bits)
    return f


def create_functions(k, L, num_dimensions=400, min_threshold=0, max_threshold=255):
    """L hash functions, each producing a k-bit key."""
    return [create_function(np.random.randint(0, num_dimensions, k),
                            np.random.randint(min_threshold, max_threshold + 1, k))
            for _ in range(L)]


def hash_vector(functions, v):
    return np.array([f(v) for f in functions])


def hash_data(functions, A):
    return np.array([hash_vector(functions, v) for v in A])


def get_candidates(hashed_A, hashed_point, query_index):
    """Rows sharing at least one of the query's L bucket keys."""
    return [i for i in range(len(hashed_A))
            if i != query_index and any(hashed_point == hashed_A[i])]


def lsh_setup(A, k=24, L=10):
    """Expensive: call once per (k, L)."""
    functions = create_functions(k=k, L=L)
    return functions, hash_data(functions, A)


def lsh_search(A, hashed_A, functions, query_index, num_neighbors=10):
    hashed_point = hash_vector(functions, A[query_index, :])
    candidates = get_candidates(hashed_A, hashed_point, query_index)
    distances = [(r, l1(A[r], A[query_index])) for r in candidates]
    return [r for r, _ in sorted(distances, key=lambda t: t[1])[:num_neighbors]]


def linear_search(A, query_index, num_neighbors):
    distances = [(r, l1(A[r], A[query_index]))
                 for r in range(A.shape[0]) if r != query_index]
    return [r for r, _ in sorted(distances, key=lambda t: t[1])[:num_neighbors]]


def plot(A, row_nums, base_filename):
    for row_num in row_nums:
        im = Image.fromarray(np.reshape(A[row_num, :], [20, 20]))
        if im.mode != 'RGB':
            im = im.convert('RGB')
        im.save("%s-%d.png" % (base_filename, row_num))


def lsh_search_robust(A, query_index, num_neighbors, k, L, max_attempts=20):
    """Re-hash until the query's buckets hold enough candidates."""
    for _ in range(max_attempts):
        functions, hashed_A = lsh_setup(A, k=k, L=L)
        neighbors = lsh_search(A, hashed_A, functions, query_index, num_neighbors)
        if len(neighbors) >= num_neighbors:
            return neighbors
    raise RuntimeError("LSH returned fewer than %d neighbors" % num_neighbors)


def error_measure(A, query_indices, lsh_neighbors, true_neighbors):
    """Mean over queries of (sum LSH distances) / (sum true distances)."""
    total = 0.0
    for j, q in enumerate(query_indices):
        approx = sum(l1(A[i], A[q]) for i in lsh_neighbors[j])
        exact = sum(l1(A[i], A[q]) for i in true_neighbors[j])
        total += approx / exact
    return total / len(query_indices)


def run_lsh(A, query_indices, num_neighbors, k, L):
    """Returns neighbor lists and mean search time (setup excluded)."""
    functions, hashed_A = lsh_setup(A, k=k, L=L)
    neighbors, elapsed = [], 0.0
    for q in query_indices:
        start = time.time()
        found = lsh_search(A, hashed_A, functions, q, num_neighbors)
        elapsed += time.time() - start
        if len(found) < num_neighbors:
            found = lsh_search_robust(A, q, num_neighbors, k, L)
        neighbors.append(found)
    return neighbors, elapsed / len(query_indices)


def run_linear(A, query_indices, num_neighbors):
    neighbors, elapsed = [], 0.0
    for q in query_indices:
        start = time.time()
        neighbors.append(linear_search(A, q, num_neighbors))
        elapsed += time.time() - start
    return neighbors, elapsed / len(query_indices)


def plot_error(xs, errors, xlabel, filename):
    plt.figure()
    plt.plot(xs, errors, marker="o")
    plt.xlabel(xlabel)
    plt.ylabel("error")
    plt.title("Error vs. %s" % xlabel)
    plt.grid(True)
    plt.savefig(filename, bbox_inches="tight")
    plt.close()


def problem4():
    A = load_data(os.path.join(HERE, "data", "patches.csv"))
    query_indices = [100 * j - 1 for j in range(1, 11)]  # columns 100..1000

    lsh_neighbors, lsh_time = run_lsh(A, query_indices, 3, k=24, L=10)
    true_neighbors, linear_time = run_linear(A, query_indices, 3)
    print("average LSH search time:    %.6f s" % lsh_time)
    print("average linear search time: %.6f s" % linear_time)
    print("error (k=24, L=10):         %.6f\n"
          % error_measure(A, query_indices, lsh_neighbors, true_neighbors))

    Ls = list(range(10, 21, 2))
    errors_L = []
    for L in Ls:
        neighbors, _ = run_lsh(A, query_indices, 3, k=24, L=L)
        errors_L.append(error_measure(A, query_indices, neighbors, true_neighbors))
        print("L = %2d -> error = %.6f" % (L, errors_L[-1]))
    plot_error(Ls, errors_L, "L", "error-vs-L.png")

    ks = list(range(16, 25, 2))
    errors_k = []
    for k in ks:
        neighbors, _ = run_lsh(A, query_indices, 3, k=k, L=10)
        errors_k.append(error_measure(A, query_indices, neighbors, true_neighbors))
        print("k = %2d -> error = %.6f" % (k, errors_k[-1]))
    plot_error(ks, errors_k, "k", "error-vs-k.png")

    query_index = 99
    lsh_top10 = lsh_search_robust(A, query_index, 10, k=24, L=10)
    linear_top10 = linear_search(A, query_index, 10)
    plot(A, [query_index], "patch-original")
    plot(A, lsh_top10, "patch-lsh")
    plot(A, linear_top10, "patch-linear")
    print("\nLSH top 10:    %s" % lsh_top10)
    print("linear top 10: %s" % linear_top10)


class TestLSH(unittest.TestCase):
    def test_l1(self):
        self.assertEqual(l1(np.array([1, 2, 3, 4]), np.array([2, 3, 2, 3])), 4)

    def test_hash_data(self):
        functions = [lambda v: sum(v), lambda v: sum(x * x for x in v)]
        A = np.array([[1, 2, 3], [4, 5, 6]])
        self.assertTrue(np.array_equal(hash_vector(functions, A[0, :]), [6, 14]))
        self.assertTrue(np.array_equal(hash_data(functions, A), [[6, 14], [15, 77]]))

    def test_linear_search(self):
        A = np.array([[0, 0], [1, 0], [5, 0], [0, 3]])
        self.assertEqual(linear_search(A, 0, 2), [1, 3])

    def test_error_measure(self):
        # Approximate neighbor is twice as far as the true one.
        A = np.array([[0, 0], [2, 0], [1, 0]])
        self.assertAlmostEqual(error_measure(A, [0], [[1]], [[2]]), 2.0)


if __name__ == '__main__':
#    unittest.main() ### TODO: Uncomment this to run tests
    problem4()
