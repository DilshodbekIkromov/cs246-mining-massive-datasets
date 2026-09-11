"""Q1(e): compare the SVD of M with the eigendecomposition of M^T M."""

import numpy as np
from scipy.linalg import svd, eigh

M = np.array([[1., 2.],
              [2., 1.],
              [3., 4.],
              [4., 3.]])

U, Sigma, Vt = svd(M, full_matrices=False)

evals, evecs = eigh(M.T @ M)
order = np.argsort(evals)[::-1]      # descending, largest eigenvalue first
evals, evecs = evals[order], evecs[:, order]

np.set_printoptions(precision=6, suppress=True)
print("U =\n%s\n" % U)
print("Sigma =\n%s\n" % Sigma)
print("V^T =\n%s\n" % Vt)
print("Evals =\n%s\n" % evals)
print("Evecs =\n%s\n" % evecs)

# V (columns of Vt transposed) and Evecs agree up to the sign of each column,
# and the eigenvalues are the squares of the singular values.
print("V =\n%s\n" % Vt.T)
print("Sigma^2 =\n%s" % (Sigma ** 2))
