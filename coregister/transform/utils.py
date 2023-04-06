import scipy
import numpy
import numpy as np


def solve(A, w, r, x0, dst, check_finite=False):
    """regularized linear least squares

    Parameters
    ----------
    A : :class:`numpy.ndarray`
        ndata x nparameter array
    w : :class:`numpy.ndarray`
        ndata x ndata diagonal weight matrix
    r : :class:`numpy.ndarray`
        nparameter x nparameter diagonal
        regularization matrix
    x0 : :class:`numpy.ndarray`
        starting point parameter array
        nparaeters x 3
    dst : :class:`numpy.ndarray`
        ndata x 3 Cartesian coordinates
        of transform destination.

    Returns
    -------
    x : :class:`numpy.ndarray`
        nparameter x 3 solution

    """
    ATW = A.T.dot(w)
    K = ATW.dot(A) + numpy.diag(r)
    lu, piv = scipy.linalg.lu_factor(
        K, overwrite_a=True, check_finite=check_finite)
    rhs_arr = (
        numpy.einsum("i,ij", r, x0)[:, None] +
        numpy.einsum("ij,jk->ki", ATW, dst))
    x = scipy.linalg.lu_solve(
        (lu, piv), rhs_arr.T,
        overwrite_b=True, check_finite=check_finite)
    return x


def initialize_weights_vec(weights_vec, example_pts):
    if isinstance(weights_vec, np.ndarray):
        return weights_vec
    if weights_vec is None:
        return np.ones(example_pts.shape[0], dtype=np.float64)
    return np.array(weights_vec, dtype=np.float64)
