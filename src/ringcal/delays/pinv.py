"""Delay step of Granichin et al. (2025), Section 3.1: summation trick + pseudoinverse.

Given current positions, every usable measurement says

    times[a, b] - flight_time[a, b]  ≈  e[a] + r[b].

Summing these over each row (one transmitter) and each column (one receiver),
counting only usable cells, gives one equation per unknown:

    (cells in row a) * e[a] + sum over usable b of r[b]  = row sum of residuals
    sum over usable a of e[a] + (cells in column b) * r[b] = column sum of residuals

These are exactly the normal equations of least squares over all usable cells,
so nothing is lost by summing. With every cell usable (the square case) the
matrix is the paper's C from eq. (6), with its two row blocks in swapped order.
The matrix depends only on the problem, so its pseudoinverse is computed once.

The matrix is singular: adding s to every e and subtracting s from every r
changes nothing. The pseudoinverse returns the solution of smallest norm, which
is the one with sum(e) == sum(r). Transmitters or receivers without a single
usable cell get NaN delays.

How the pseudoinverse is computed matters. ``numpy.linalg.pinv`` treats
singular values below 1e-15 times the largest as zero; for this matrix the true
zero comes out of floating point at about 1e-12 for some n (e.g. n = 1024), so
it is inverted and the result is garbage. Instead we use the exactly known null
direction p = (1, ..., 1, -1, ..., -1) / sqrt(size): for a symmetric matrix M
whose only null direction is p, pinv(M) = inv(M + p p^T) - p p^T. No tolerance
is involved.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_array
from scipy.sparse.csgraph import connected_components

from ringcal.problem import CalibrationProblem, Delays


def is_connected(mask: NDArray[np.bool_]) -> bool:
    """Whether every transmitter and receiver is linked to every other one
    through a chain of usable cells (transmitter a heard by receiver b, ...)."""
    n_tx, n_rx = mask.shape
    w = csr_array(mask.astype(np.int8))
    graph = csr_array((n_tx + n_rx, n_tx + n_rx), dtype=np.int8)
    graph = graph.tolil()
    graph[:n_tx, n_tx:] = w
    n_components, _ = connected_components(graph.tocsr(), directed=False)
    return n_components == 1


def pinv_with_known_null_direction(
    matrix: NDArray[np.float64], null: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Pseudoinverse of a symmetric matrix whose null space is spanned by ``null``."""
    p = null / np.linalg.norm(null)
    pp = np.outer(p, p)
    return np.linalg.inv(matrix + pp) - pp


def summed_system_matrix(mask: NDArray[np.bool_]) -> NDArray[np.float64]:
    """Coefficient matrix of the summed equations, unknowns ordered ``(e, r)``.

    ``[[diag(row counts), W], [W^T, diag(column counts)]]`` with ``W = mask``,
    shape ``(n_tx + n_rx, n_tx + n_rx)``. Symmetric.
    """
    w = mask.astype(np.float64)
    return np.block([[np.diag(w.sum(axis=1)), w], [w.T, np.diag(w.sum(axis=0))]])


class PinvDelays:
    """Least-squares delays via a once-computed pseudoinverse (paper's method)."""

    def __init__(self) -> None:
        self._problem: CalibrationProblem | None = None

    def prepare(self, problem: CalibrationProblem) -> None:
        mask = problem.mask
        self._rows = mask.any(axis=1)  # transmitters with at least one usable cell
        self._cols = mask.any(axis=0)  # receivers with at least one usable cell
        reduced = mask[np.ix_(self._rows, self._cols)]

        if not is_connected(reduced):
            raise ValueError(
                "the usable measurements split into groups that never hear each other; "
                "delays cannot be estimated jointly"
            )
        matrix = summed_system_matrix(reduced)
        null = np.concatenate([np.ones(reduced.shape[0]), -np.ones(reduced.shape[1])])
        self.matrix = matrix
        self._pinv = pinv_with_known_null_direction(matrix, null)
        self._problem = problem

    def estimate(self, positions: NDArray[np.float64]) -> Delays:
        problem = self._problem
        if problem is None:
            raise RuntimeError("call prepare(problem) before estimate()")

        residual = np.where(problem.mask, problem.times - problem.flight_times(positions), 0.0)
        residual = residual[np.ix_(self._rows, self._cols)]
        rhs = np.concatenate([residual.sum(axis=1), residual.sum(axis=0)])
        solution = self._pinv @ rhs

        n_rows = int(self._rows.sum())
        e = np.full(problem.n_tx, np.nan)
        r = np.full(problem.n_rx, np.nan)
        e[self._rows] = solution[:n_rows]
        r[self._cols] = solution[n_rows:]
        return Delays(e=e, r=r)
