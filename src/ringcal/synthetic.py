"""Synthetic test data, as in Granichin et al. (2025), Section 4.1.

The paper's recipe, for n sensors:

1. Place n points equidistantly on a circle of diameter d = 0.22 m.
2. Shift each coordinate of each point by uniform noise on
   [-0.001 d, 0.001 d] (about ±0.22 mm).
3. Compute the true times of flight t_ij = |x_i - x_j| / c with c = 1500 m/s.
4. Emit delays:    e_i = 0.01 * (1/n) * sum_{j != i} t_ij
   Receive delays: r_i = e_i + xi_i,  xi_i ~ U[-0.01 e_i, 0.01 e_i]
5. Measurements:  m_ij = t_ij + e_i + r_j + eps_ij,  eps_ij ~ U[-4e-8, 4e-8] s,
   for i != j (an element does not measure itself).

Every number above is a keyword argument of ``paper_case`` with the paper's
value as default. Random draws are made in a fixed order (positions, then
receive-delay spread, then time noise) from the generator passed in, so the
same seed always gives the same case.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ringcal.geometry import RingGeometry
from ringcal.problem import CalibrationProblem, Delays


def pairwise_distances(positions: NDArray[np.float64]) -> NDArray[np.float64]:
    """Distances between all pairs of points, shape ``(n, n)``."""
    diff = positions[:, None, :] - positions[None, :, :]
    return np.linalg.norm(diff, axis=-1)


def forward_times(positions: NDArray[np.float64], delays: Delays, c: float) -> NDArray[np.float64]:
    """Noise-free measurement model for the square case (every element fires and listens).

    ``times[i, j] = |x_i - x_j| / c + e_i + r_j``, shape ``(n, n)``.
    The diagonal is computed too, but it has no physical meaning.
    """
    return pairwise_distances(positions) / c + delays.e[:, None] + delays.r[None, :]


@dataclass(frozen=True, eq=False)
class SyntheticCase:
    """A generated problem together with the true answer."""

    problem: CalibrationProblem
    positions: NDArray[np.float64]  # true positions, shape (n, 2), metres
    delays: Delays  # true delays, seconds


def paper_case(
    n: int,
    rng: np.random.Generator | int | None = None,
    *,
    diameter: float = 0.22,
    c: float = 1500.0,
    position_noise: float = 0.001,
    delay_fraction: float = 0.01,
    receive_spread: float = 0.01,
    time_noise: float = 4e-8,
) -> SyntheticCase:
    """Generate one synthetic case following Granichin et al. (2025), Section 4.1.

    Parameters
    ----------
    n:
        Number of sensors, equidistant on the circle (no arcs, no gaps).
    rng:
        A ``numpy.random.Generator``, or a seed for one.
    diameter:
        Ring diameter in metres (paper: 0.22).
    c:
        Speed of sound in m/s (paper: 1500).
    position_noise:
        Half-width of the uniform position error per coordinate, as a fraction
        of the diameter (paper: 0.001, i.e. ±0.22 mm).
    delay_fraction:
        Emit delay as a fraction of the element's mean time of flight
        (paper: 0.01).
    receive_spread:
        Receive delay differs from the emit delay by up to this fraction of it
        (paper: 0.01).
    time_noise:
        Half-width of the uniform timing noise in seconds (paper: 4e-8, from
        25 MHz sampling).
    """
    if n < 3:
        raise ValueError(f"need at least 3 sensors, got {n}")
    rng = np.random.default_rng(rng)

    geometry = RingGeometry(n_arcs=1, elements_per_arc=n, gap_slots=0, radius=diameter / 2)

    shift = position_noise * diameter
    positions = geometry.nominal_positions() + rng.uniform(-shift, shift, size=(n, 2))

    true_times = pairwise_distances(positions) / c  # diagonal is 0
    e = delay_fraction * true_times.sum(axis=1) / n  # paper divides by n, not n - 1
    r = e + rng.uniform(-receive_spread * e, receive_spread * e)
    delays = Delays(e=e, r=r)

    times = forward_times(positions, delays, c) + rng.uniform(-time_noise, time_noise, (n, n))
    np.fill_diagonal(times, np.nan)  # no self-measurements; masked by CalibrationProblem

    positions.flags.writeable = False
    return SyntheticCase(
        problem=CalibrationProblem(geometry, times, c=c),
        positions=positions,
        delays=delays,
    )
