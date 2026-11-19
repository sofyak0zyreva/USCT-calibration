"""One recording, ready for calibration.

A ``CalibrationProblem`` holds the measured times of flight of one recording:
which elements fired (one row each), which elements listened (one column
each), the times themselves in seconds, a mask of usable cells, and the speed
of sound. Real and synthetic data both end up in this form, and every later
step (delays, positions, solver, metrics) takes it as input.

The measurement model is

    times[a, b] = |x[tx_elements[a]] - x[rx_elements[b]]| / c + e[a] + r[b] + noise,

with one emit delay ``e`` per transmitter and one receive delay ``r`` per
receiver (see ``Delays``).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ringcal.geometry import RingGeometry


def _readonly(array: NDArray) -> NDArray:
    array.flags.writeable = False
    return array


def _element_indices(values: ArrayLike | None, n_elements: int, name: str) -> NDArray[np.int64]:
    if values is None:
        return _readonly(np.arange(n_elements))
    indices = np.array(values, dtype=np.int64).reshape(-1)
    if indices.size == 0:
        raise ValueError(f"{name} must not be empty")
    if np.unique(indices).size != indices.size:
        raise ValueError(f"{name} contains duplicates")
    bad = indices[(indices < 0) | (indices >= n_elements)]
    if bad.size:
        raise ValueError(f"{name} out of range [0, {n_elements}): {bad[:5].tolist()}")
    return _readonly(indices)


@dataclass(frozen=True, eq=False)
class CalibrationProblem:
    """Measured times of flight of one recording.

    Parameters
    ----------
    geometry:
        Nominal layout of the ring.
    times:
        Measured times in seconds, shape ``(n_tx, n_rx)``. Row ``a`` belongs to
        transmitter ``tx_elements[a]``, column ``b`` to receiver ``rx_elements[b]``.
        Cells that are masked out may hold anything, including NaN.
    c:
        Speed of sound in m/s.
    tx_elements:
        Elements that fired, in row order. ``None`` means every element.
    rx_elements:
        Elements that listened, in column order. ``None`` means every element.
    mask:
        ``True`` where a cell may be used, shape ``(n_tx, n_rx)``. ``None`` means
        every cell. Cells where an element would hear itself are always
        excluded, whatever the given mask says.

    All arrays are copied and made read-only, so a problem never changes after
    it is created. Use ``without_elements`` to derive a problem with more cells
    masked out.
    """

    geometry: RingGeometry
    times: NDArray[np.float64]
    c: float
    tx_elements: NDArray[np.int64] = None  # type: ignore[assignment]
    rx_elements: NDArray[np.int64] = None  # type: ignore[assignment]
    mask: NDArray[np.bool_] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        n = self.geometry.n_elements
        tx = _element_indices(self.tx_elements, n, "tx_elements")
        rx = _element_indices(self.rx_elements, n, "rx_elements")

        times = np.array(self.times, dtype=np.float64)
        if times.shape != (tx.size, rx.size):
            raise ValueError(
                f"times has shape {times.shape}, expected (n_tx, n_rx) = ({tx.size}, {rx.size})"
            )

        if self.mask is None:
            mask = np.ones(times.shape, dtype=bool)
        else:
            mask = np.array(self.mask, dtype=bool)
            if mask.shape != times.shape:
                raise ValueError(f"mask has shape {mask.shape}, expected {times.shape}")
        mask &= tx[:, None] != rx[None, :]  # an element hearing itself says nothing

        if not np.isfinite(times[mask]).all():
            raise ValueError("times must be finite wherever the mask is True")
        if not (np.isfinite(self.c) and self.c > 0):
            raise ValueError(f"c must be a positive number, got {self.c}")

        object.__setattr__(self, "tx_elements", tx)
        object.__setattr__(self, "rx_elements", rx)
        object.__setattr__(self, "times", _readonly(times))
        object.__setattr__(self, "mask", _readonly(mask))
        object.__setattr__(self, "c", float(self.c))

    @property
    def n_tx(self) -> int:
        return int(self.tx_elements.size)

    @property
    def n_rx(self) -> int:
        return int(self.rx_elements.size)

    @property
    def active_elements(self) -> NDArray[np.int64]:
        """Elements that fired or listened, sorted. Only these can be calibrated."""
        return np.union1d(self.tx_elements, self.rx_elements)

    def flight_times(self, positions: NDArray[np.float64]) -> NDArray[np.float64]:
        """Pure travel times |x_tx - x_rx| / c for given element positions.

        ``positions`` has shape ``(n_elements, 2)``, indexed by element number;
        the result has shape ``(n_tx, n_rx)``, like ``times``.
        """
        x = np.asarray(positions, dtype=np.float64)
        if x.shape != (self.geometry.n_elements, 2):
            raise ValueError(
                f"positions has shape {x.shape}, expected ({self.geometry.n_elements}, 2)"
            )
        diff = x[self.tx_elements][:, None, :] - x[self.rx_elements][None, :, :]
        return np.linalg.norm(diff, axis=-1) / self.c

    def without_elements(self, elements: Iterable[int]) -> CalibrationProblem:
        """A copy with every row and column of ``elements`` masked out.

        Typical use: elements known to be broken.
        """
        drop = np.array(list(elements), dtype=np.int64)
        mask = self.mask.copy()
        mask[np.isin(self.tx_elements, drop), :] = False
        mask[:, np.isin(self.rx_elements, drop)] = False
        return dataclasses.replace(self, mask=mask)


@dataclass(frozen=True, eq=False)
class Delays:
    """Emit and receive delays in seconds.

    ``e[a]`` belongs to transmitter ``tx_elements[a]``, ``r[b]`` to receiver
    ``rx_elements[b]`` of the problem they were estimated for.
    """

    e: NDArray[np.float64]
    r: NDArray[np.float64]

    def __post_init__(self) -> None:
        e = np.array(self.e, dtype=np.float64)
        r = np.array(self.r, dtype=np.float64)
        if e.ndim != 1 or r.ndim != 1:
            raise ValueError(f"e and r must be 1-D, got shapes {e.shape} and {r.shape}")
        object.__setattr__(self, "e", _readonly(e))
        object.__setattr__(self, "r", _readonly(r))

    @classmethod
    def zeros(cls, problem: CalibrationProblem) -> Delays:
        """All-zero delays sized for ``problem``: the usual starting point."""
        return cls(e=np.zeros(problem.n_tx), r=np.zeros(problem.n_rx))
