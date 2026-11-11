"""Ring geometry: slots, arcs, gaps, nominal element positions.

The ring is a circle divided into equally spaced *slots*. Elements fill the
slots in ``n_arcs`` arcs of ``elements_per_arc`` consecutive slots each, and
every arc is followed by ``gap_slots`` empty slots. The lab's ring has 2112
slots: 8 arcs of 256 elements with gaps of 8, diameter about 0.22 m.

Elements are numbered 0, 1, 2, ... along the ring, starting at the first slot
of arc 0. ``first_slot`` says which slot that is, and ``angle_offset`` is the
angle of slot 0, so the whole numbering can be rotated without touching the
layout.

This class describes the *construction* of a scanner model only. Things that
belong to one device or one recording (which elements transmitted, which are
broken, sampling rate, speed of sound) live with the measurement data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class RingGeometry:
    """Nominal layout of a ring transducer array.

    Parameters
    ----------
    n_arcs, elements_per_arc, gap_slots:
        Layout: ``n_arcs`` arcs of ``elements_per_arc`` elements, each arc
        followed by ``gap_slots`` empty slots.
    radius:
        Nominal ring radius in metres.
    first_slot:
        Slot occupied by element 0.
    angle_offset:
        Angle of slot 0 in radians, measured counter-clockwise from the x-axis.
    """

    n_arcs: int = 8
    elements_per_arc: int = 256
    gap_slots: int = 8
    radius: float = 0.11
    first_slot: int = 0
    angle_offset: float = 0.0

    def __post_init__(self) -> None:
        if self.n_arcs < 1:
            raise ValueError(f"n_arcs must be at least 1, got {self.n_arcs}")
        if self.elements_per_arc < 1:
            raise ValueError(f"elements_per_arc must be at least 1, got {self.elements_per_arc}")
        if self.gap_slots < 0:
            raise ValueError(f"gap_slots must be non-negative, got {self.gap_slots}")
        if not self.radius > 0:
            raise ValueError(f"radius must be positive, got {self.radius}")
        if not 0 <= self.first_slot < self.n_slots:
            raise ValueError(f"first_slot must be in [0, {self.n_slots}), got {self.first_slot}")

    # --- sizes ---------------------------------------------------------------

    @property
    def n_elements(self) -> int:
        return self.n_arcs * self.elements_per_arc

    @property
    def n_slots(self) -> int:
        return self.n_arcs * (self.elements_per_arc + self.gap_slots)

    # --- per-element arrays --------------------------------------------------

    @property
    def element_arcs(self) -> NDArray[np.int64]:
        """Arc index of every element, shape ``(n_elements,)``."""
        return np.arange(self.n_elements) // self.elements_per_arc

    @property
    def element_slots(self) -> NDArray[np.int64]:
        """Slot occupied by every element, shape ``(n_elements,)``."""
        k = np.arange(self.n_elements)
        arc, pos = np.divmod(k, self.elements_per_arc)
        slot = arc * (self.elements_per_arc + self.gap_slots) + pos
        return (slot + self.first_slot) % self.n_slots

    @property
    def element_angles(self) -> NDArray[np.float64]:
        """Nominal angle of every element in radians, shape ``(n_elements,)``."""
        return self.angle_offset + 2 * np.pi * self.element_slots / self.n_slots

    def nominal_positions(self) -> NDArray[np.float64]:
        """Ideal element positions in metres, shape ``(n_elements, 2)``."""
        theta = self.element_angles
        return self.radius * np.column_stack([np.cos(theta), np.sin(theta)])
