"""Interface shared by all delay steps."""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from ringcal.problem import CalibrationProblem, Delays


class DelayStep(Protocol):
    """Estimates emit and receive delays given element positions.

    ``prepare`` is called once per problem, before the iterations start; it is
    the place for expensive set-up that depends only on the problem (such as a
    pseudoinverse). ``estimate`` is then called on every iteration.
    """

    def prepare(self, problem: CalibrationProblem) -> None: ...

    def estimate(self, positions: NDArray[np.float64]) -> Delays: ...
