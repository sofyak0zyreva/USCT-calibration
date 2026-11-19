"""Delay steps: estimate emit/receive delays given current positions.

Implemented: ``PinvDelays`` (pseudoinverse, as in the paper).
Planned: a block-structured solve (Topic A).
"""

from ringcal.delays.base import DelayStep
from ringcal.delays.pinv import PinvDelays

__all__ = ["DelayStep", "PinvDelays"]
