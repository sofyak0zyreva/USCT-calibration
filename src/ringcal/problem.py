"""Calibration problem and result types.

Will hold ``CalibrationProblem`` (measured times, shape ``(n_tx, n_rx)``, in
seconds, plus a validity mask), ``Delays`` (``e`` per transmitter, ``r`` per
receiver) and ``CalibrationResult`` (positions, delays, history, settings).
Real and synthetic data both end up as a ``CalibrationProblem``.
"""
