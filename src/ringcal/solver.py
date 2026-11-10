"""Alternating solver.

Will repeat ``DelayStep`` then ``PositionStep`` until a ``StoppingRule``
fires, recording per-iteration history (errors, per-phase timing) via hooks.
"""
