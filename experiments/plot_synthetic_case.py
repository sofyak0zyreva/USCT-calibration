"""Plot one synthetic case generated as in Granichin et al. (2025), Section 4.1.

Left: ideal and actual sensor positions (like the paper's Fig. 3), with the
position errors exaggerated so they are visible. Right: the measured times.

Usage:
    uv run python experiments/plot_synthetic_case.py [n] [seed] [output.png]
"""

import sys

import matplotlib.pyplot as plt
import numpy as np

from ringcal.synthetic import paper_case


def exaggeration_for(spacing: float, max_shift: float) -> int:
    """Largest of 1, 2, 5, 10, 20, 50, ... that keeps every drawn shift below a
    quarter of the neighbour spacing, so the picture never suggests that
    sensors swap places when they do not."""
    factor = 1
    for candidate in (2, 5, 10, 20, 50, 100, 200, 500):
        if candidate * max_shift <= 0.25 * spacing:
            factor = candidate
    return factor


def main(n: int = 32, seed: int = 0, output: str = "synthetic_case.png") -> None:
    case = paper_case(n, rng=seed)
    ideal = case.problem.geometry.nominal_positions()
    error = case.positions - ideal
    spacing = float(np.linalg.norm(ideal[1] - ideal[0]))
    factor = exaggeration_for(spacing, float(np.linalg.norm(error, axis=1).max()))
    shown = ideal + factor * error

    fig, (ax_ring, ax_times) = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)

    mm = 1e3
    theta = np.linspace(0, 2 * np.pi, 400)
    radius = case.problem.geometry.radius * mm
    ax_ring.plot(radius * np.cos(theta), radius * np.sin(theta), c="k", lw=1)
    ax_ring.scatter(*(ideal * mm).T, s=30, c="tab:green", label="ideal", zorder=3)
    label = "actual" if factor == 1 else f"actual (error ×{factor})"
    ax_ring.scatter(*(shown * mm).T, s=30, c="tab:red", label=label, zorder=3)
    for a, b in zip(ideal * mm, shown * mm, strict=True):
        ax_ring.plot([a[0], b[0]], [a[1], b[1]], c="0.6", lw=0.8)
    ax_ring.set_aspect("equal")
    ax_ring.set_xlabel("x, mm")
    ax_ring.set_ylabel("y, mm")
    max_err = np.abs(error).max() * mm
    ax_ring.set_title(f"n = {n}, seed = {seed}; max coordinate error {max_err:.3f} mm")
    ax_ring.legend(loc="center")

    image = ax_times.imshow(case.problem.times * 1e6, cmap="viridis")
    ax_times.set_title("measured times m_ij, µs (diagonal empty)")
    ax_times.set_xlabel("receiver j")
    ax_times.set_ylabel("emitter i")
    fig.colorbar(image, ax=ax_times, shrink=0.8)

    fig.savefig(output, dpi=150)
    print(f"saved {output}")


if __name__ == "__main__":
    args = sys.argv[1:]
    main(
        n=int(args[0]) if len(args) > 0 else 32,
        seed=int(args[1]) if len(args) > 1 else 0,
        output=args[2] if len(args) > 2 else "synthetic_case.png",
    )
