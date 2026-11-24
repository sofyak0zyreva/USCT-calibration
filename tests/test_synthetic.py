"""Tests for the synthetic generator: it must follow Granichin et al. (2025), Section 4.1."""

import numpy as np
import pytest

from ringcal.problem import Delays
from ringcal.synthetic import forward_times, pairwise_distances, paper_case

D = 0.22  # ring diameter, m
C = 1500.0  # speed of sound, m/s


# --- the measurement model -------------------------------------------------------


def test_pairwise_distances_on_a_right_triangle() -> None:
    x = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]])
    expected = np.array([[0, 3, 4], [3, 0, 5], [4, 5, 0]], dtype=float)
    np.testing.assert_allclose(pairwise_distances(x), expected)


def test_forward_times_adds_emit_delay_of_row_and_receive_delay_of_column() -> None:
    x = np.array([[0.0, 0.0], [3.0, 0.0]])
    delays = Delays(e=np.array([10.0, 20.0]), r=np.array([1.0, 2.0]))
    t = forward_times(x, delays, c=1.0)
    assert t[0, 1] == 3.0 + 10.0 + 2.0  # 0 fires (e_0), 1 hears (r_1)
    assert t[1, 0] == 3.0 + 20.0 + 1.0  # 1 fires (e_1), 0 hears (r_0)


# --- shape and reproducibility -------------------------------------------------------


@pytest.mark.parametrize("n", [16, 32, 64])
def test_shapes(n: int) -> None:
    case = paper_case(n, rng=0)
    assert case.problem.times.shape == (n, n)
    assert case.positions.shape == (n, 2)
    assert case.delays.e.shape == (n,) and case.delays.r.shape == (n,)
    assert case.problem.c == C


def test_same_seed_gives_same_case() -> None:
    a, b = paper_case(32, rng=7), paper_case(32, rng=7)
    np.testing.assert_array_equal(a.positions, b.positions)
    np.testing.assert_array_equal(a.problem.times, b.problem.times)


def test_different_seeds_give_different_cases() -> None:
    a, b = paper_case(32, rng=1), paper_case(32, rng=2)
    assert not np.array_equal(a.positions, b.positions)


def test_diagonal_is_masked_out() -> None:
    case = paper_case(16, rng=0)
    assert np.array_equal(case.problem.mask, ~np.eye(16, dtype=bool))
    assert np.isnan(np.diag(case.problem.times)).all()


# --- each step of the paper's recipe --------------------------------------------------


def test_ring_is_equidistant_with_paper_diameter() -> None:
    case = paper_case(32, rng=0)
    g = case.problem.geometry
    assert g.n_elements == 32 and g.gap_slots == 0
    assert g.radius == pytest.approx(D / 2)


def test_without_position_noise_positions_are_ideal() -> None:
    case = paper_case(32, rng=0, position_noise=0.0)
    np.testing.assert_allclose(case.positions, case.problem.geometry.nominal_positions())


def test_position_errors_are_within_one_thousandth_of_diameter() -> None:
    case = paper_case(64, rng=0)
    err = case.positions - case.problem.geometry.nominal_positions()
    assert np.abs(err).max() <= 0.001 * D
    assert np.abs(err).max() > 0.5 * 0.001 * D  # really perturbed, not just tiny


def test_emit_delay_is_one_percent_of_mean_time_of_flight() -> None:
    n = 32
    case = paper_case(n, rng=0)
    tof = pairwise_distances(case.positions) / C
    np.testing.assert_allclose(case.delays.e, 0.01 * tof.sum(axis=1) / n)


def test_receive_delay_is_within_one_percent_of_emit_delay() -> None:
    case = paper_case(64, rng=0)
    e, r = case.delays.e, case.delays.r
    assert np.all(np.abs(r - e) <= 0.01 * e)
    assert not np.array_equal(r, e)


def test_without_time_noise_times_follow_the_model_exactly() -> None:
    case = paper_case(32, rng=0, time_noise=0.0)
    model = forward_times(case.positions, case.delays, C)
    off = case.problem.mask
    np.testing.assert_allclose(case.problem.times[off], model[off], rtol=1e-14)


def test_time_noise_is_within_40_ns() -> None:
    case = paper_case(64, rng=0)
    model = forward_times(case.positions, case.delays, C)
    off = case.problem.mask
    noise = case.problem.times[off] - model[off]
    assert np.abs(noise).max() <= 4e-8
    assert np.abs(noise).max() > 3e-8  # really noisy


# --- magnitudes quoted in the paper ----------------------------------------------------


def test_magnitudes_match_the_paper() -> None:
    case = paper_case(64, rng=0)
    tof = pairwise_distances(case.positions) / C
    assert tof.max() <= 1.5e-4 * 1.002  # "at most 1.5e-4 s" (plus the tiny position error)
    assert 5e-7 < case.delays.e.mean() < 1e-6  # 1% of a mean flight time of about 90 µs


def test_too_few_sensors_raises() -> None:
    with pytest.raises(ValueError):
        paper_case(2, rng=0)
