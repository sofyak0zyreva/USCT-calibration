"""Tests for the delay step of Granichin et al. (2025), Section 3.1."""

import numpy as np
import pytest

from ringcal.delays import PinvDelays
from ringcal.delays.pinv import pinv_with_known_null_direction, summed_system_matrix
from ringcal.problem import CalibrationProblem
from ringcal.synthetic import paper_case


def paper_matrix_c(n: int) -> np.ndarray:
    """Eq. (6) of the paper, typed in literally."""
    ones, eye = np.ones((n, n)), np.eye(n)
    return np.block([[ones, (n - 1) * eye], [(n - 1) * eye, ones]]) - np.eye(2 * n)


def gauge_shift(estimated_e: np.ndarray, true_e: np.ndarray) -> float:
    """The constant s with estimated_e ≈ true_e + s."""
    return float(np.mean(estimated_e - true_e))


# --- the system matrix ---------------------------------------------------------------


def test_square_matrix_is_papers_c_with_row_blocks_swapped() -> None:
    n = 8
    ours = summed_system_matrix(~np.eye(n, dtype=bool))
    swapped = np.vstack([ours[n:], ours[:n]])  # paper lists column sums first
    np.testing.assert_array_equal(swapped, paper_matrix_c(n))


def test_papers_c_has_the_eigenvalues_stated_in_the_paper() -> None:
    n = 8
    eig = np.sort(np.linalg.eigvals(paper_matrix_c(n)).real)
    expected = np.sort([2 * (n - 1), 0] + [-n] * (n - 1) + [n - 2] * (n - 1))
    np.testing.assert_allclose(eig, expected, atol=1e-10)


def test_matrix_is_singular_along_e_plus_s_r_minus_s() -> None:
    n = 8
    ours = summed_system_matrix(~np.eye(n, dtype=bool))
    p = np.concatenate([np.ones(n), -np.ones(n)])  # the paper's eigenvector t
    np.testing.assert_allclose(ours @ p, 0.0)
    assert np.linalg.matrix_rank(ours) == 2 * n - 1


# --- recovering delays --------------------------------------------------------------------


@pytest.mark.parametrize("n", [16, 64])
def test_true_positions_and_no_noise_give_true_delays_up_to_shift(n: int) -> None:
    case = paper_case(n, rng=0, time_noise=0.0)
    step = PinvDelays()
    step.prepare(case.problem)
    d = step.estimate(case.positions)

    s = gauge_shift(d.e, case.delays.e)
    np.testing.assert_allclose(d.e, case.delays.e + s, rtol=0, atol=1e-18)
    np.testing.assert_allclose(d.r, case.delays.r - s, rtol=0, atol=1e-18)


def test_estimate_satisfies_sum_e_equals_sum_r() -> None:
    case = paper_case(32, rng=3)
    step = PinvDelays()
    step.prepare(case.problem)
    d = step.estimate(case.positions)
    assert d.e.sum() == pytest.approx(d.r.sum(), abs=1e-18)


def test_with_paper_noise_delays_are_recovered_well_below_one_sample() -> None:
    case = paper_case(64, rng=1)  # timing noise up to ±40 ns
    step = PinvDelays()
    step.prepare(case.problem)
    d = step.estimate(case.positions)
    s = gauge_shift(d.e, case.delays.e)
    # summing ~63 noisy cells per equation averages the noise down
    assert np.abs(d.e - (case.delays.e + s)).max() < 1.5e-8
    assert np.abs(d.r - (case.delays.r - s)).max() < 1.5e-8


def test_pseudoinverse_is_computed_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    original = np.linalg.inv
    monkeypatch.setattr(np.linalg, "inv", lambda a: calls.append(1) or original(a))
    case = paper_case(16, rng=0)
    step = PinvDelays()
    step.prepare(case.problem)
    for _ in range(5):
        step.estimate(case.positions)
    assert len(calls) == 1


# --- the pseudoinverse itself ------------------------------------------------------------


def test_matches_numpy_pinv_where_numpy_is_reliable() -> None:
    n = 64
    m = summed_system_matrix(~np.eye(n, dtype=bool))
    null = np.concatenate([np.ones(n), -np.ones(n)])
    np.testing.assert_allclose(
        pinv_with_known_null_direction(m, null), np.linalg.pinv(m), atol=1e-14
    )


@pytest.mark.parametrize("n", [64, 1024])
def test_satisfies_moore_penrose_conditions(n: int) -> None:
    m = summed_system_matrix(~np.eye(n, dtype=bool))
    null = np.concatenate([np.ones(n), -np.ones(n)])
    p = pinv_with_known_null_direction(m, null)
    scale = np.abs(m).max()
    np.testing.assert_allclose(m @ p @ m, m, atol=1e-10 * scale)
    np.testing.assert_allclose(p @ m @ p, p, atol=1e-10 / scale)
    np.testing.assert_allclose(m @ p, (m @ p).T, atol=1e-12)


def test_regression_n1024_where_numpy_pinv_fails() -> None:
    """numpy's default pinv tolerance keeps a rounding-error singular value at
    n = 1024 and returns delays wrong by ~400 ns; our version must not."""
    case = paper_case(1024, rng=0)
    step = PinvDelays()
    step.prepare(case.problem)
    d = step.estimate(case.positions)
    s = gauge_shift(d.e, case.delays.e)
    assert np.abs(d.e - (case.delays.e + s)).max() < 1e-8


# --- usable cells only ----------------------------------------------------------------------


def test_broken_element_gets_nan_and_others_are_still_recovered() -> None:
    case = paper_case(16, rng=0, time_noise=0.0)
    problem = case.problem.without_elements([5])
    step = PinvDelays()
    step.prepare(problem)
    d = step.estimate(case.positions)

    assert np.isnan(d.e[5]) and np.isnan(d.r[5])
    keep = np.arange(16) != 5
    s = gauge_shift(d.e[keep], case.delays.e[keep])
    np.testing.assert_allclose(d.e[keep], case.delays.e[keep] + s, rtol=0, atol=1e-18)
    np.testing.assert_allclose(d.r[keep], case.delays.r[keep] - s, rtol=0, atol=1e-18)


def test_masked_cells_are_ignored_even_if_their_times_are_garbage() -> None:
    case = paper_case(16, rng=0, time_noise=0.0)
    times = case.problem.times.copy()
    times[2, 9] = 1.0  # one second: absurd
    mask = case.problem.mask.copy()
    mask[2, 9] = False
    problem = CalibrationProblem(case.problem.geometry, times, c=case.problem.c, mask=mask)

    step = PinvDelays()
    step.prepare(problem)
    d = step.estimate(case.positions)
    s = gauge_shift(d.e, case.delays.e)
    np.testing.assert_allclose(d.e, case.delays.e + s, rtol=0, atol=1e-18)


def test_disconnected_measurements_raise() -> None:
    case = paper_case(8, rng=0)
    halves = np.arange(8) < 4
    mask = halves[:, None] == halves[None, :]  # elements only hear their own half
    problem = CalibrationProblem(
        case.problem.geometry, case.problem.times, c=case.problem.c, mask=mask
    )
    with pytest.raises(ValueError, match="groups"):
        PinvDelays().prepare(problem)


# --- usage errors ----------------------------------------------------------------------


def test_estimate_before_prepare_raises() -> None:
    with pytest.raises(RuntimeError):
        PinvDelays().estimate(np.zeros((8, 2)))


def test_positions_of_wrong_shape_raise() -> None:
    case = paper_case(8, rng=0)
    step = PinvDelays()
    step.prepare(case.problem)
    with pytest.raises(ValueError):
        step.estimate(np.zeros((7, 2)))
