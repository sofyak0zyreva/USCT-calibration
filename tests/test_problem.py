import numpy as np
import pytest

from ringcal.geometry import RingGeometry
from ringcal.problem import CalibrationProblem, Delays

C_WATER = 1469.064  # m/s


@pytest.fixture
def ring8() -> RingGeometry:
    """A tiny ring: one arc of 8 elements, no gaps."""
    return RingGeometry(n_arcs=1, elements_per_arc=8, gap_slots=0, radius=0.11)


def random_times(n_tx: int, n_rx: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(1e-6, 150e-6, size=(n_tx, n_rx))


# --- defaults ------------------------------------------------------------------


def test_defaults_every_element_fires_and_listens(ring8: RingGeometry) -> None:
    p = CalibrationProblem(ring8, random_times(8, 8), c=C_WATER)
    assert p.n_tx == 8 and p.n_rx == 8
    assert np.array_equal(p.tx_elements, np.arange(8))
    assert np.array_equal(p.rx_elements, np.arange(8))


def test_default_mask_excludes_only_self_cells(ring8: RingGeometry) -> None:
    p = CalibrationProblem(ring8, random_times(8, 8), c=C_WATER)
    assert p.mask.dtype == bool
    assert np.array_equal(p.mask, ~np.eye(8, dtype=bool))


# --- custom transmitters and receivers ------------------------------------------


def test_every_second_element_fires(ring8: RingGeometry) -> None:
    tx = [0, 2, 4, 6]
    p = CalibrationProblem(ring8, random_times(4, 8), c=C_WATER, tx_elements=tx)
    assert p.times.shape == (4, 8)
    # self cells: row of element k, column of element k
    expected_self = [(0, 0), (1, 2), (2, 4), (3, 6)]
    assert [tuple(ij) for ij in np.argwhere(~p.mask)] == expected_self


def test_only_some_elements_listen(ring8: RingGeometry) -> None:
    p = CalibrationProblem(
        ring8, random_times(4, 4), c=C_WATER, tx_elements=[0, 2, 4, 6], rx_elements=[1, 3, 5, 7]
    )
    assert p.times.shape == (4, 4)
    assert p.mask.all()  # nobody hears itself


def test_row_and_column_order_follows_given_elements(ring8: RingGeometry) -> None:
    # Rows are in the order the transmitters were given, not sorted.
    p = CalibrationProblem(ring8, random_times(2, 8), c=C_WATER, tx_elements=[5, 1])
    assert np.array_equal(p.tx_elements, [5, 1])
    assert not p.mask[0, 5] and not p.mask[1, 1]


def test_active_elements_are_union_of_both_sides(ring8: RingGeometry) -> None:
    p = CalibrationProblem(
        ring8, random_times(2, 3), c=C_WATER, tx_elements=[0, 4], rx_elements=[1, 4, 6]
    )
    assert np.array_equal(p.active_elements, [0, 1, 4, 6])


# --- mask --------------------------------------------------------------------


def test_custom_mask_is_combined_with_self_cells(ring8: RingGeometry) -> None:
    mask = np.ones((8, 8), dtype=bool)
    mask[0, 3] = False
    p = CalibrationProblem(ring8, random_times(8, 8), c=C_WATER, mask=mask)
    assert not p.mask[0, 3]
    assert not p.mask[2, 2]  # self cell, even though the given mask said True
    assert p.mask.sum() == 64 - 8 - 1


def test_missing_times_allowed_where_masked(ring8: RingGeometry) -> None:
    times = random_times(8, 8)
    times[1, 5] = np.nan  # e.g. the arrival picker found nothing
    mask = np.ones((8, 8), dtype=bool)
    mask[1, 5] = False
    p = CalibrationProblem(ring8, times, c=C_WATER, mask=mask)
    assert not p.mask[1, 5]


def test_without_elements_masks_rows_and_columns(ring8: RingGeometry) -> None:
    tx = [0, 2, 4, 6]
    p = CalibrationProblem(ring8, random_times(4, 8), c=C_WATER, tx_elements=tx)
    q = p.without_elements([4, 5])  # 4 fires and listens, 5 only listens
    assert not q.mask[2, :].any()  # row of transmitter 4
    assert not q.mask[:, 4].any() and not q.mask[:, 5].any()
    assert q.mask[0, 1]  # unrelated cell untouched
    assert p.mask[0, 5]  # original problem unchanged


# --- immutability ------------------------------------------------------------


def test_arrays_are_read_only(ring8: RingGeometry) -> None:
    p = CalibrationProblem(ring8, random_times(8, 8), c=C_WATER)
    with pytest.raises(ValueError):
        p.times[0, 1] = 0.0
    with pytest.raises(ValueError):
        p.mask[0, 1] = False


def test_input_arrays_are_copied(ring8: RingGeometry) -> None:
    times = random_times(8, 8)
    p = CalibrationProblem(ring8, times, c=C_WATER)
    times[0, 1] = -1.0
    assert p.times[0, 1] != -1.0


# --- validation ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("shape", "kwargs"),
    [
        ((8, 7), {}),  # wrong number of columns
        ((3, 8), dict(tx_elements=[0, 2, 4, 6])),  # wrong number of rows
        ((1, 8), dict(tx_elements=[8])),  # transmitter not on the ring
        ((8, 1), dict(rx_elements=[-1])),  # receiver not on the ring
        ((2, 8), dict(tx_elements=[3, 3])),  # duplicate transmitter
        ((0, 8), dict(tx_elements=[])),  # nobody fires
        ((8, 8), dict(c=0.0)),  # speed of sound must be positive
        ((8, 8), dict(mask=np.ones((8, 7), dtype=bool))),  # mask of wrong shape
    ],
)
def test_invalid_input_raises(ring8: RingGeometry, shape: tuple, kwargs: dict) -> None:
    args = dict(c=C_WATER) | kwargs
    with pytest.raises(ValueError):
        CalibrationProblem(ring8, random_times(*shape), **args)


def test_non_finite_usable_time_raises(ring8: RingGeometry) -> None:
    times = random_times(8, 8)
    times[1, 5] = np.inf
    with pytest.raises(ValueError, match="finite"):
        CalibrationProblem(ring8, times, c=C_WATER)


# --- delays ------------------------------------------------------------------


def test_zero_delays_match_problem(ring8: RingGeometry) -> None:
    p = CalibrationProblem(ring8, random_times(4, 8), c=C_WATER, tx_elements=[0, 2, 4, 6])
    d = Delays.zeros(p)
    assert d.e.shape == (4,) and d.r.shape == (8,)
    assert not d.e.any() and not d.r.any()


def test_delays_must_be_one_dimensional() -> None:
    with pytest.raises(ValueError):
        Delays(e=np.zeros((2, 2)), r=np.zeros(4))
