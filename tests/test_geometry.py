import dataclasses

import numpy as np
import pytest

from ringcal.geometry import RingGeometry

# The lab's ring: 2112 slots, 8 arcs of 256 elements, gaps of 8 empty slots,
# diameter about 0.22 m.
LAB = dict(n_arcs=8, elements_per_arc=256, gap_slots=8, radius=0.11)


@pytest.fixture
def lab() -> RingGeometry:
    return RingGeometry(**LAB)


# --- counts and slots -------------------------------------------------------


def test_lab_counts(lab: RingGeometry) -> None:
    assert lab.n_elements == 2048
    assert lab.n_slots == 2112


def test_element_slots_follow_arcs_and_gaps(lab: RingGeometry) -> None:
    slots = lab.element_slots
    assert slots.shape == (2048,)
    assert slots[0] == 0
    assert slots[255] == 255  # last element of arc 0
    assert slots[256] == 264  # first element of arc 1, after 8 empty slots
    assert slots[-1] == 7 * 264 + 255


def test_empty_slots_form_eight_gaps_of_eight(lab: RingGeometry) -> None:
    empty = np.setdiff1d(np.arange(lab.n_slots), lab.element_slots)
    assert empty.size == 64
    gaps = np.split(empty, np.where(np.diff(empty) > 1)[0] + 1)
    assert len(gaps) == 8
    assert all(g.size == 8 for g in gaps)


def test_first_slot_rotates_numbering(lab: RingGeometry) -> None:
    shifted = dataclasses.replace(lab, first_slot=8)
    assert shifted.element_slots[0] == 8
    assert np.array_equal(shifted.element_slots, (lab.element_slots + 8) % lab.n_slots)


def test_arc_of_element(lab: RingGeometry) -> None:
    arcs = lab.element_arcs
    assert arcs[0] == 0 and arcs[255] == 0
    assert arcs[256] == 1 and arcs[-1] == 7
    assert np.array_equal(np.bincount(arcs), np.full(8, 256))


# --- nominal positions -------------------------------------------------------


def test_nominal_positions_lie_on_the_circle(lab: RingGeometry) -> None:
    x = lab.nominal_positions()
    assert x.shape == (2048, 2)
    np.testing.assert_allclose(np.linalg.norm(x, axis=1), 0.11, rtol=1e-12)


def test_neighbour_spacing_within_arc_and_across_gap(lab: RingGeometry) -> None:
    angles = lab.element_angles
    step = np.diff(angles)
    one_slot = 2 * np.pi / 2112
    np.testing.assert_allclose(step[0], one_slot)  # inside arc 0
    np.testing.assert_allclose(step[255], 9 * one_slot)  # across the first gap


def test_angle_offset_rotates_positions(lab: RingGeometry) -> None:
    rotated = dataclasses.replace(lab, angle_offset=np.pi / 2)
    x, y = lab.nominal_positions(), rotated.nominal_positions()
    # rotating (a, b) by 90 degrees gives (-b, a)
    np.testing.assert_allclose(y, np.column_stack([-x[:, 1], x[:, 0]]), atol=1e-15)


def test_single_arc_without_gaps_is_a_regular_polygon() -> None:
    ring = RingGeometry(n_arcs=1, elements_per_arc=16, gap_slots=0, radius=1.0)
    x = ring.nominal_positions()
    sides = np.linalg.norm(np.roll(x, -1, axis=0) - x, axis=1)
    np.testing.assert_allclose(sides, 2 * np.sin(np.pi / 16))


# --- validation ----------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(n_arcs=0),
        dict(elements_per_arc=0),
        dict(gap_slots=-1),
        dict(radius=0.0),
        dict(first_slot=2112),
    ],
)
def test_invalid_parameters_raise(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        RingGeometry(**{**LAB, **kwargs})


def test_geometry_is_immutable(lab: RingGeometry) -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        lab.radius = 0.2  # type: ignore[misc]
