import importlib

import pytest

import ringcal


def test_version() -> None:
    assert ringcal.__version__ == "0.1.0"


@pytest.mark.parametrize(
    "module",
    [
        "ringcal.geometry",
        "ringcal.problem",
        "ringcal.synthetic",
        "ringcal.solver",
        "ringcal.gauge",
        "ringcal.metrics",
        "ringcal.delays",
        "ringcal.positions",
    ],
)
def test_modules_import(module: str) -> None:
    importlib.import_module(module)
