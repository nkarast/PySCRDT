"""
Smoke tests for PySCRDT.

These tests cover instantiation and configuration of the class without
requiring any Twiss input files or running the heavy sympy calculations.
"""

import logging

import pytest
from PySCRDT import PySCRDT, __version__


# ---------------------------------------------------------------------------
# Package metadata
# ---------------------------------------------------------------------------

def test_version():
    assert __version__ == "1.1.0"


def test_import_class():
    assert PySCRDT is not None


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------

def test_instantiation_no_args(caplog):
    """Class should construct without arguments, emitting guidance log messages."""
    with caplog.at_level(logging.INFO, logger="PySCRDT._core"):
        obj = PySCRDT()
    messages = " ".join(r.message for r in caplog.records)
    assert "setParameters" in messages or "readParameters" in messages
    assert obj.parameters is None
    assert obj.data is None
    assert obj.mode is None


def test_instantiation_with_parameters(caplog):
    with caplog.at_level(logging.INFO, logger="PySCRDT._core"):
        obj = PySCRDT(parameters=True)
    assert obj.parameters is not None
    messages = " ".join(r.message for r in caplog.records)
    assert "prepareData" in messages or "loadTwissFromXsuite" in messages


# ---------------------------------------------------------------------------
# setParameters / getParameters
# ---------------------------------------------------------------------------

def test_set_parameters_defaults():
    obj = PySCRDT()
    obj.setParameters()
    p = obj.getParameters()
    assert p["intensity"] == pytest.approx(41e10)
    assert p["bunchLength"] == pytest.approx(5.96)
    assert p["ro"] == pytest.approx(1.5347e-18)
    assert p["emittance_x"] == pytest.approx(2e-6)
    assert p["emittance_y"] == pytest.approx(1.1e-6)
    assert p["dpp_rms"] == pytest.approx(0.5e-3)
    assert p["dpp"] == pytest.approx(0.0)
    assert p["bF"] is None
    assert p["harmonic"] == 1


def test_set_parameters_custom():
    obj = PySCRDT()
    obj.setParameters(intensity=10e10, dpp=1e-3)
    p = obj.getParameters()
    assert p["intensity"] == pytest.approx(10e10)
    assert p["dpp"] == pytest.approx(1e-3)


def test_get_parameters_raises_when_unset():
    obj = PySCRDT()
    with pytest.raises(ValueError, match="setParameters"):
        obj.getParameters()


# ---------------------------------------------------------------------------
# setMode / getMode
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", [3, 5])
def test_set_mode_valid(mode):
    obj = PySCRDT()
    obj.setMode(mode)
    assert obj.getMode() == mode


def test_set_mode_invalid():
    obj = PySCRDT()
    with pytest.raises(ValueError):
        obj.setMode(4)


def test_get_mode_raises_when_unset():
    obj = PySCRDT()
    with pytest.raises(ValueError, match="setMode"):
        obj.getMode()


# ---------------------------------------------------------------------------
# setOrder / getOrder
# ---------------------------------------------------------------------------

def test_set_order_mode3():
    obj = PySCRDT()
    obj.setMode(3)
    obj.setOrder([2, 0, 1])
    assert obj.getOrder() == (2, 0, 1)


def test_set_order_mode3_any_harmonic():
    obj = PySCRDT()
    obj.setMode(3)
    obj.setOrder([2, 2, "any"])
    assert obj.getOrder() == (2, 2, "any")


def test_set_order_mode5():
    obj = PySCRDT()
    obj.setMode(5)
    obj.setOrder([1, 1, 0, 0, 2])
    assert obj.getOrder() == (1, 1, 0, 0, 2)


def test_set_order_wrong_type_raises():
    obj = PySCRDT()
    obj.setMode(3)
    with pytest.raises(TypeError):
        obj.setOrder([2.0, 0, 1])  # float not allowed for m


def test_set_order_wrong_harmonic_raises():
    obj = PySCRDT()
    obj.setMode(3)
    with pytest.raises(TypeError):
        obj.setOrder([2, 0, "notallowed"])


def test_set_order_mode_mismatch_raises():
    obj = PySCRDT()
    obj.setMode(5)
    with pytest.raises(ValueError):
        obj.setOrder([2, 0, 1])  # length 3 forbidden in mode 5


# ---------------------------------------------------------------------------
# updateParameters  (tests the critical Python-3 fix: .items() not .iteritems())
# ---------------------------------------------------------------------------

def test_update_parameters():
    obj = PySCRDT()
    obj.setParameters()
    obj.updateParameters(intensity=5e10)
    assert obj.parameters["intensity"] == pytest.approx(5e10)


def test_update_parameters_unknown_key_raises():
    obj = PySCRDT()
    obj.setParameters()
    with pytest.raises(ValueError, match="unknown parameter"):
        obj.updateParameters(nonexistent_key=42)


def test_update_parameters_no_params_raises():
    obj = PySCRDT()
    with pytest.raises(ValueError, match="setParameters"):
        obj.updateParameters(intensity=1e10)


# ---------------------------------------------------------------------------
# checkWriting
# ---------------------------------------------------------------------------

def test_check_writing():
    obj = PySCRDT()
    cw = obj.checkWriting()
    assert "Set & Update" in cw
    assert "Update only" in cw
    assert "intensity" in cw["Set & Update"]
    assert "b" in cw["Update only"]
