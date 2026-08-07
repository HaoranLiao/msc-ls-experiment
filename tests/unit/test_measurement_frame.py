"""Tests for reset-free measurement-qubit state tracking."""

from __future__ import annotations

import pytest

from msc.decoding import MeasurementFrame


def test_first_measurement_starts_from_zero() -> None:
    frame = MeasurementFrame()

    assert frame.advance("ancilla", raw_bit=1) == 1
    assert frame.current("ancilla") == 1


def test_advance_removes_residual_bit_from_next_readout() -> None:
    frame = MeasurementFrame()

    first_outcome = frame.advance("ancilla", raw_bit=1)
    second_outcome = frame.advance("ancilla", raw_bit=0)

    assert (first_outcome, second_outcome) == (1, 1)
    assert frame.current("ancilla") == 0


def test_frame_decodes_repeated_no_reset_parity_measurements() -> None:
    frame = MeasurementFrame()

    # Intended parities 1, 1, 0 give cumulative raw bits 1, 0, 0 when
    # the measurement ancilla is reused without reset.
    interpreted = [frame.advance("merged_boundary_ancilla", raw_bit) for raw_bit in (1, 0, 0)]

    assert interpreted == [1, 1, 0]


def test_measurement_qubits_have_independent_residual_states() -> None:
    frame = MeasurementFrame()
    frame.advance("ancilla_a", raw_bit=1)

    assert frame.advance("ancilla_a", raw_bit=0) == 1
    assert frame.advance("ancilla_b", raw_bit=0) == 0


def test_reset_clears_residual_measurement_state() -> None:
    frame = MeasurementFrame()
    frame.advance("ancilla", raw_bit=1)

    frame.reset("ancilla")

    assert frame.current("ancilla") == 0
    assert frame.advance("ancilla", raw_bit=1) == 1
    assert frame.as_dict() == {"ancilla": 1}


def test_invalid_labels_and_bits_are_rejected() -> None:
    frame = MeasurementFrame()

    with pytest.raises(ValueError, match="non-empty string"):
        frame.advance("", raw_bit=1)
    with pytest.raises(ValueError, match="bit must be 0 or 1"):
        frame.advance("ancilla", raw_bit=2)
