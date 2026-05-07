"""Tests for Steane-code magic-state cultivation helpers."""

from __future__ import annotations

import math

import numpy as np
import pytest

from research_qec_haoran.steane_code import (
    CULTIVATION_FLAG_QUBITS,
    CULTIVATION_ROOT_QUBITS,
    STEANE_DATA_QUBITS,
    build_cultivation_projection_circuit,
    build_cultivation_stage_circuits,
    build_noisy_t_state_injection_circuit,
    build_noisy_t_state_then_cultivation_circuit,
    build_t_state_injection_circuit,
    build_t_state_injection_stage_circuits,
    build_t_state_then_cultivation_circuit,
    encoded_t_statevector,
    encoded_theta_statevector,
    logical_bloch_components,
    noisy_encoded_t_density_matrix,
    project_onto_magic_axis,
)


def test_encoded_t_state_is_on_magic_axis() -> None:
    bloch = logical_bloch_components(encoded_t_statevector())

    assert bloch["X"] == pytest.approx(1 / math.sqrt(2))
    assert bloch["Y"] == pytest.approx(0.0)
    assert bloch["Z"] == pytest.approx(1 / math.sqrt(2))


def test_magic_axis_projection_accepts_t_state_with_probability_one() -> None:
    result = project_onto_magic_axis(encoded_t_statevector())

    assert result.acceptance_probability == pytest.approx(1.0)
    np.testing.assert_allclose(result.state.data, encoded_t_statevector().data, atol=1e-12)


def test_magic_axis_projection_rejects_orthogonal_state() -> None:
    with pytest.raises(ValueError, match="zero probability"):
        project_onto_magic_axis(encoded_theta_statevector(5 * math.pi / 4))


def test_magic_axis_projection_probability_for_logical_zero() -> None:
    result = project_onto_magic_axis(encoded_theta_statevector(0.0))

    expected = (1 + 1 / math.sqrt(2)) / 2
    assert result.acceptance_probability == pytest.approx(expected)
    assert logical_bloch_components(result.state)["X"] == pytest.approx(1 / math.sqrt(2))
    assert logical_bloch_components(result.state)["Z"] == pytest.approx(1 / math.sqrt(2))


def test_noisy_encoded_t_density_matrix_is_normalized() -> None:
    rho = noisy_encoded_t_density_matrix(0.01)

    assert np.trace(rho.data).real == pytest.approx(1.0)


def test_injection_circuit_uses_paper_t_state_prep_and_noise_channel() -> None:
    circuit = build_noisy_t_state_injection_circuit(injection_error_probability=1e-3)
    operation_names = [instruction.operation.name for instruction in circuit.data]

    assert circuit.num_qubits == 11
    assert circuit.num_clbits == 4
    assert "ry" in operation_names
    assert "kraus" in operation_names
    assert circuit.count_ops()["measure"] == 4


def test_ideal_injection_wrapper_has_no_noise_channel() -> None:
    circuit = build_t_state_injection_circuit()
    operation_names = [instruction.operation.name for instruction in circuit.data]

    assert "ry" in operation_names
    assert "cx" in operation_names
    assert "cz" not in operation_names
    assert "kraus" not in operation_names
    assert circuit.count_ops()["h"] == 31
    assert circuit.count_ops()["cx"] == 24


def test_stage_builders_return_ordered_cumulative_circuits() -> None:
    injection_stages = build_t_state_injection_stage_circuits()
    cultivation_stages = build_cultivation_stage_circuits()

    assert [label for label, _ in injection_stages] == [
        "1_physical_t_input",
        "2_steane_encoding_network",
        "3_injection_detector_readout",
    ]
    assert [label for label, _ in cultivation_stages] == [
        "1_apply_transversal_paper_t",
        "2_forward_fold_to_roots",
        "3_root_measurement",
        "4_time_reversed_fold_and_paper_t_dagger",
        "5_flag_measurements",
    ]
    assert injection_stages[-1][1].count_ops()["measure"] == 4
    assert cultivation_stages[-1][1].count_ops()["measure"] == 6


def test_cultivation_projection_circuit_shape_and_measurements() -> None:
    circuit = build_cultivation_projection_circuit()

    assert circuit.num_qubits == 12
    assert circuit.num_clbits == len(CULTIVATION_ROOT_QUBITS) + len(CULTIVATION_FLAG_QUBITS)
    assert circuit.count_ops()["measure"] == 6
    assert circuit.count_ops()["reset"] == 6


def test_combined_circuit_stops_after_cultivation_projection() -> None:
    circuit = build_t_state_then_cultivation_circuit()

    assert circuit.num_qubits == 12
    assert circuit.num_clbits == 10
    assert circuit.count_ops()["measure"] == 10
    assert STEANE_DATA_QUBITS == (0, 1, 3, 5, 7, 8, 10)
    assert circuit.data[-1].operation.name == "measure"


def test_noisy_combined_circuit_can_still_be_built() -> None:
    circuit = build_noisy_t_state_then_cultivation_circuit(injection_error_probability=1e-3)

    assert "kraus" in [instruction.operation.name for instruction in circuit.data]


def test_probability_validation() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        build_noisy_t_state_injection_circuit(injection_error_probability=-0.1)

    with pytest.raises(ValueError, match="between 0 and 1"):
        noisy_encoded_t_density_matrix(1.1)
