import stim

from research_qec_haoran import __version__
from research_qec_haoran.paths import REPO_ROOT
from research_qec_haoran.surface_code import (
    D3_UNROTATED_DATA_COORDS,
    D3_UNROTATED_DATA_QUBITS,
    D3_UNROTATED_LOGICAL_X_COORDS,
    D3_UNROTATED_LOGICAL_X_LABEL,
    D3_UNROTATED_LOGICAL_X_QUBITS,
    D3_UNROTATED_LOGICAL_Z_COORDS,
    D3_UNROTATED_LOGICAL_Z_LABEL,
    D3_UNROTATED_LOGICAL_Z_QUBITS,
    D3_UNROTATED_MEASUREMENT_LABELS,
    D3_UNROTATED_NUM_QUBITS,
    D3_UNROTATED_STABILIZER_MEASUREMENT_LABELS,
    D3_UNROTATED_X_ANCILLAS,
    D3_UNROTATED_X_ANCILLA_COORDS,
    D3_UNROTATED_X_STABILIZERS,
    D3_UNROTATED_Z_ANCILLAS,
    D3_UNROTATED_Z_ANCILLA_COORDS,
    D3_UNROTATED_Z_STABILIZERS,
    append_d3_unrotated_data_readout,
    append_d3_unrotated_initial_reset,
    append_d3_unrotated_logical_readout,
    append_d3_unrotated_qubit_coords,
    append_d3_unrotated_stabilizer_round,
    append_d3_unrotated_stabilizer_rounds,
    build_d3_unrotated_base_circuit,
    get_d3_unrotated_measurement_labels,
)


def _strip_detector_instructions(circuit: stim.Circuit) -> str:
    return "\n".join(
        line
        for line in str(circuit).splitlines()
        if not line.startswith(("DETECTOR", "OBSERVABLE_INCLUDE", "SHIFT_COORDS"))
    )


def test_package_version_is_defined() -> None:
    assert __version__


def test_repo_root_name() -> None:
    assert REPO_ROOT.name == "research-qec-haoran"


def test_d3_unrotated_layout_sizes() -> None:
    assert len(D3_UNROTATED_DATA_COORDS) == 13
    assert D3_UNROTATED_LOGICAL_Z_COORDS == ((0, 0), (0, 2), (0, 4))
    assert D3_UNROTATED_LOGICAL_X_COORDS == ((0, 0), (2, 0), (4, 0))
    assert len(D3_UNROTATED_X_ANCILLA_COORDS) == 6
    assert len(D3_UNROTATED_Z_ANCILLA_COORDS) == 6
    assert len(D3_UNROTATED_X_STABILIZERS) == 6
    assert len(D3_UNROTATED_Z_STABILIZERS) == 6
    assert D3_UNROTATED_DATA_QUBITS == (0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24)
    assert D3_UNROTATED_X_ANCILLAS == (1, 3, 11, 13, 21, 23)
    assert D3_UNROTATED_Z_ANCILLAS == (5, 7, 9, 15, 17, 19)
    assert D3_UNROTATED_LOGICAL_Z_QUBITS == (0, 2, 4)
    assert D3_UNROTATED_LOGICAL_X_QUBITS == (0, 10, 20)


def test_d3_unrotated_single_round_circuit_sizes() -> None:
    circuit = build_d3_unrotated_base_circuit()
    append_d3_unrotated_stabilizer_rounds(circuit, rounds=1)
    assert D3_UNROTATED_NUM_QUBITS == 25
    assert circuit.num_qubits == 25
    assert circuit.num_measurements == 12
    assert len(D3_UNROTATED_MEASUREMENT_LABELS) == 12


def test_d3_unrotated_stim_measurement_order() -> None:
    assert D3_UNROTATED_MEASUREMENT_LABELS == D3_UNROTATED_STABILIZER_MEASUREMENT_LABELS
    assert D3_UNROTATED_MEASUREMENT_LABELS == (
        "X0",
        "X1",
        "Z0",
        "Z1",
        "Z2",
        "X2",
        "X3",
        "Z3",
        "Z4",
        "Z5",
        "X4",
        "X5",
    )
    assert get_d3_unrotated_measurement_labels("Z")[-1] == D3_UNROTATED_LOGICAL_Z_LABEL
    assert get_d3_unrotated_measurement_labels("X")[-1] == D3_UNROTATED_LOGICAL_X_LABEL


def test_d3_unrotated_base_circuit_has_only_layout() -> None:
    circuit = build_d3_unrotated_base_circuit()
    assert circuit.num_qubits == 25
    assert circuit.num_measurements == 0


def test_d3_unrotated_qubit_coords_can_be_appended_manually() -> None:
    circuit = stim.Circuit()
    append_d3_unrotated_qubit_coords(circuit)
    assert circuit.num_qubits == 25
    assert circuit.num_measurements == 0


def test_d3_unrotated_stabilizer_round_uses_measure_reset() -> None:
    circuit = build_d3_unrotated_base_circuit()
    append_d3_unrotated_stabilizer_round(circuit, reset_ancillas=False)
    circuit_text = str(circuit)
    assert "\nMR " in circuit_text
    assert "\nM " not in f"\n{circuit_text}"


def test_d3_unrotated_repeated_rounds_do_not_need_explicit_resets() -> None:
    circuit = build_d3_unrotated_base_circuit()
    append_d3_unrotated_stabilizer_rounds(circuit, rounds=2, reset_before_first_round=False)
    circuit_text = str(circuit)
    assert circuit_text.count("\nMR ") == 2
    assert "\nR " not in f"\n{circuit_text}"


def test_d3_unrotated_logical_readout_appends_one_measurement() -> None:
    z_circuit = build_d3_unrotated_base_circuit()
    append_d3_unrotated_initial_reset(z_circuit, "Z")
    append_d3_unrotated_stabilizer_rounds(z_circuit, rounds=1)
    append_d3_unrotated_logical_readout(z_circuit, "Z")
    x_circuit = build_d3_unrotated_base_circuit()
    append_d3_unrotated_initial_reset(x_circuit, "Z")
    append_d3_unrotated_stabilizer_rounds(x_circuit, rounds=1)
    append_d3_unrotated_logical_readout(x_circuit, "X")
    assert z_circuit.num_measurements == 13
    assert x_circuit.num_measurements == 13
    assert "MPP Z0*Z2*Z4" in str(z_circuit)
    assert "MPP X0*X10*X20" in str(x_circuit)


def test_d3_unrotated_logical_z_readout_is_deterministic() -> None:
    circuit = build_d3_unrotated_base_circuit()
    append_d3_unrotated_initial_reset(circuit, "Z")
    append_d3_unrotated_stabilizer_rounds(circuit, rounds=1)
    append_d3_unrotated_logical_readout(circuit, "Z")
    samples = circuit.compile_sampler().sample(shots=8)
    logical_z = samples[:, -1]
    assert logical_z.tolist() == [False] * 8


def test_d3_unrotated_z_memory_circuit_matches_stim_without_detectors() -> None:
    circuit = build_d3_unrotated_base_circuit()
    append_d3_unrotated_initial_reset(circuit, "Z")
    append_d3_unrotated_stabilizer_rounds(circuit, rounds=1)
    append_d3_unrotated_data_readout(circuit, "Z")

    stim_circuit = stim.Circuit.generated("surface_code:unrotated_memory_z", rounds=1, distance=3)
    assert str(circuit) == _strip_detector_instructions(stim_circuit)


def test_d3_unrotated_x_memory_circuit_matches_stim_without_detectors() -> None:
    circuit = build_d3_unrotated_base_circuit()
    append_d3_unrotated_initial_reset(circuit, "X")
    append_d3_unrotated_stabilizer_rounds(circuit, rounds=1)
    append_d3_unrotated_data_readout(circuit, "X")

    stim_circuit = stim.Circuit.generated("surface_code:unrotated_memory_x", rounds=1, distance=3)
    assert str(circuit) == _strip_detector_instructions(stim_circuit)
