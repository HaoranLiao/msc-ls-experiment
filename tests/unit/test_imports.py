from research_qec_haoran import __version__
from research_qec_haoran.paths import REPO_ROOT
from research_qec_haoran.surface_code import (
    D3_UNROTATED_DATA_COORDS,
    D3_UNROTATED_MEASUREMENT_LABELS,
    D3_UNROTATED_NUM_QUBITS,
    D3_UNROTATED_X_ANCILLA_COORDS,
    D3_UNROTATED_X_STABILIZERS,
    D3_UNROTATED_Z_ANCILLA_COORDS,
    D3_UNROTATED_Z_STABILIZERS,
    build_d3_unrotated_logical_zero_init_circuit,
)


def test_package_version_is_defined() -> None:
    assert __version__


def test_repo_root_name() -> None:
    assert REPO_ROOT.name == "research-qec-haoran"


def test_d3_unrotated_layout_sizes() -> None:
    assert len(D3_UNROTATED_DATA_COORDS) == 13
    assert len(D3_UNROTATED_X_ANCILLA_COORDS) == 6
    assert len(D3_UNROTATED_Z_ANCILLA_COORDS) == 6
    assert len(D3_UNROTATED_X_STABILIZERS) == 6
    assert len(D3_UNROTATED_Z_STABILIZERS) == 6


def test_d3_unrotated_logical_zero_stim_circuit_sizes() -> None:
    circuit = build_d3_unrotated_logical_zero_init_circuit(rounds=1)
    assert D3_UNROTATED_NUM_QUBITS == 25
    assert circuit.num_qubits == 25
    assert circuit.num_measurements == 12
    assert len(D3_UNROTATED_MEASUREMENT_LABELS) == 12


def test_d3_unrotated_stim_measurement_order() -> None:
    assert D3_UNROTATED_MEASUREMENT_LABELS[:6] == ("Z0", "Z1", "Z2", "Z3", "Z4", "Z5")
    assert D3_UNROTATED_MEASUREMENT_LABELS[6:] == ("X0", "X1", "X2", "X3", "X4", "X5")


def test_d3_unrotated_single_round_has_no_explicit_resets() -> None:
    circuit_text = str(build_d3_unrotated_logical_zero_init_circuit(rounds=1))
    assert "\nR " not in f"\n{circuit_text}"


def test_d3_unrotated_repeated_rounds_do_reset_ancillas() -> None:
    circuit_text = str(build_d3_unrotated_logical_zero_init_circuit(rounds=2))
    assert circuit_text.count("\nR ") == 2
