from research_qec_haoran import __version__
from research_qec_haoran.paths import REPO_ROOT
from research_qec_haoran.surface_code import (
    D3_UNROTATED_DATA_COORDS,
    D3_UNROTATED_STIM_MEASUREMENT_LABELS,
    D3_UNROTATED_STIM_NUM_QUBITS,
    D3_UNROTATED_X_STABILIZERS,
    D3_UNROTATED_Z_STABILIZERS,
    build_d3_unrotated_logical_zero_init_circuit,
    build_d3_unrotated_logical_zero_init_stim_circuit,
)


def test_package_version_is_defined() -> None:
    assert __version__


def test_repo_root_name() -> None:
    assert REPO_ROOT.name == "research-qec-haoran"


def test_d3_unrotated_layout_sizes() -> None:
    assert len(D3_UNROTATED_DATA_COORDS) == 13
    assert len(D3_UNROTATED_X_STABILIZERS) == 6
    assert len(D3_UNROTATED_Z_STABILIZERS) == 6


def test_d3_unrotated_logical_zero_circuit_sizes() -> None:
    circuit = build_d3_unrotated_logical_zero_init_circuit(rounds=1)
    assert circuit.num_qubits == 25
    assert circuit.num_clbits == 12


def test_d3_unrotated_logical_zero_stim_circuit_sizes() -> None:
    circuit = build_d3_unrotated_logical_zero_init_stim_circuit(rounds=1)
    assert D3_UNROTATED_STIM_NUM_QUBITS == 25
    assert circuit.num_qubits == 25
    assert circuit.num_measurements == 12
    assert len(D3_UNROTATED_STIM_MEASUREMENT_LABELS) == 12
