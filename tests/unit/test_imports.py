import pytest
import stim

from research_qec_haoran import __version__
from research_qec_haoran.paths import REPO_ROOT
from research_qec_haoran.surface_code import (
    D3_UNROTATED_DATA_COORDS,
    D3_UNROTATED_X_STABILIZERS,
    D3_UNROTATED_Z_STABILIZERS,
    generate_d3_unrotated_memory_circuit,
)


def test_package_version_is_defined() -> None:
    assert __version__


def test_repo_root_name() -> None:
    assert REPO_ROOT.name == "research-qec-haoran"


def test_d3_unrotated_layout_shapes() -> None:
    assert len(D3_UNROTATED_DATA_COORDS) == 13
    assert len(D3_UNROTATED_X_STABILIZERS) == 6
    assert len(D3_UNROTATED_Z_STABILIZERS) == 6
    assert D3_UNROTATED_X_STABILIZERS[0].name == "X0"
    assert D3_UNROTATED_Z_STABILIZERS[0].name == "Z0"


def test_generate_d3_unrotated_memory_z_matches_stim() -> None:
    circuit = generate_d3_unrotated_memory_circuit(rounds=2, basis="Z")
    expected = stim.Circuit.generated("surface_code:unrotated_memory_z", distance=3, rounds=2)
    assert circuit == expected
    assert circuit.num_qubits == 25
    assert circuit.num_detectors == 24
    assert circuit.num_observables == 1


def test_generate_d3_unrotated_memory_x_matches_stim() -> None:
    circuit = generate_d3_unrotated_memory_circuit(rounds=2, basis="X")
    expected = stim.Circuit.generated("surface_code:unrotated_memory_x", distance=3, rounds=2)
    assert circuit == expected
    assert circuit.num_qubits == 25
    assert circuit.num_detectors == 24
    assert circuit.num_observables == 1


def test_generate_d3_unrotated_memory_circuit_rejects_bad_args() -> None:
    with pytest.raises(ValueError, match="rounds must be at least 1"):
        generate_d3_unrotated_memory_circuit(rounds=0)

    with pytest.raises(ValueError, match="basis must be 'X' or 'Z'"):
        generate_d3_unrotated_memory_circuit(basis="Y")
