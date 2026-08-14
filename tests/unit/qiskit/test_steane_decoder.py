"""Qiskit integration tests for the Steane lookup decoder and Pauli frame."""

from __future__ import annotations

import math

import pytest
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator

from msc.decoding import PauliFrame, StabilizerFrame, SteaneDecoder
from msc.noise import append_pauli_faults
from msc.steane_code.decoding.decoder import (
    SYNDROME_BIT_LABELS,
    X_STABILIZER_LABELS,
    Z_STABILIZER_LABELS,
)
from msc.steane_code.state_injection_qiskit import (
    NUM_PATCH_QUBITS,
    PATCH_DATA_SITES,
    PATCH_DEFAULT_LOGICAL_X,
    PATCH_DEFAULT_LOGICAL_Z,
    InjectedStateName,
    append_patch_stabilizer_measurement,
    append_patch_steane_state_injection_encoder_crumble_sequences,
)
from msc.utils.normalize import parse_qiskit_bitstring

TOLERANCE = 1e-10
INVERSE_SQRT_TWO = 1.0 / math.sqrt(2)
# Expected (logical Z, logical X) values for each injected state.
EXPECTED_LOGICAL_EXPECTATIONS: dict[str, tuple[float, float]] = {
    "0": (1.0, 0.0),
    "T": (0.0, INVERSE_SQRT_TWO),
    "Ty": (INVERSE_SQRT_TWO, INVERSE_SQRT_TWO),
}
SINGLE_QUBIT_FAULTS = tuple(
    (pauli, data_site) for pauli in ("X", "Y", "Z") for data_site in PATCH_DATA_SITES
)
TWO_ROUND_FAULTS = (
    # Same site, same Pauli.
    (("X", 0), ("X", 0)),
    (("Z", 4), ("Z", 4)),
    (("Y", 9), ("Y", 9)),
    # Same site, different Pauli.
    (("X", 3), ("Z", 3)),
    (("Z", 10), ("X", 10)),
    (("X", 5), ("Y", 5)),
    # Different sites, same Pauli.
    (("X", 0), ("X", 8)),
    (("Z", 3), ("Z", 10)),
    (("Y", 4), ("Y", 9)),
    # Different sites, different Pauli.
    (("X", 0), ("Z", 9)),
    (("Z", 3), ("X", 5)),
    (("Y", 8), ("X", 10)),
    (("X", 4), ("Y", 0)),
    # Logical-support and off-support pairs.
    (("X", 8), ("X", 9)),
    (("Z", 9), ("Z", 10)),
    (("X", 8), ("Z", 10)),
    (("X", 0), ("X", 3)),
    (("Z", 4), ("Z", 5)),
)

_Z_LOGICAL = SparsePauliOp(
    "".join(
        "Z" if data_site in PATCH_DEFAULT_LOGICAL_Z else "I"
        for data_site in reversed(range(NUM_PATCH_QUBITS))
    )
)
_X_LOGICAL = SparsePauliOp(
    "".join(
        "X" if data_site in PATCH_DEFAULT_LOGICAL_X else "I"
        for data_site in reversed(range(NUM_PATCH_QUBITS))
    )
)
# Patch indices select the data subsystem from the data-plus-ancilla statevector.
_PATCH_QUBIT_INDICES = list(range(NUM_PATCH_QUBITS))


def _parse_syndrome_round(
    syndrome_counts_key: str,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return raw X-check and Z-check outcomes from one counts register."""
    syndrome_bits = parse_qiskit_bitstring(syndrome_counts_key, SYNDROME_BIT_LABELS)
    return (
        tuple(syndrome_bits[label] for label in SYNDROME_BIT_LABELS[:3]),
        tuple(syndrome_bits[label] for label in SYNDROME_BIT_LABELS[3:]),
    )


# ===========================================================================
# One-round decode: fault → syndrome → decode → expectation check
# ===========================================================================


@pytest.mark.parametrize("injected_state", tuple(EXPECTED_LOGICAL_EXPECTATIONS))
@pytest.mark.parametrize(
    "fault",
    SINGLE_QUBIT_FAULTS,
    ids=tuple(f"{pauli}_{site}" for pauli, site in SINGLE_QUBIT_FAULTS),
)
def test_one_round_logical_expectations(
    injected_state: InjectedStateName,
    fault: tuple[str, int],
) -> None:
    # Inject the state, apply one data fault, and measure its syndrome.
    patch = QuantumRegister(NUM_PATCH_QUBITS, "patch")
    syndrome = ClassicalRegister(6, "syndrome")
    circuit = QuantumCircuit(patch, syndrome)
    patch_qubits = list(patch)
    append_patch_steane_state_injection_encoder_crumble_sequences(
        circuit,
        patch_qubits=patch_qubits,
        injected_state=injected_state,
        add_barriers=False,
    )
    append_pauli_faults(circuit, [fault], qubit_map=patch_qubits)
    append_patch_stabilizer_measurement(
        circuit,
        patch_qubits=patch_qubits,
        classical_bits=list(syndrome),
        add_barriers=False,
    )
    circuit.save_statevector(label="statevector")
    result = AerSimulator(method="statevector").run(circuit, shots=1).result()
    syndrome_counts_key = next(iter(result.get_counts()))
    statevector = result.data()["statevector"]

    # Decode the syndrome into the logical Pauli frame.
    x_raw, z_raw = _parse_syndrome_round(syndrome_counts_key)
    stabilizer_frame = StabilizerFrame()
    stabilizer_frame.record_many(dict.fromkeys((*X_STABILIZER_LABELS, *Z_STABILIZER_LABELS), 0))
    pauli_frame = PauliFrame()
    pauli_frame.register("L0")
    SteaneDecoder().decode_raw_syndromes(
        pauli_frame=pauli_frame,
        logical_qubit_label="L0",
        x_raw=x_raw,
        z_raw=z_raw,
        stab_frame=stabilizer_frame,
    )

    # Verify the frame-corrected logical expectations.
    raw_logical_z = statevector.expectation_value(_Z_LOGICAL, _PATCH_QUBIT_INDICES).real
    raw_logical_x = statevector.expectation_value(_X_LOGICAL, _PATCH_QUBIT_INDICES).real
    corrected_logical_z = pauli_frame.correct_z_expectation("L0", raw_logical_z)
    corrected_logical_x = pauli_frame.correct_x_expectation("L0", raw_logical_x)
    expected_logical_z, expected_logical_x = EXPECTED_LOGICAL_EXPECTATIONS[injected_state]
    assert corrected_logical_z == pytest.approx(expected_logical_z, abs=TOLERANCE), (
        f"state {injected_state!r}, fault {fault}: logical Z = "
        f"{corrected_logical_z:.6f}, expected {expected_logical_z:.6f}"
    )
    assert corrected_logical_x == pytest.approx(expected_logical_x, abs=TOLERANCE), (
        f"state {injected_state!r}, fault {fault}: logical X = "
        f"{corrected_logical_x:.6f}, expected {expected_logical_x:.6f}"
    )


# ===========================================================================
# Two-round decode: fault → syndrome → decode → fault → syndrome → decode
# ===========================================================================


@pytest.mark.parametrize("injected_state", tuple(EXPECTED_LOGICAL_EXPECTATIONS))
@pytest.mark.parametrize(
    ("first_fault", "second_fault"),
    TWO_ROUND_FAULTS,
    ids=tuple(
        f"{first_pauli}{first_site}-{second_pauli}{second_site}"
        for (first_pauli, first_site), (second_pauli, second_site) in TWO_ROUND_FAULTS
    ),
)
def test_two_round_logical_expectations(
    injected_state: InjectedStateName,
    first_fault: tuple[str, int],
    second_fault: tuple[str, int],
) -> None:
    # Inject the state and measure after each of two data faults.
    patch = QuantumRegister(NUM_PATCH_QUBITS, "patch")
    first_syndrome = ClassicalRegister(6, "first_syndrome")
    second_syndrome = ClassicalRegister(6, "second_syndrome")
    circuit = QuantumCircuit(patch, first_syndrome, second_syndrome)
    patch_qubits = list(patch)
    append_patch_steane_state_injection_encoder_crumble_sequences(
        circuit,
        patch_qubits=patch_qubits,
        injected_state=injected_state,
        add_barriers=False,
    )
    append_pauli_faults(circuit, [first_fault], qubit_map=patch_qubits)
    append_patch_stabilizer_measurement(
        circuit,
        patch_qubits=patch_qubits,
        classical_bits=list(first_syndrome),
        add_barriers=False,
    )
    append_pauli_faults(circuit, [second_fault], qubit_map=patch_qubits)
    append_patch_stabilizer_measurement(
        circuit,
        patch_qubits=patch_qubits,
        classical_bits=list(second_syndrome),
        add_barriers=False,
    )
    circuit.save_statevector(label="statevector")
    result = AerSimulator(method="statevector").run(circuit, shots=1).result()
    syndrome_counts_key = next(iter(result.get_counts())).replace(" ", "")
    statevector = result.data()["statevector"]
    assert len(syndrome_counts_key) == 12

    # Qiskit prints the most recently added classical register first.
    second_syndrome_counts_key = syndrome_counts_key[:6]
    first_syndrome_counts_key = syndrome_counts_key[6:]

    # Decode both rounds; the stabilizer frame converts raw values into changes.
    stabilizer_frame = StabilizerFrame()
    stabilizer_frame.record_many(dict.fromkeys((*X_STABILIZER_LABELS, *Z_STABILIZER_LABELS), 0))
    pauli_frame = PauliFrame()
    pauli_frame.register("L0")
    decoder = SteaneDecoder()
    first_x_raw, first_z_raw = _parse_syndrome_round(first_syndrome_counts_key)
    decoder.decode_raw_syndromes(
        pauli_frame=pauli_frame,
        logical_qubit_label="L0",
        x_raw=first_x_raw,
        z_raw=first_z_raw,
        stab_frame=stabilizer_frame,
    )
    second_x_raw, second_z_raw = _parse_syndrome_round(second_syndrome_counts_key)
    decoder.decode_raw_syndromes(
        pauli_frame=pauli_frame,
        logical_qubit_label="L0",
        x_raw=second_x_raw,
        z_raw=second_z_raw,
        stab_frame=stabilizer_frame,
    )

    # Verify the accumulated frame against the final logical state.
    raw_logical_z = statevector.expectation_value(_Z_LOGICAL, _PATCH_QUBIT_INDICES).real
    raw_logical_x = statevector.expectation_value(_X_LOGICAL, _PATCH_QUBIT_INDICES).real
    corrected_logical_z = pauli_frame.correct_z_expectation("L0", raw_logical_z)
    corrected_logical_x = pauli_frame.correct_x_expectation("L0", raw_logical_x)
    expected_logical_z, expected_logical_x = EXPECTED_LOGICAL_EXPECTATIONS[injected_state]
    fault_label = f"state {injected_state!r}, faults {first_fault} then {second_fault}"
    assert corrected_logical_z == pytest.approx(expected_logical_z, abs=TOLERANCE), (
        f"{fault_label}: logical Z = {corrected_logical_z:.6f}, "
        f"expected {expected_logical_z:.6f}"
    )
    assert corrected_logical_x == pytest.approx(expected_logical_x, abs=TOLERANCE), (
        f"{fault_label}: logical X = {corrected_logical_x:.6f}, "
        f"expected {expected_logical_x:.6f}"
    )
