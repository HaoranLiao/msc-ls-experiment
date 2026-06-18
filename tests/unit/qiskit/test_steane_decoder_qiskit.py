"""Tests for the Steane-code lookup-table decoder on the 11-site patch.

Each test injects a single-qubit Pauli fault after state injection, measures
all 6 stabilizers (1 round), decodes via the lookup table, applies a Pauli-frame
correction, and verifies the corrected destructive logical readout.
"""

from __future__ import annotations

import pytest
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit_aer import AerSimulator

from research_qec_haoran.decoding import (
    PauliFrame,
    StabilizerFrame,
    decode_steane_syndrome,
    STEANE_PATCH_X_STABILIZER_LABELS,
    STEANE_PATCH_Z_STABILIZER_LABELS,
)
from research_qec_haoran.steane_code.state_injection_qiskit import (
    NUM_PATCH_QUBITS,
    PATCH_DATA_SITES,
    PATCH_DEFAULT_LOGICAL_X,
    PATCH_DEFAULT_LOGICAL_Z,
    InjectedStateName,
    append_patch_logical_measurement,
    append_patch_stabilizer_measurement,
    append_patch_steane_state_injection_encoder,
)
from research_qec_haoran.noise import append_patch_pauli_faults


# States for which we test logical Z readout (eigenvalue is deterministic).
_Z_EIGENSTATES: list[tuple[InjectedStateName, int]] = [
    ("0", 0),   # |0_L⟩ → Z_L = +1 → readout bit 0
    ("1", 1),   # |1_L⟩ → Z_L = -1 → readout bit 1
]

# States for which we test logical X readout (eigenvalue is deterministic).
_X_EIGENSTATES: list[tuple[InjectedStateName, int]] = [
    ("+", 0),   # |+_L⟩ → X_L = +1 → readout bit 0
    ("-", 1),   # |-_L⟩ → X_L = -1 → readout bit 1
]

_ALL_SINGLE_QUBIT_FAULTS: list[tuple[str, int]] = [
    (pauli, site) for pauli in ("X", "Y", "Z") for site in PATCH_DATA_SITES
]


# ===========================================================================
# One-round decode: fault → syndrome → decode
# ===========================================================================


def _build_patch_inject_fault_and_measure(
    injected_state: InjectedStateName,
    fault_pauli: str,
    fault_site: int,
    logical_basis: str,
) -> QuantumCircuit:
    """Build: state-injection → fault → stabilizer measurement → logical measurement.

    Returns a circuit with registers:
      - patch (11 qubits): 11-site patch data+helper
      - syn_anc (6 qubits): syndrome ancillas
      - syn_cl (6 bits): syndrome measurement results
      - log_cl (3 bits): destructive logical measurement results (parity of 3 qubits)
    """
    patch = QuantumRegister(NUM_PATCH_QUBITS, "patch")
    syn_anc = QuantumRegister(6, "syn_anc")
    syn_cl = ClassicalRegister(6, "syn")
    log_cl = ClassicalRegister(3, "log")
    circuit = QuantumCircuit(patch, syn_anc, syn_cl, log_cl)

    # 1. State injection on the 11-site patch.
    append_patch_steane_state_injection_encoder(
        circuit,
        patch_qubits=list(patch),
        injected_state=injected_state,
        add_barriers=False,
    )

    # 2. Inject a single-qubit fault.
    append_patch_pauli_faults(circuit, [(fault_pauli, fault_site)], patch_qubits=list(patch))

    # 3. Measure all 6 stabilizers (1 round).
    append_patch_stabilizer_measurement(
        circuit,
        patch_qubits=list(patch),
        ancilla_qubits=list(syn_anc),
        classical_bits=list(syn_cl),
        add_barriers=False,
    )

    # 4. Destructive logical measurement using default support.
    support = PATCH_DEFAULT_LOGICAL_Z if logical_basis == "Z" else PATCH_DEFAULT_LOGICAL_X
    append_patch_logical_measurement(
        circuit,
        logical_basis,
        patch_qubits=list(patch),
        support=support,
        classical_bits=list(log_cl),
        destructive=True,
    )

    return circuit


def _run_and_decode(
    circuit: QuantumCircuit,
    logical_basis: str,
) -> int:
    """Run the circuit (1 shot, noiseless), decode, and return the corrected logical bit."""
    sim = AerSimulator()
    result = sim.run(circuit, shots=1).result()
    bitstring = next(iter(result.get_counts())).replace(" ", "")

    # Parse the bitstring: Qiskit returns registers in reverse declaration order.
    # Register order in bitstring (MSB to LSB): log_cl (3) | syn_cl (6)
    # Total classical bits = 6 + 3 = 9.
    n_bits = len(bitstring)
    assert n_bits == 9, f"Expected 9 classical bits, got {n_bits}"

    # Qiskit big-endian: leftmost = highest-index register declared last.
    # Declaration order: syn_cl (6), log_cl (3)
    # Bitstring layout: log_cl[2] log_cl[1] log_cl[0] syn_cl[5] ... syn_cl[0]
    # → log bits are the first 3 chars, syn bits are the last 6 chars.
    log_bits_str = bitstring[:3]
    syn_bits_str = bitstring[3:]

    # Parse syndrome bits (index 0 = rightmost char in syn_bits_str).
    syn_bits = [int(b) for b in reversed(syn_bits_str)]
    # syn_bits[0..2] = X-stabilizer outcomes, syn_bits[3..5] = Z-stabilizer outcomes.
    x_raw = tuple(syn_bits[0:3])
    z_raw = tuple(syn_bits[3:6])

    # Parse logical measurement bits (parity of 3 data qubits).
    log_bits = [int(b) for b in reversed(log_bits_str)]
    raw_logical_parity = log_bits[0] ^ log_bits[1] ^ log_bits[2]

    # Initialize stabilizer frame to all +1 (state injection preserves codespace).
    stab_frame = StabilizerFrame()
    for label in STEANE_PATCH_X_STABILIZER_LABELS:
        stab_frame.register(label, 0)
    for label in STEANE_PATCH_Z_STABILIZER_LABELS:
        stab_frame.register(label, 0)

    # Compute change-detected syndrome.
    x_measurements = {
        label: x_raw[i] for i, label in enumerate(STEANE_PATCH_X_STABILIZER_LABELS)
    }
    z_measurements = {
        label: z_raw[i] for i, label in enumerate(STEANE_PATCH_Z_STABILIZER_LABELS)
    }
    x_deltas = stab_frame.advance(x_measurements)
    z_deltas = stab_frame.advance(z_measurements)

    x_syndrome = tuple(x_deltas[label] for label in STEANE_PATCH_X_STABILIZER_LABELS)
    z_syndrome = tuple(z_deltas[label] for label in STEANE_PATCH_Z_STABILIZER_LABELS)

    # Decode.
    recovery = decode_steane_syndrome(z_syndrome=z_syndrome, x_syndrome=x_syndrome)

    # Update Pauli frame.
    pauli_frame = PauliFrame()
    pauli_frame.register("L0")
    pauli_frame.update_x("L0", flip=recovery.flips_logical_z)
    pauli_frame.update_z("L0", flip=recovery.flips_logical_x)

    # Interpret the logical readout through the Pauli frame.
    if logical_basis == "Z":
        corrected = pauli_frame.interpret_z_readout("L0", raw_logical_parity)
    else:
        corrected = pauli_frame.interpret_x_readout("L0", raw_logical_parity)

    return corrected


class TestOneRoundDecodeCorrect:
    """Single correctable fault: decode one round, verify logical state."""

    @pytest.mark.parametrize(
        "state, expected_bit", _Z_EIGENSTATES, ids=[s for s, _ in _Z_EIGENSTATES]
    )
    @pytest.mark.parametrize(
        "fault_pauli, fault_site", _ALL_SINGLE_QUBIT_FAULTS,
        ids=[f"{p}_{s}" for p, s in _ALL_SINGLE_QUBIT_FAULTS],
    )
    def test_logical_z(
        self,
        state: InjectedStateName,
        expected_bit: int,
        fault_pauli: str,
        fault_site: int,
    ) -> None:
        """Inject a single-qubit fault, decode, and verify corrected Z_L readout."""
        circuit = _build_patch_inject_fault_and_measure(
            injected_state=state,
            fault_pauli=fault_pauli,
            fault_site=fault_site,
            logical_basis="Z",
        )
        corrected = _run_and_decode(circuit, logical_basis="Z")
        assert corrected == expected_bit, (
            f"State {state!r}, fault {fault_pauli}@{fault_site}: "
            f"corrected Z_L readout {corrected} != expected {expected_bit}"
        )

    @pytest.mark.parametrize(
        "state, expected_bit", _X_EIGENSTATES, ids=[s for s, _ in _X_EIGENSTATES]
    )
    @pytest.mark.parametrize(
        "fault_pauli, fault_site", _ALL_SINGLE_QUBIT_FAULTS,
        ids=[f"{p}_{s}" for p, s in _ALL_SINGLE_QUBIT_FAULTS],
    )
    def test_logical_x(
        self,
        state: InjectedStateName,
        expected_bit: int,
        fault_pauli: str,
        fault_site: int,
    ) -> None:
        """Inject a single-qubit fault, decode, and verify corrected X_L readout."""
        circuit = _build_patch_inject_fault_and_measure(
            injected_state=state,
            fault_pauli=fault_pauli,
            fault_site=fault_site,
            logical_basis="X",
        )
        corrected = _run_and_decode(circuit, logical_basis="X")
        assert corrected == expected_bit, (
            f"State {state!r}, fault {fault_pauli}@{fault_site}: "
            f"corrected X_L readout {corrected} != expected {expected_bit}"
        )


# ===========================================================================
# Two-round decode: fault → syndrome → decode → fault → syndrome → decode
# ===========================================================================

_TWO_ROUND_FAULT_PAIRS: list[tuple[tuple[str, int], tuple[str, int]]] = [
    (f1, f2)
    for f1 in _ALL_SINGLE_QUBIT_FAULTS
    for f2 in _ALL_SINGLE_QUBIT_FAULTS
]

# Use a representative subset to keep runtime reasonable (full cross-product
# is 21 × 21 = 441 pairs × 4 states = 1764 tests per basis).
_TWO_ROUND_SAMPLE: list[tuple[tuple[str, int], tuple[str, int]]] = [
    # Same site, same Pauli (cancels)
    (("X", 0), ("X", 0)),
    (("Z", 4), ("Z", 4)),
    (("Y", 9), ("Y", 9)),
    # Same site, different Pauli
    (("X", 3), ("Z", 3)),
    (("Z", 10), ("X", 10)),
    (("X", 5), ("Y", 5)),
    # Different sites, same Pauli
    (("X", 0), ("X", 8)),
    (("Z", 3), ("Z", 10)),
    (("Y", 4), ("Y", 9)),
    # Different sites, different Pauli
    (("X", 0), ("Z", 9)),
    (("Z", 3), ("X", 5)),
    (("Y", 8), ("X", 10)),
    (("X", 4), ("Y", 0)),
    # Faults in the logical support (8, 9, 10)
    (("X", 8), ("X", 9)),
    (("Z", 9), ("Z", 10)),
    (("X", 8), ("Z", 10)),
    # Faults outside the logical support
    (("X", 0), ("X", 3)),
    (("Z", 4), ("Z", 5)),
]

_TWO_ROUND_IDS = [
    f"{p1}{s1}-{p2}{s2}" for (p1, s1), (p2, s2) in _TWO_ROUND_SAMPLE
]


def _build_two_round_circuit(
    injected_state: InjectedStateName,
    fault1: tuple[str, int],
    fault2: tuple[str, int],
    logical_basis: str,
) -> QuantumCircuit:
    """Build: injection → fault1 → syn round 1 → fault2 → syn round 2 → logical readout."""
    patch = QuantumRegister(NUM_PATCH_QUBITS, "patch")
    syn_anc = QuantumRegister(6, "syn_anc")
    syn1_cl = ClassicalRegister(6, "syn1")
    syn2_cl = ClassicalRegister(6, "syn2")
    log_cl = ClassicalRegister(3, "log")
    circuit = QuantumCircuit(patch, syn_anc, syn1_cl, syn2_cl, log_cl)

    # 1. State injection.
    append_patch_steane_state_injection_encoder(
        circuit,
        patch_qubits=list(patch),
        injected_state=injected_state,
        add_barriers=False,
    )

    # 2. Round 1: fault → stabilizer measurement.
    append_patch_pauli_faults(circuit, [fault1], patch_qubits=list(patch))
    append_patch_stabilizer_measurement(
        circuit,
        patch_qubits=list(patch),
        ancilla_qubits=list(syn_anc),
        classical_bits=list(syn1_cl),
        add_barriers=False,
    )

    # 3. Reset ancillas for round 2.
    for anc in syn_anc:
        circuit.reset(anc)

    # 4. Round 2: fault → stabilizer measurement.
    append_patch_pauli_faults(circuit, [fault2], patch_qubits=list(patch))
    append_patch_stabilizer_measurement(
        circuit,
        patch_qubits=list(patch),
        ancilla_qubits=list(syn_anc),
        classical_bits=list(syn2_cl),
        add_barriers=False,
    )

    # 5. Destructive logical measurement.
    support = PATCH_DEFAULT_LOGICAL_Z if logical_basis == "Z" else PATCH_DEFAULT_LOGICAL_X
    append_patch_logical_measurement(
        circuit,
        logical_basis,
        patch_qubits=list(patch),
        support=support,
        classical_bits=list(log_cl),
        destructive=True,
    )

    return circuit


def _parse_syndrome_bits(syn_str: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Parse a 6-char syndrome chunk into (x_raw, z_raw) tuples."""
    bits = [int(b) for b in reversed(syn_str)]
    return tuple(bits[0:3]), tuple(bits[3:6])


def _run_and_decode_two_rounds(
    circuit: QuantumCircuit,
    logical_basis: str,
) -> int:
    """Run a two-round circuit, decode both rounds, return corrected logical bit."""
    sim = AerSimulator()
    bitstring = next(iter(sim.run(circuit, shots=1).result().get_counts())).replace(" ", "")

    # Bitstring layout (Qiskit: last-declared register is leftmost):
    # log_cl(3) | syn2_cl(6) | syn1_cl(6)  → total 15 bits
    assert len(bitstring) == 15, f"Expected 15 classical bits, got {len(bitstring)}"

    log_str = bitstring[:3]
    syn2_str = bitstring[3:9]
    syn1_str = bitstring[9:15]

    # Parse logical measurement parity.
    log_bits = [int(b) for b in reversed(log_str)]
    raw_logical_parity = log_bits[0] ^ log_bits[1] ^ log_bits[2]

    # Initialize frames.
    stab_frame = StabilizerFrame()
    for label in STEANE_PATCH_X_STABILIZER_LABELS:
        stab_frame.register(label, 0)
    for label in STEANE_PATCH_Z_STABILIZER_LABELS:
        stab_frame.register(label, 0)

    pauli_frame = PauliFrame()
    pauli_frame.register("L0")

    # --- Decode round 1 ---
    x1_raw, z1_raw = _parse_syndrome_bits(syn1_str)
    x1_meas = {label: x1_raw[i] for i, label in enumerate(STEANE_PATCH_X_STABILIZER_LABELS)}
    z1_meas = {label: z1_raw[i] for i, label in enumerate(STEANE_PATCH_Z_STABILIZER_LABELS)}

    x1_deltas = stab_frame.advance(x1_meas)
    z1_deltas = stab_frame.advance(z1_meas)

    x1_syn = tuple(x1_deltas[l] for l in STEANE_PATCH_X_STABILIZER_LABELS)
    z1_syn = tuple(z1_deltas[l] for l in STEANE_PATCH_Z_STABILIZER_LABELS)

    recovery1 = decode_steane_syndrome(z_syndrome=z1_syn, x_syndrome=x1_syn)
    pauli_frame.update_x("L0", flip=recovery1.flips_logical_z)
    pauli_frame.update_z("L0", flip=recovery1.flips_logical_x)

    # Update stabilizer frame: after virtual correction, the stabilizer
    # frame should reflect the corrected state (all +1 again).
    # The correction undoes the error, so the next round's reference is
    # the post-correction syndrome = the raw syndrome of this round
    # (already stored by advance above).

    # --- Decode round 2 ---
    x2_raw, z2_raw = _parse_syndrome_bits(syn2_str)
    x2_meas = {label: x2_raw[i] for i, label in enumerate(STEANE_PATCH_X_STABILIZER_LABELS)}
    z2_meas = {label: z2_raw[i] for i, label in enumerate(STEANE_PATCH_Z_STABILIZER_LABELS)}

    x2_deltas = stab_frame.advance(x2_meas)
    z2_deltas = stab_frame.advance(z2_meas)

    x2_syn = tuple(x2_deltas[l] for l in STEANE_PATCH_X_STABILIZER_LABELS)
    z2_syn = tuple(z2_deltas[l] for l in STEANE_PATCH_Z_STABILIZER_LABELS)

    recovery2 = decode_steane_syndrome(z_syndrome=z2_syn, x_syndrome=x2_syn)
    pauli_frame.update_x("L0", flip=recovery2.flips_logical_z)
    pauli_frame.update_z("L0", flip=recovery2.flips_logical_x)

    # Interpret logical readout through accumulated Pauli frame.
    if logical_basis == "Z":
        corrected = pauli_frame.interpret_z_readout("L0", raw_logical_parity)
    else:
        corrected = pauli_frame.interpret_x_readout("L0", raw_logical_parity)

    return corrected


class TestTwoRoundDecodeCorrect:
    """Two consecutive correctable faults: decode both rounds, verify logical state."""

    @pytest.mark.parametrize(
        "state, expected_bit", _Z_EIGENSTATES, ids=[s for s, _ in _Z_EIGENSTATES]
    )
    @pytest.mark.parametrize(
        "fault1, fault2", _TWO_ROUND_SAMPLE, ids=_TWO_ROUND_IDS,
    )
    def test_logical_z(
        self,
        state: InjectedStateName,
        expected_bit: int,
        fault1: tuple[str, int],
        fault2: tuple[str, int],
    ) -> None:
        """Two-round Z_L readout: fault1 → decode → fault2 → decode → verify."""
        circuit = _build_two_round_circuit(state, fault1, fault2, logical_basis="Z")
        corrected = _run_and_decode_two_rounds(circuit, logical_basis="Z")
        assert corrected == expected_bit, (
            f"State {state!r}, faults {fault1}→{fault2}: "
            f"corrected Z_L = {corrected} != expected {expected_bit}"
        )

    @pytest.mark.parametrize(
        "state, expected_bit", _X_EIGENSTATES, ids=[s for s, _ in _X_EIGENSTATES]
    )
    @pytest.mark.parametrize(
        "fault1, fault2", _TWO_ROUND_SAMPLE, ids=_TWO_ROUND_IDS,
    )
    def test_logical_x(
        self,
        state: InjectedStateName,
        expected_bit: int,
        fault1: tuple[str, int],
        fault2: tuple[str, int],
    ) -> None:
        """Two-round X_L readout: fault1 → decode → fault2 → decode → verify."""
        circuit = _build_two_round_circuit(state, fault1, fault2, logical_basis="X")
        corrected = _run_and_decode_two_rounds(circuit, logical_basis="X")
        assert corrected == expected_bit, (
            f"State {state!r}, faults {fault1}→{fault2}: "
            f"corrected X_L = {corrected} != expected {expected_bit}"
        )

