"""Tests for the Steane-code lookup-table decoder on the 11-site patch.

Verification is by logical expectation values: after injecting a correctable
fault, measuring the syndrome, and decoding via the Pauli frame, the
frame-corrected ⟨Z_L⟩ and ⟨X_L⟩ must match the original encoded state.
This works for any injected state — including non-eigenstates like |T_L⟩
and |Ty_L⟩ — and checks both logical operators simultaneously.
"""

from __future__ import annotations

import math

import pytest
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer import AerSimulator

from magic_state_cultivation.decoding import PauliFrame, StabilizerFrame, SteaneDecoder
from magic_state_cultivation.noise import append_pauli_faults
from magic_state_cultivation.steane_code.decoding.decoder import (
    SYNDROME_BIT_LABELS,
    X_STABILIZER_LABELS,
    Z_STABILIZER_LABELS,
)
from magic_state_cultivation.steane_code.state_injection_qiskit import (
    NUM_PATCH_QUBITS,
    PATCH_DATA_SITES,
    PATCH_DEFAULT_LOGICAL_X,
    PATCH_DEFAULT_LOGICAL_Z,
    InjectedStateName,
    append_patch_stabilizer_measurement,
    append_patch_steane_state_injection_encoder_crumble_sequences,
)
from magic_state_cultivation.utils.normalize import parse_qiskit_bitstring

TOLERANCE = 1e-10

_SYNDROME_BIT_LABELS = SYNDROME_BIT_LABELS

# (⟨Z_L⟩, ⟨X_L⟩) for each injected state.
_inv_sqrt2 = 1.0 / math.sqrt(2)
_EXPECTED: dict[str, tuple[float, float]] = {
    "0": (1.0, 0.0),
    "T": (0.0, _inv_sqrt2),
    "Ty": (_inv_sqrt2, _inv_sqrt2),
}

_TEST_STATES: list[InjectedStateName] = list(_EXPECTED.keys())

_ALL_SINGLE_QUBIT_FAULTS: list[tuple[str, int]] = [
    (pauli, site) for pauli in ("X", "Y", "Z") for site in PATCH_DATA_SITES
]

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

_TWO_ROUND_IDS = [f"{p1}{s1}-{p2}{s2}" for (p1, s1), (p2, s2) in _TWO_ROUND_SAMPLE]


# ===========================================================================
# Shared helpers
# ===========================================================================


def _logical_operator(basis: str, support: tuple[int, ...]) -> SparsePauliOp:
    """Build the logical Z_L or X_L operator on the 11-qubit patch."""
    chars = ["I"] * NUM_PATCH_QUBITS
    for q in support:
        chars[q] = basis
    return SparsePauliOp("".join(reversed(chars)))


_Z_L = _logical_operator("Z", PATCH_DEFAULT_LOGICAL_Z)
_X_L = _logical_operator("X", PATCH_DEFAULT_LOGICAL_X)

# Patch qubit indices used when computing expectation values on a larger
# statevector (e.g., 17 qubits = 11 patch + 6 ancilla).
_PATCH_QUBIT_INDICES = list(range(NUM_PATCH_QUBITS))


def _init_stab_frame() -> StabilizerFrame:
    """Return a StabilizerFrame initialized to all +1."""
    frame = StabilizerFrame()
    for label in X_STABILIZER_LABELS:
        frame.register(label, 0)
    for label in Z_STABILIZER_LABELS:
        frame.register(label, 0)
    return frame


def _parse_syndrome_bits(syn_str: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Parse a 6-char syndrome chunk into (x_raw, z_raw) tuples."""
    bits = parse_qiskit_bitstring(syn_str, _SYNDROME_BIT_LABELS)
    return (
        tuple(bits[label] for label in _SYNDROME_BIT_LABELS[:3]),
        tuple(bits[label] for label in _SYNDROME_BIT_LABELS[3:]),
    )


_decoder = SteaneDecoder()


def _corrected_logical_expectations(
    sv: Statevector,
    pauli_frame: PauliFrame,
) -> tuple[float, float]:
    """Return frame-corrected (⟨Z_L⟩, ⟨X_L⟩) from the post-measurement statevector."""
    raw_z = sv.expectation_value(_Z_L, _PATCH_QUBIT_INDICES).real
    raw_x = sv.expectation_value(_X_L, _PATCH_QUBIT_INDICES).real
    return (
        pauli_frame.correct_z_expectation("L0", raw_z),
        pauli_frame.correct_x_expectation("L0", raw_x),
    )


def _assert_logical_expectations(
    corrected_z: float,
    corrected_x: float,
    state: InjectedStateName,
    label: str,
) -> None:
    """Assert corrected ⟨Z_L⟩ and ⟨X_L⟩ match expected values for the state."""
    expected_z, expected_x = _EXPECTED[state]
    assert corrected_z == pytest.approx(
        expected_z, abs=TOLERANCE
    ), f"{label}: ⟨Z_L⟩ = {corrected_z:.6f}, expected {expected_z:.6f}"
    assert corrected_x == pytest.approx(
        expected_x, abs=TOLERANCE
    ), f"{label}: ⟨X_L⟩ = {corrected_x:.6f}, expected {expected_x:.6f}"


# ===========================================================================
# One-round decode: fault → syndrome → decode → expectation check
# ===========================================================================


def _build_one_round_syndrome_circuit(
    injected_state: InjectedStateName,
    fault: tuple[str, int],
) -> QuantumCircuit:
    """Build: injection → fault → stabilizer measurement (syndrome only)."""
    patch = QuantumRegister(NUM_PATCH_QUBITS, "patch")
    syn_cl = ClassicalRegister(6, "syn")
    circuit = QuantumCircuit(patch, syn_cl)

    append_patch_steane_state_injection_encoder_crumble_sequences(
        circuit,
        patch_qubits=list(patch),
        injected_state=injected_state,
        add_barriers=False,
    )
    append_pauli_faults(circuit, [fault], qubit_map=list(patch))
    append_patch_stabilizer_measurement(
        circuit,
        patch_qubits=list(patch),
        classical_bits=list(syn_cl),
        add_barriers=False,
    )
    return circuit


class TestOneRoundDecodeCorrect:
    """Single correctable fault: decode, check frame-corrected logical expectations."""

    @pytest.mark.parametrize("state", _TEST_STATES)
    @pytest.mark.parametrize(
        "fault",
        _ALL_SINGLE_QUBIT_FAULTS,
        ids=[f"{p}_{s}" for p, s in _ALL_SINGLE_QUBIT_FAULTS],
    )
    def test_logical_expectations(self, state: InjectedStateName, fault: tuple[str, int]) -> None:
        """Frame-corrected ⟨Z_L⟩ and ⟨X_L⟩ match expected after one correctable fault."""
        # 1. Run syndrome circuit → syndrome bits + post-measurement statevector.
        syn_circuit = _build_one_round_syndrome_circuit(state, fault)
        syn_circuit.save_statevector(label="sv")

        sim = AerSimulator(method="statevector")
        result = sim.run(syn_circuit, shots=1).result()
        bitstring = next(iter(result.get_counts())).replace(" ", "")
        sv = result.data()["sv"]

        x_raw, z_raw = _parse_syndrome_bits(bitstring)

        # 2. Decode and update Pauli frame (frame is updated internally by the decoder).
        stab_frame = _init_stab_frame()
        pauli_frame = PauliFrame()
        pauli_frame.register("L0")
        _decoder.decode_from_raw_syndromes(
            pauli_frame=pauli_frame,
            logical_qubit_label="L0",
            x_raw=x_raw,
            z_raw=z_raw,
            stab_frame=stab_frame,
        )

        # 3. Check frame-corrected logical expectations on the post-measurement
        #    statevector. The logical operators are I on ancilla qubits, so the
        #    collapsed ancilla state doesn't affect the expectation values.
        corrected_z, corrected_x = _corrected_logical_expectations(sv, pauli_frame)
        _assert_logical_expectations(
            corrected_z,
            corrected_x,
            state,
            label=f"State {state!r}, fault {fault}",
        )


# ===========================================================================
# Two-round decode: fault → syndrome → decode → fault → syndrome → decode
# ===========================================================================


def _build_two_round_syndrome_circuit(
    injected_state: InjectedStateName,
    fault1: tuple[str, int],
    fault2: tuple[str, int],
) -> QuantumCircuit:
    """Build: injection → fault1 → syn1 → fault2 → syn2 (syndrome only)."""
    patch = QuantumRegister(NUM_PATCH_QUBITS, "patch")
    syn1_cl = ClassicalRegister(6, "syn1")
    syn2_cl = ClassicalRegister(6, "syn2")
    circuit = QuantumCircuit(patch, syn1_cl, syn2_cl)

    append_patch_steane_state_injection_encoder_crumble_sequences(
        circuit,
        patch_qubits=list(patch),
        injected_state=injected_state,
        add_barriers=False,
    )

    # Round 1.
    append_pauli_faults(circuit, [fault1], qubit_map=list(patch))
    append_patch_stabilizer_measurement(
        circuit,
        patch_qubits=list(patch),
        classical_bits=list(syn1_cl),
        add_barriers=False,
    )

    # Round 2.
    append_pauli_faults(circuit, [fault2], qubit_map=list(patch))
    append_patch_stabilizer_measurement(
        circuit,
        patch_qubits=list(patch),
        classical_bits=list(syn2_cl),
        add_barriers=False,
    )

    return circuit


class TestTwoRoundDecodeCorrect:
    """Two consecutive correctable faults: decode both, check frame-corrected expectations."""

    @pytest.mark.parametrize("state", _TEST_STATES)
    @pytest.mark.parametrize(
        "fault1, fault2",
        _TWO_ROUND_SAMPLE,
        ids=_TWO_ROUND_IDS,
    )
    def test_logical_expectations(
        self,
        state: InjectedStateName,
        fault1: tuple[str, int],
        fault2: tuple[str, int],
    ) -> None:
        """Frame-corrected ⟨Z_L⟩ and ⟨X_L⟩ match expected after two correctable faults."""
        # 1. Run the two-round syndrome circuit → syndrome bits + statevector.
        syn_circuit = _build_two_round_syndrome_circuit(state, fault1, fault2)
        syn_circuit.save_statevector(label="sv")

        sim = AerSimulator(method="statevector")
        result = sim.run(syn_circuit, shots=1).result()
        bitstring = next(iter(result.get_counts())).replace(" ", "")
        sv = result.data()["sv"]

        assert len(bitstring) == 12, f"Expected 12 classical bits, got {len(bitstring)}"
        syn2_str = bitstring[:6]
        syn1_str = bitstring[6:12]

        # 2. Decode both rounds → accumulated Pauli frame.
        #
        # stab_frame: tracks the last raw syndrome as a reference. Each call to
        # _decode_one_syndrome advances it, so that round 2 sees the *change*
        # relative to round 1 (i.e., only the new error's contribution).
        #
        # pauli_frame: accumulates logical byproducts across rounds via XOR.
        # An X byproduct means Z_L readouts must be sign-flipped; a Z byproduct
        # means X_L readouts must be sign-flipped.
        stab_frame = _init_stab_frame()
        pauli_frame = PauliFrame()
        pauli_frame.register("L0")

        x1_raw, z1_raw = _parse_syndrome_bits(syn1_str)
        _decoder.decode_from_raw_syndromes(
            pauli_frame=pauli_frame,
            logical_qubit_label="L0",
            x_raw=x1_raw,
            z_raw=z1_raw,
            stab_frame=stab_frame,
        )

        x2_raw, z2_raw = _parse_syndrome_bits(syn2_str)
        _decoder.decode_from_raw_syndromes(
            pauli_frame=pauli_frame,
            logical_qubit_label="L0",
            x_raw=x2_raw,
            z_raw=z2_raw,
            stab_frame=stab_frame,
        )

        # 3. Check frame-corrected logical expectations on the post-measurement
        #    statevector directly from the syndrome circuit.
        corrected_z, corrected_x = _corrected_logical_expectations(sv, pauli_frame)
        _assert_logical_expectations(
            corrected_z,
            corrected_x,
            state,
            label=f"State {state!r}, faults {fault1}→{fault2}",
        )
