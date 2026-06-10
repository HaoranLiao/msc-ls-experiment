from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from qiskit import QuantumCircuit
from qiskit.circuit import Qubit

from research_qec_haoran.utils.gates import apply_ty


InjectedStateName = Literal["0", "1", "+", "-", "i", "-i", "Y+", "Y-", "S", "T", "Tdg", "H"]

# Steane [[7,1,3]] qubit layout (Fano plane in triangular form):
#
#        0
#     1     2
#        3
# 4      5      6
#
# Faces (stabilizers, weight-4):
#   Top: {0, 1, 2, 3}
#   Bottom-left: {1, 3, 4, 5}
#   Bottom-right: {2, 3, 5, 6}
#
# Sides (logicals, weight-3):
#   Left: {0, 1, 4}    Right: {0, 2, 6}    Bottom: {4, 5, 6}

# Steane [[7,1,3]] stabilizers and logical operators for qubits 0–6.
X_STABILIZERS: tuple[tuple[int, ...], ...] = (
    (0, 1, 2, 3),
    (1, 3, 4, 5),
    (2, 3, 5, 6),
)

Z_STABILIZERS: tuple[tuple[int, ...], ...] = (
    (0, 1, 2, 3),
    (1, 3, 4, 5),
    (2, 3, 5, 6),
)

DEFAULT_LOGICAL_X: tuple[int, ...] = (0, 1, 4)
DEFAULT_LOGICAL_Z: tuple[int, ...] = (0, 1, 4)

# All weight-3 logical representatives (Fano plane lines).
LOGICAL_X_REPRESENTATIVES: tuple[tuple[int, ...], ...] = (
    (0, 1, 4),
    (0, 2, 6),
    (0, 3, 5),
    (1, 2, 5),
    (1, 3, 6),
    (2, 3, 4),
    (4, 5, 6),
)

LOGICAL_Z_REPRESENTATIVES: tuple[tuple[int, ...], ...] = (
    (0, 1, 4),
    (0, 2, 6),
    (0, 3, 5),
    (1, 2, 5),
    (1, 3, 6),
    (2, 3, 4),
    (4, 5, 6),
)


STEANE_ENCODER_CNOTS: tuple[tuple[int, int], ...] = (
    (6, 2),
    (0, 6),
    (4, 0),
    (2, 5),
    (2, 3),
    (0, 1),
    (1, 3),
    (1, 4),
    (4, 5),
    (0, 2),
)


def prepare_physical_state(
    circuit: QuantumCircuit,
    qubit: int | Qubit,
    state: InjectedStateName,
) -> None:
    """Prepare the single-qubit state to be encoded.

    Parameters
    ----------
    circuit
        Circuit being built.
    qubit
        Qubit on which the unencoded input state is prepared.
    state
        Named state to prepare.
    """
    if state == "0":
        return

    if state == "1":
        circuit.x(qubit)
        return

    if state == "+":
        circuit.h(qubit)
        return

    if state == "-":
        circuit.x(qubit)
        circuit.h(qubit)
        return

    if state in {"i", "Y+", "S"}:
        # +1 eigenstate of Pauli Y
        # |Y+> = S|+> = (|0> + i|1>) / sqrt(2)
        circuit.h(qubit)
        circuit.s(qubit)
        return

    if state in {"-i", "Y-"}:
        # -1 eigenstate of Pauli Y
        # |Y-> = Sdg|+> = (|0> - i|1>) / sqrt(2)
        circuit.h(qubit)
        circuit.sdg(qubit)
        return

    if state == "T":
        # +1 eigenstate of (X+Y)/sqrt(2)
        # |T> = T|+> = (|0> + exp(i*pi/4)|1>) / sqrt(2)
        circuit.h(qubit)
        circuit.t(qubit)
        return

    if state == "Tdg":
        # +1 eigenstate of (X-Y)/sqrt(2)
        # |Tdg> = Tdg|+> = (|0> + exp(-i*pi/4)|1>) / sqrt(2)
        circuit.h(qubit)
        circuit.tdg(qubit)
        return

    if state == "H":
        # +1 eigenstate of H
        # |H> = cos(pi/8)|0> + sin(pi/8)|1> = T_y|0>
        apply_ty(circuit, qubit)
        return

    raise ValueError(f"Unknown state name: {state!r}")


def build_steane_state_injection_encoder(
    injected_state: InjectedStateName = "H",
    *,
    add_barriers: bool = True,
) -> QuantumCircuit:
    """Build a Steane-code state-injection encoder.

    The circuit implements the map
    |psi> |+> |+> |+> |0> |0> |0> -> |psi_L>.

    Parameters
    ----------
    injected_state
        The unencoded single-qubit state to inject on qubit 0. The output is
        the corresponding encoded state in the Steane code.
    add_barriers
        Whether to insert barriers between preparation and encoding stages.

    Returns
    -------
    QuantumCircuit
        A 7-qubit circuit preparing the encoded Steane state.
    """
    circuit = QuantumCircuit(7, name=f"steane_encode_{injected_state}")

    # Prepare the unencoded state on q0.
    prepare_physical_state(circuit, 0, injected_state)

    # Prepare the three Steane codeword-superposition variables u, v, w.
    circuit.h(1)
    circuit.h(4)
    circuit.h(6)

    if add_barriers:
        circuit.barrier()

    # CNOT encoder.
    for control_qubit, target_qubit in STEANE_ENCODER_CNOTS:
        circuit.cx(control_qubit, target_qubit)

    return circuit


def append_stabilizer_measurement(
    circuit: QuantumCircuit,
    *,
    data_qubits: Sequence,
    ancilla_qubits: Sequence,
    classical_bits: Sequence,
    add_barriers: bool = True,
) -> None:
    """Append non-destructive stabilizer measurements to an existing circuit.

    Measures all 6 Steane-code stabilizers (3 X-type, 3 Z-type) using
    dedicated ancilla qubits.

    Parameters
    ----------
    circuit
        Circuit being modified.
    data_qubits
        Seven qubits holding the encoded Steane-code state (ordered 0–6).
    ancilla_qubits
        Six ancilla qubits: indices 0–2 for X-stabilizers, 3–5 for Z-stabilizers.
    classical_bits
        Six classical bits: bits 0–2 hold X-stabilizer outcomes,
        bits 3–5 hold Z-stabilizer outcomes.
    add_barriers
        Whether to insert a barrier between X and Z stabilizer rounds.
    """
    if len(data_qubits) < 7:
        raise ValueError("data_qubits must contain at least 7 qubits.")
    if len(ancilla_qubits) < 6:
        raise ValueError("ancilla_qubits must contain at least 6 qubits.")
    if len(classical_bits) < 6:
        raise ValueError("classical_bits must contain at least 6 bits.")

    dq = list(data_qubits)
    aq = list(ancilla_qubits)
    cb = list(classical_bits)

    # X-stabilizer measurement: ancilla in |+⟩, CX(anc→data), H, measure.
    for i, support in enumerate(X_STABILIZERS):
        circuit.h(aq[i])
        for abstract_q in support:
            circuit.cx(aq[i], dq[abstract_q])
        circuit.h(aq[i])
        circuit.measure(aq[i], cb[i])

    if add_barriers:
        circuit.barrier()

    # Z-stabilizer measurement: ancilla in |0⟩, CX(data→anc), measure.
    for i, support in enumerate(Z_STABILIZERS):
        for abstract_q in support:
            circuit.cx(dq[abstract_q], aq[3 + i])
        circuit.measure(aq[3 + i], cb[3 + i])


def append_logical_measurement(
    circuit: QuantumCircuit,
    basis: str = "Z",
    *,
    data_qubits: Sequence,
    support: tuple[int, ...] | None = None,
    ancilla_qubit=None,
    classical_bits: Sequence,
    destructive: bool = True,
) -> None:
    """Append a logical measurement to an existing circuit.

    Parameters
    ----------
    circuit
        Circuit being modified.
    basis
        Which logical to measure: ``"Z"`` for Z_L or ``"X"`` for X_L.
    data_qubits
        Seven qubits holding the encoded Steane-code state (ordered 0–6).
    support
        Abstract qubit indices (0–6) for the logical representative. Defaults
        to ``DEFAULT_LOGICAL_Z`` or ``DEFAULT_LOGICAL_X``.
    ancilla_qubit
        For non-destructive measurement, the ancilla qubit to use. Required
        when ``destructive=False``.
    classical_bits
        Classical bits for measurement results. For destructive: one per
        support qubit. For non-destructive: at least one bit.
    destructive
        If True (default), directly measure the data qubits in the
        appropriate basis. If False, use the ancilla for indirect measurement.
    """
    if len(data_qubits) < 7:
        raise ValueError("data_qubits must contain at least 7 qubits.")
    if support is None:
        support = DEFAULT_LOGICAL_Z if basis == "Z" else DEFAULT_LOGICAL_X

    dq = list(data_qubits)
    cb = list(classical_bits)

    if destructive:
        if len(cb) < len(support):
            raise ValueError(
                f"classical_bits must have at least {len(support)} entries for destructive measurement."
            )
        if basis == "X":
            for abstract_q in support:
                circuit.h(dq[abstract_q])
        for i, abstract_q in enumerate(support):
            circuit.measure(dq[abstract_q], cb[i])
    else:
        if ancilla_qubit is None:
            raise ValueError("ancilla_qubit is required for non-destructive measurement.")
        if len(cb) < 1:
            raise ValueError("classical_bits must have at least 1 entry for non-destructive measurement.")
        if basis == "Z":
            for abstract_q in support:
                circuit.cx(dq[abstract_q], ancilla_qubit)
            circuit.measure(ancilla_qubit, cb[0])
        else:
            circuit.h(ancilla_qubit)
            for abstract_q in support:
                circuit.cx(ancilla_qubit, dq[abstract_q])
            circuit.h(ancilla_qubit)
            circuit.measure(ancilla_qubit, cb[0])


# =============================================================================
# Map to 11-site cultivation patch
# =============================================================================

# 11-site labeling used in the cultivation patch:
#
#               0
#
#           1       2
#
#       3       4       5
#
#           6       7
#
#       8       9       10
#
# Data sites:   (0, 3, 4, 5, 8, 9, 10)
# Helper sites: (1, 2, 6, 7)

# The 7 abstract Steane qubits (0–6) map to 11-site patch data sites as:
#   abstract 0 → patch site 0  (injection)
#   abstract 1 → patch site 3  (|+⟩)
#   abstract 2 → patch site 5  (|0⟩)
#   abstract 3 → patch site 4  (|0⟩)
#   abstract 4 → patch site 8  (|+⟩)
#   abstract 5 → patch site 9  (|0⟩)
#   abstract 6 → patch site 10 (|+⟩)

INJECTION_SITE: int = 0
PLUS_SITES: tuple[int, ...] = (3, 8, 10)

# Map from abstract Steane qubit index (0–6) to 11-site patch data site.
_STEANE_TO_PATCH: tuple[int, ...] = (0, 3, 5, 4, 8, 9, 10)

NUM_PATCH_QUBITS: int = 11
PATCH_DATA_SITES: tuple[int, ...] = (0, 3, 4, 5, 8, 9, 10)
PATCH_HELPER_SITES: tuple[int, ...] = (1, 2, 6, 7)


def _map_support_to_patch(support: tuple[int, ...]) -> tuple[int, ...]:
    """Map abstract Steane qubit indices to 11-site patch data sites."""
    return tuple(sorted(_STEANE_TO_PATCH[i] for i in support))


# ---------------------------------------------------------------------------
# Mapped stabilizers (expressed in 11-site patch labels)
# ---------------------------------------------------------------------------

PATCH_X_STABILIZERS: tuple[tuple[int, ...], ...] = (
    (0, 3, 4, 5),
    (3, 4, 8, 9),
    (4, 5, 9, 10),
)

PATCH_Z_STABILIZERS: tuple[tuple[int, ...], ...] = (
    (0, 3, 4, 5),
    (3, 4, 8, 9),
    (4, 5, 9, 10),
)

# ---------------------------------------------------------------------------
# Mapped logical operators (expressed in 11-site patch labels)
# ---------------------------------------------------------------------------

PATCH_DEFAULT_LOGICAL_X: tuple[int, ...] = (0, 3, 8)

PATCH_DEFAULT_LOGICAL_Z: tuple[int, ...] = (0, 3, 8)

PATCH_LOGICAL_X_REPRESENTATIVES: tuple[tuple[int, ...], ...] = (
    (0, 3, 8),
    (0, 5, 10),
    (0, 4, 9),
    (3, 5, 9),
    (3, 4, 10),
    (4, 5, 8),
    (8, 9, 10),
)

PATCH_LOGICAL_Z_REPRESENTATIVES: tuple[tuple[int, ...], ...] = (
    (0, 3, 8),
    (0, 5, 10),
    (0, 4, 9),
    (3, 5, 9),
    (3, 4, 10),
    (4, 5, 8),
    (8, 9, 10),
)


# ---------------------------------------------------------------------------
# Encoder on the 11-site patch
# ---------------------------------------------------------------------------

def append_patch_steane_state_injection_encoder(
    circuit: QuantumCircuit,
    *,
    patch_qubits: Sequence,
    injected_state: InjectedStateName = "H",
    add_barriers: bool = True,
) -> None:
    """Append the Steane-code state-injection encoder on the 11-site patch.

    Composes the 7-qubit encoder from build_steane_state_injection_encoder
    onto the 7 data sites of the 11-site patch via _STEANE_TO_PATCH.

    Parameters
    ----------
    circuit
        Circuit being modified.
    patch_qubits
        Eleven Qiskit qubits indexed by the 11-site patch labels.
    injected_state
        The physical one-qubit state to inject on site 0.
    add_barriers
        Whether to insert a barrier after the encoder.
    """
    if len(patch_qubits) < 11:
        raise ValueError("patch_qubits must contain at least 11 qubits.")

    q = list(patch_qubits)

    encoder = build_steane_state_injection_encoder(
        injected_state=injected_state, add_barriers=add_barriers,
    )

    # Map abstract encoder qubits 0–6 to the corresponding patch data sites.
    data_qubits = [q[_STEANE_TO_PATCH[i]] for i in range(7)]
    circuit.compose(encoder, qubits=data_qubits, inplace=True)
    if add_barriers:
        circuit.barrier(label="state injection")


# ---------------------------------------------------------------------------
# Stabilizer measurement on the 11-site patch
# ---------------------------------------------------------------------------

def append_patch_stabilizer_measurement(
    circuit: QuantumCircuit,
    *,
    patch_qubits: Sequence,
    ancilla_qubits: Sequence,
    classical_bits: Sequence,
    add_barriers: bool = True,
) -> None:
    """Append non-destructive stabilizer measurements on the 11-site patch.

    Uses 6 ancilla qubits (one per stabilizer) to measure all stabilizers
    without disturbing the encoded state on the patch data sites.

    Parameters
    ----------
    circuit
        Circuit being modified.
    patch_qubits
        Eleven qubits indexed by the 11-site patch labels.
    ancilla_qubits
        Six ancilla qubits: indices 0–2 for X-stabilizers, 3–5 for Z-stabilizers.
    classical_bits
        Six classical bits: bits 0–2 hold X-stabilizer outcomes,
        bits 3–5 hold Z-stabilizer outcomes.
    add_barriers
        Whether to insert a barrier between X and Z stabilizer rounds.
    """
    if len(patch_qubits) < 11:
        raise ValueError("patch_qubits must contain at least 11 qubits.")
    if len(ancilla_qubits) < 6:
        raise ValueError("ancilla_qubits must contain at least 6 qubits.")
    if len(classical_bits) < 6:
        raise ValueError("classical_bits must contain at least 6 bits.")

    pq = list(patch_qubits)
    aq = list(ancilla_qubits)
    cb = list(classical_bits)

    # X-stabilizer measurement: ancilla in |+⟩, CX(anc→data), H, measure.
    for i, support in enumerate(PATCH_X_STABILIZERS):
        circuit.h(aq[i])
        for site in support:
            circuit.cx(aq[i], pq[site])
        circuit.h(aq[i])
        circuit.measure(aq[i], cb[i])

    if add_barriers:
        circuit.barrier()

    # Z-stabilizer measurement: ancilla in |0⟩, CX(data→anc), measure.
    for i, support in enumerate(PATCH_Z_STABILIZERS):
        for site in support:
            circuit.cx(pq[site], aq[3 + i])
        circuit.measure(aq[3 + i], cb[3 + i])


# ---------------------------------------------------------------------------
# Logical measurement on the 11-site patch
# ---------------------------------------------------------------------------

def append_patch_logical_measurement(
    circuit: QuantumCircuit,
    basis: str = "Z",
    *,
    patch_qubits: Sequence,
    support: tuple[int, ...] | None = None,
    ancilla_qubit=None,
    classical_bits: Sequence,
    destructive: bool = True,
) -> None:
    """Append a logical measurement on the 11-site patch.

    Parameters
    ----------
    circuit
        Circuit being modified.
    basis
        Which logical to measure: ``"Z"`` for Z_L or ``"X"`` for X_L.
    patch_qubits
        Eleven qubits indexed by the 11-site patch labels.
    support
        Patch-site indices for the logical representative. Defaults to
        ``PATCH_DEFAULT_LOGICAL_Z`` or ``PATCH_DEFAULT_LOGICAL_X``.
    ancilla_qubit
        For non-destructive measurement, the ancilla qubit to use. Required
        when ``destructive=False``.
    classical_bits
        Classical bits for measurement results. For destructive: one per
        support qubit. For non-destructive: at least one bit.
    destructive
        If True (default), directly measure the data qubits in the
        appropriate basis. If False, use the ancilla for indirect measurement.
    """
    if len(patch_qubits) < 11:
        raise ValueError("patch_qubits must contain at least 11 qubits.")
    if support is None:
        support = PATCH_DEFAULT_LOGICAL_Z if basis == "Z" else PATCH_DEFAULT_LOGICAL_X

    pq = list(patch_qubits)
    cb = list(classical_bits)

    if destructive:
        if len(cb) < len(support):
            raise ValueError(
                f"classical_bits must have at least {len(support)} entries for destructive measurement."
            )
        if basis == "X":
            for site in support:
                circuit.h(pq[site])
        for i, site in enumerate(support):
            circuit.measure(pq[site], cb[i])
    else:
        if ancilla_qubit is None:
            raise ValueError("ancilla_qubit is required for non-destructive measurement.")
        if len(cb) < 1:
            raise ValueError("classical_bits must have at least 1 entry for non-destructive measurement.")
        if basis == "Z":
            for site in support:
                circuit.cx(pq[site], ancilla_qubit)
            circuit.measure(ancilla_qubit, cb[0])
        else:
            circuit.h(ancilla_qubit)
            for site in support:
                circuit.cx(ancilla_qubit, pq[site])
            circuit.h(ancilla_qubit)
            circuit.measure(ancilla_qubit, cb[0])