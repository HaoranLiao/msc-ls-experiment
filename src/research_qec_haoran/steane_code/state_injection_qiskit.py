from __future__ import annotations

from typing import Literal

from qiskit import QuantumCircuit
from qiskit.circuit import Qubit

from research_qec_haoran.utils.gates import apply_google_ty


InjectedStateName = Literal["0", "1", "+", "-", "i", "-i", "Y+", "Y-", "S", "T", "Tdg", "H"]


# Steane [[7,1,3]] stabilizers and logical operators for qubits 0–6.
X_STABILIZERS: tuple[tuple[int, ...], ...] = (
    (0, 2, 4, 6),
    (1, 2, 5, 6),
    (3, 4, 5, 6),
)

Z_STABILIZERS: tuple[tuple[int, ...], ...] = (
    (0, 2, 4, 6),
    (1, 2, 5, 6),
    (3, 4, 5, 6),
)

DEFAULT_LOGICAL_X: tuple[int, ...] = (0, 1, 2)
DEFAULT_LOGICAL_Z: tuple[int, ...] = (0, 1, 2)

# All weight-3 logical representatives (related by stabilizer multiplication).
LOGICAL_X_REPRESENTATIVES: tuple[tuple[int, ...], ...] = (
    (0, 1, 2),
    (0, 3, 4),
    (0, 5, 6),
    (1, 3, 5),
    (1, 4, 6),
    (2, 3, 6),
    (2, 4, 5),
)

LOGICAL_Z_REPRESENTATIVES: tuple[tuple[int, ...], ...] = (
    (0, 1, 2),
    (0, 3, 4),
    (0, 5, 6),
    (1, 3, 5),
    (1, 4, 6),
    (2, 3, 6),
    (2, 4, 5),
)


STEANE_ENCODER_CNOTS: tuple[tuple[int, int], ...] = (
    (3, 4),
    (0, 3),
    (1, 0),
    (4, 5),
    (4, 6),
    (0, 2),
    (2, 6),
    (2, 1),
    (1, 5),
    (0, 4),
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
        apply_google_ty(circuit, qubit)
        return

    raise ValueError(f"Unknown state name: {state!r}")


def build_steane_state_injection_encoder(
    injected_state: InjectedStateName = "H",
    *,
    add_barriers: bool = True,
) -> QuantumCircuit:
    """Build a Steane-code state-injection encoder.

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

    Notes
    -----
    The circuit implements the map

    ``|psi> |+> |+> |+> |0> |0> |0> -> |psi_L>``.
    """
    circuit = QuantumCircuit(7, name=f"steane_encode_{injected_state}")

    # Prepare the unencoded state on q0.
    prepare_physical_state(circuit, 0, injected_state)

    # Prepare the three Steane codeword-superposition variables u, v, w.
    circuit.h(1)
    circuit.h(2)
    circuit.h(3)

    if add_barriers:
        circuit.barrier()

    # CNOT encoder.
    for control_qubit, target_qubit in STEANE_ENCODER_CNOTS:
        circuit.cx(control_qubit, target_qubit)

    return circuit


def build_stabilizer_measurement_circuit(
    *,
    add_barriers: bool = True,
) -> QuantumCircuit:
    """Build a non-destructive stabilizer measurement circuit for the Steane code.

    Uses 6 ancilla qubits (one per stabilizer) to measure all stabilizers
    without disturbing the encoded state.

    Layout: qubits 0–6 are data, qubits 7–9 are X-stabilizer ancillas,
    qubits 10–12 are Z-stabilizer ancillas.

    Parameters
    ----------
    add_barriers
        Whether to insert barriers between X and Z stabilizer rounds.

    Returns
    -------
    QuantumCircuit
        A 13-qubit, 6-clbit circuit. Classical bits 0–2 hold X-stabilizer
        outcomes, bits 3–5 hold Z-stabilizer outcomes.
    """
    n_data = 7
    n_ancilla = 6
    circuit = QuantumCircuit(n_data + n_ancilla, n_ancilla, name="steane_stab_meas")

    # X-stabilizer measurement: ancilla prepared in |+⟩, CX from ancilla to
    # each data qubit in the support, then measure ancilla in X basis.
    for i, support in enumerate(X_STABILIZERS):
        anc = n_data + i
        circuit.h(anc)
        for data_q in support:
            circuit.cx(anc, data_q)
        circuit.h(anc)
        circuit.measure(anc, i)

    if add_barriers:
        circuit.barrier()

    # Z-stabilizer measurement: ancilla prepared in |0⟩, CX from each data
    # qubit in the support to ancilla, then measure ancilla in Z basis.
    for i, support in enumerate(Z_STABILIZERS):
        anc = n_data + 3 + i
        for data_q in support:
            circuit.cx(data_q, anc)
        circuit.measure(anc, 3 + i)

    return circuit


def build_logical_measurement_circuit(
    basis: Literal["Z", "X"] = "Z",
    *,
    support: tuple[int, ...] | None = None,
    destructive: bool = True,
) -> QuantumCircuit:
    """Build a circuit to measure the logical operator of the Steane code.

    Parameters
    ----------
    basis
        Which logical to measure: ``"Z"`` for Z_L or ``"X"`` for X_L.
    support
        Qubit indices for the logical representative. Defaults to
        ``DEFAULT_LOGICAL_Z`` or ``DEFAULT_LOGICAL_X``.
    destructive
        If True (default), directly measure the data qubits in the
        appropriate basis. If False, use an ancilla for indirect measurement.

    Returns
    -------
    QuantumCircuit
        The measurement circuit. For destructive measurement the circuit has
        7 data qubits and classical bits for the logical-support qubits.
        For non-destructive measurement it has 7 data qubits + 1 ancilla
        and 1 classical bit.
    """
    n_data = 7
    if support is None:
        support = DEFAULT_LOGICAL_Z if basis == "Z" else DEFAULT_LOGICAL_X

    if destructive:
        circuit = QuantumCircuit(n_data, len(support), name=f"steane_meas_{basis}_L_dest")
        if basis == "X":
            for i, q in enumerate(support):
                circuit.h(q)
        for i, q in enumerate(support):
            circuit.measure(q, i)
    else:
        # Non-destructive: one ancilla qubit.
        circuit = QuantumCircuit(n_data + 1, 1, name=f"steane_meas_{basis}_L_nd")
        anc = n_data
        if basis == "Z":
            # Ancilla in |0⟩, CX from each support qubit to ancilla.
            for q in support:
                circuit.cx(q, anc)
            circuit.measure(anc, 0)
        else:
            # Ancilla in |+⟩, CX from ancilla to each support qubit.
            circuit.h(anc)
            for q in support:
                circuit.cx(anc, q)
            circuit.h(anc)
            circuit.measure(anc, 0)

    return circuit


if __name__ == "__main__":
    steane_t_encoder = build_steane_state_injection_encoder("T")
    print(steane_t_encoder.draw(output="text"))

    steane_h_encoder = build_steane_state_injection_encoder("H")
    print(steane_h_encoder.draw(output="text"))

    stabilizer_meas = build_stabilizer_measurement_circuit()
    print(stabilizer_meas.draw(output="text"))

    logical_meas = build_logical_measurement_circuit(basis="Z")
    print(logical_meas.draw(output="text"))

