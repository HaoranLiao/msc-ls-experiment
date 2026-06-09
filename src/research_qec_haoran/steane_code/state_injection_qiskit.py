from __future__ import annotations

from typing import Literal

from qiskit import QuantumCircuit
from qiskit.circuit import Qubit

from research_qec_haoran.utils.gates import apply_google_ty


InjectedStateName = Literal["0", "1", "+", "-", "i", "-i", "Y+", "Y-", "S", "T", "Tdg", "H"]


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


def build_steane_state_injection_encoder_qiskit(
    injected_state: InjectedStateName = "T",
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


if __name__ == "__main__":
    steane_t_encoder = build_steane_state_injection_encoder_qiskit("T")
    print(steane_t_encoder.draw(output="text"))

    steane_h_encoder = build_steane_state_injection_encoder_qiskit("H")
    print(steane_h_encoder.draw(output="text"))