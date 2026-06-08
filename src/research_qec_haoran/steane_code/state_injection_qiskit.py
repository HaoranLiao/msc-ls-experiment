from __future__ import annotations

from collections.abc import Sequence
from math import cos, pi, sin, sqrt
from typing import Literal

import numpy as np
from qiskit import QuantumCircuit

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


def _prepare_injected_state(
    circuit: QuantumCircuit,
    qubit_index: int,
    injected_state: InjectedStateName | Sequence[complex],
) -> None:
    """Prepare the single-qubit state to be encoded.

    Parameters
    ----------
    circuit
        Circuit being built.
    qubit_index
        Qubit on which the unencoded input state is prepared.
    injected_state
        Either a named state or a length-2 complex vector ``[alpha, beta]``.
        The encoder maps ``alpha |0> + beta |1>`` to
        ``alpha |0_L> + beta |1_L>``.

    Notes
    -----
    Named non-stabilizer states:

    ``"T"``
        ``(|0> + exp(i pi/4) |1>) / sqrt(2)``.

    ``"H"``
        The +1 eigenstate of the Hadamard operator,
        ``cos(pi/8)|0> + sin(pi/8)|1>``.
    """
    if isinstance(injected_state, str):
        state = injected_state.upper()

        if state == "0":
            return

        if state == "1":
            circuit.x(qubit_index)
            return

        if state == "+":
            circuit.h(qubit_index)
            return

        if state == "-":
            circuit.x(qubit_index)
            circuit.h(qubit_index)
            return

        if state in {"I", "Y+", "S"}:
            # |Y+> = S|+> = (|0> + i|1>) / sqrt(2).
            # This is the +1 eigenstate of Pauli Y.
            circuit.h(qubit_index)
            circuit.s(qubit_index)
            return

        if state in {"-I", "Y-"}:
            # |Y-> = S^\dagger |+> = (|0> - i|1>) / sqrt(2).
            circuit.h(qubit_index)
            circuit.sdg(qubit_index)
            return

        if state == "T":
            # |T> = T|+>.
            circuit.h(qubit_index)
            circuit.t(qubit_index)
            return

        if state == "TDG":
            circuit.h(qubit_index)
            circuit.tdg(qubit_index)
            return

        if state == "H":
            # +1 eigenstate of H:
            # cos(pi/8)|0> + sin(pi/8)|1> = T_y|0>
            apply_google_ty(circuit, qubit_index)
            return

        raise ValueError(f"Unknown injected state name: {injected_state!r}")

    if len(injected_state) != 2:
        raise ValueError("Custom injected_state must be a length-2 vector [alpha, beta].")

    alpha, beta = complex(injected_state[0]), complex(injected_state[1])
    norm = sqrt(abs(alpha) ** 2 + abs(beta) ** 2)
    if norm == 0:
        raise ValueError("Custom injected_state cannot be the zero vector.")

    circuit.initialize([alpha / norm, beta / norm], [qubit_index])


def build_steane_state_injection_encoder_qiskit(
    injected_state: InjectedStateName | Sequence[complex] = "T",
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

    With ``injected_state="T"``, this prepares

    ``|T_L> = (|0_L> + exp(i*pi/4)|1_L>) / sqrt(2)``.
    """
    circuit = QuantumCircuit(7, name=f"steane_encode_{injected_state}")

    # Prepare the unencoded state on q0.
    _prepare_injected_state(circuit, 0, injected_state)

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