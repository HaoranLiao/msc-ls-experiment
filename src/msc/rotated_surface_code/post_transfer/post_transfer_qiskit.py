"""Append full rotated-surface stabilizer rounds in Qiskit."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from qiskit import QuantumCircuit

CircuitAppender = Callable[..., None]


def append_rotated_full_stabilizer_rounds(
    circuit: QuantumCircuit,
    *,
    surface_qubits: Sequence[object],
    z_ancillas: Sequence[object],
    x_ancillas: Sequence[object],
    z_stabilizer_names: Sequence[str],
    x_stabilizer_names: Sequence[str],
    z_bits: Sequence[object],
    x_bits: Sequence[object],
    stabilizer_rounds: int,
    append_scheduled_stabilizer_round: CircuitAppender,
    add_barriers: bool = True,
) -> None:
    """Append full rounds into caller-owned round-major Z/X bits.

    This helper owns only the repeated rotated-patch schedule. Destructive
    readout and its classical register belong to the enclosing protocol.
    """
    if stabilizer_rounds < 0:
        raise ValueError("stabilizer_rounds must be nonnegative.")
    if stabilizer_rounds == 0:
        return

    z_round_width = len(z_stabilizer_names)
    x_round_width = len(x_stabilizer_names)
    if len(z_ancillas) < z_round_width:
        raise ValueError(f"z_ancillas must contain at least {z_round_width} qubits.")
    if len(x_ancillas) < x_round_width:
        raise ValueError(f"x_ancillas must contain at least {x_round_width} qubits.")
    required_z_bits = stabilizer_rounds * z_round_width
    required_x_bits = stabilizer_rounds * x_round_width
    if len(z_bits) < required_z_bits:
        raise ValueError(f"z_bits must contain at least {required_z_bits} bits.")
    if len(x_bits) < required_x_bits:
        raise ValueError(f"x_bits must contain at least {required_x_bits} bits.")

    stabilizer_names = tuple(z_stabilizer_names) + tuple(x_stabilizer_names)
    ancillas = tuple(z_ancillas[:z_round_width]) + tuple(x_ancillas[:x_round_width])
    for round_index in range(stabilizer_rounds):
        z_start = round_index * z_round_width
        x_start = round_index * x_round_width
        append_scheduled_stabilizer_round(
            circuit,
            surface_qubits=surface_qubits,
            selected_stabilizer_names=stabilizer_names,
            ancillas=ancillas,
            bits=tuple(z_bits[z_start : z_start + z_round_width])
            + tuple(x_bits[x_start : x_start + x_round_width]),
            round_label=f"post-transfer round {round_index + 1}",
            reset_ancillas_initially=True,
            add_barriers=False,
        )
        if add_barriers:
            circuit.barrier(label=f"post-transfer scheduled stabilizer round {round_index + 1}")
