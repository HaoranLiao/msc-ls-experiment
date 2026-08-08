"""Optional rotated-surface stabilization and logical readout after transfer."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from qiskit import QuantumCircuit

CircuitAppender = Callable[..., None]


def append_rotated_post_transfer(
    circuit: QuantumCircuit,
    *,
    surface_qubits: Sequence[object],
    z_ancillas: Sequence[object],
    x_ancillas: Sequence[object],
    z_stabilizer_names: Sequence[str],
    x_stabilizer_names: Sequence[str],
    z_classical_bits: Sequence[object],
    x_classical_bits: Sequence[object],
    stabilizer_rounds: int,
    append_scheduled_stabilizer_round: CircuitAppender,
    logical_readout_basis: str | None,
    logical_classical_bits: Sequence[object],
    append_logical_measurement: CircuitAppender,
    add_barriers: bool = True,
) -> None:
    """Append optional four-step scheduled stabilizer rounds and logical readout.

    ``stabilizer_rounds=0`` preserves the transfer-only behavior. When rounds
    are requested, every surface X and Z stabilizer is measured in each round
    before the optional destructive logical measurement.
    """
    if stabilizer_rounds < 0:
        raise ValueError("stabilizer_rounds must be nonnegative.")

    z_bits = list(z_classical_bits)
    x_bits = list(x_classical_bits)
    required_z_bits = stabilizer_rounds * len(z_ancillas)
    required_x_bits = stabilizer_rounds * len(x_ancillas)
    if len(z_bits) < required_z_bits:
        raise ValueError(f"z_classical_bits must contain at least {required_z_bits} bits.")
    if len(x_bits) < required_x_bits:
        raise ValueError(f"x_classical_bits must contain at least {required_x_bits} bits.")

    for round_index in range(stabilizer_rounds):
        z_start = round_index * len(z_ancillas)
        x_start = round_index * len(x_ancillas)
        append_scheduled_stabilizer_round(
            circuit,
            surface_qubits=surface_qubits,
            selected_stabilizer_names=tuple(z_stabilizer_names) + tuple(x_stabilizer_names),
            ancillas=tuple(z_ancillas) + tuple(x_ancillas),
            classical_bits=tuple(z_bits[z_start : z_start + len(z_ancillas)])
            + tuple(x_bits[x_start : x_start + len(x_ancillas)]),
            round_label=f"post-transfer round {round_index + 1}",
            reset_ancillas_initially=True,
            add_barriers=False,
        )
        if add_barriers:
            circuit.barrier(label=f"post-transfer scheduled stabilizer round {round_index + 1}")

    if logical_readout_basis is None:
        return
    append_logical_measurement(
        circuit,
        logical_readout_basis,
        surface_qubits=surface_qubits,
        classical_bits=logical_classical_bits,
        destructive=True,
    )
    if add_barriers:
        circuit.barrier(label=f"destructive logical {logical_readout_basis}")
