"""Stim continuation of a rotated surface patch after state transfer."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import stim

from msc.rotated_surface_code import helper_stim as helper

CircuitAppender = Callable[..., None]


def append_rotated_post_transfer(
    circuit: stim.Circuit,
    *,
    surface_qubits: Sequence[int],
    z_ancillas: Sequence[int],
    x_ancillas: Sequence[int],
    z_stabilizer_names: Sequence[str],
    x_stabilizer_names: Sequence[str],
    stabilizer_rounds: int,
    append_scheduled_stabilizer_round: CircuitAppender,
    logical_readout_basis: str | None,
    append_logical_measurement: CircuitAppender,
    z_reference_measurement_positions: Sequence[int] | None = None,
    x_reference_measurement_positions: Sequence[int] | None = None,
    observable_index: int | None = None,
    add_ticks: bool = True,
) -> None:
    """Append full rounds, optional change detectors, and logical readout.

    Supplying both reference-position sequences appends change detectors for
    every full stabilizer round. ``observable_index`` includes the destructive
    logical measurement parity as a Stim observable. Protocol-specific Pauli-
    frame measurements must be appended separately by the enclosing builder.
    """
    if stabilizer_rounds < 0:
        raise ValueError("stabilizer_rounds must be nonnegative.")
    if (z_reference_measurement_positions is None) != (x_reference_measurement_positions is None):
        raise ValueError("both X and Z stabilizer references must be supplied together.")

    first_round_measurement = circuit.num_measurements
    if stabilizer_rounds:
        round_circuit = stim.Circuit()
        append_scheduled_stabilizer_round(
            round_circuit,
            surface_qubits=surface_qubits,
            selected_stabilizer_names=tuple(z_stabilizer_names) + tuple(x_stabilizer_names),
            ancillas=tuple(z_ancillas) + tuple(x_ancillas),
            round_label="post-transfer round",
            reset_ancillas_initially=True,
            add_ticks=add_ticks,
        )
        if stabilizer_rounds == 1:
            circuit += round_circuit
        else:
            circuit.append(stim.CircuitRepeatBlock(stabilizer_rounds, round_circuit))

        # The rounds are placed as one (possibly REPEAT) block, so their change
        # detectors are annotated here by absolute record position rather than
        # inside each round: the first round compares against an external prior-
        # stage reference that does not fit a REPEAT-internal relative offset.
        if z_reference_measurement_positions is not None:
            if x_reference_measurement_positions is None:  # pragma: no cover - guarded above.
                raise RuntimeError("missing X stabilizer reference positions.")
            z_round_width = len(z_stabilizer_names)
            round_width = z_round_width + len(x_stabilizer_names)
            z_rounds: list[tuple[int, ...]] = []
            x_rounds: list[tuple[int, ...]] = []
            for round_index in range(stabilizer_rounds):
                round_start = first_round_measurement + round_index * round_width
                z_rounds.append(tuple(range(round_start, round_start + z_round_width)))
                x_rounds.append(
                    tuple(range(round_start + z_round_width, round_start + round_width))
                )
            helper.append_stabilizer_detectors(
                circuit, rounds=z_rounds, first_round_reference=z_reference_measurement_positions
            )
            helper.append_stabilizer_detectors(
                circuit, rounds=x_rounds, first_round_reference=x_reference_measurement_positions
            )

    if logical_readout_basis is None:
        return
    append_logical_measurement(
        circuit,
        logical_readout_basis,
        surface_qubits=surface_qubits,
        destructive=True,
        observable_index=observable_index,
    )
    if add_ticks:
        circuit.append("TICK", [])
