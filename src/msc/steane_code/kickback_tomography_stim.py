"""Stim logical-Y Steane-patch kickback tomography (KT) experiment.

Clifford model of :mod:`kickback_tomography_qiskit`. The KT circuit reuses the
Stim injection encoder and logical-Y cultivation round. Because this is
logical-Y cultivation (proxy for logical ``H_XY``), the circuit prepares and
verifies ``|S>`` instead of ``|T>``. All returned summaries report
the native ``S``-state logical error rate; no conversion to a ``T``-state
estimate is applied.

The public builder and experiment container intentionally use the same names as
their Qiskit counterparts. The module name identifies the framework. Shared
layout and summary dataclasses store Stim measurement-record indices, which are
zero-based in append order.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import stim

from msc.steane_code.kickback_tomography_qiskit import (
    KT_DEFAULT_QED_CYCLE_INDICES,
    KT_STABILIZER_BITS_PER_CYCLE,
    FinalStabilizerMode,
    KickbackCultivationRoundBits,
    KickbackTomographyBitLayout,
    KickbackTomographySummary,
    cultivation_round_bits_from_positions,
    final_stabilizer_bit_count,
    normalize_qed_cycle_indices,
)
from msc.steane_code.magic_state_cultivation_qiskit import (
    CULTIVATION_MEASUREMENTS_PER_ROUND,
)
from msc.steane_code.magic_state_cultivation_stim import (
    append_y_projection_round,
)
from msc.steane_code.state_injection_stim import (
    DEFAULT_STIM_INJECTED_STATE,
    PATCH_11_SITE_HELPER_SITES,
    StimInjectedStateName,
    append_11_site_patch_rosenfeld_encoder,
    append_11_site_patch_stabilizer_measurement,
)
from msc.utils import circuit_stim


@dataclass(frozen=True)
class KickbackTomographyExperiment:
    """A Stim KT circuit together with its measurement-record bit layout."""

    circuit: stim.Circuit
    layout: KickbackTomographyBitLayout


def build_kickback_tomography_experiment(
    injected_state: StimInjectedStateName = DEFAULT_STIM_INJECTED_STATE,
    *,
    include_preparation_cultivation: bool = True,
    measure_encoder_flags: bool = False,
    qed_cycle_indices: Iterable[int] = KT_DEFAULT_QED_CYCLE_INDICES,
    final_stabilizer_basis: FinalStabilizerMode = "X",
    add_ticks: bool = True,
) -> KickbackTomographyExperiment:
    """Build a Clifford (Stim) Steane-patch kickback tomography experiment.

    The protocol order is

    ``encode -> QED1 -> [prep cultivation] -> QED2 -> KT measure -> QED3
    -> KT check -> QED4 -> final stabilizer check``.

    This parallels :func:`kickback_tomography_qiskit.build_kickback_tomography_experiment`
    with logical-Y cultivation (proxy for logical ``H_XY``).
    Every cultivation and KT round therefore measures the logical-Y axis.

    When ``measure_encoder_flags`` is true, the four Rosenfeld-encoder flag
    qubits are measured immediately after encoding, before any QED cycle. KT
    postselection then requires each flag to be zero, flagging encoding faults.

    ``qed_cycle_indices`` uses the paper's one-based numbering and defaults to
    ``(1, 3)``. An empty tuple omits the QED cycles. The final stabilizer basis
    may be ``"X"``, ``"Z"``, ``"BOTH"``, or ``"NONE"``.
    """
    qed_indices = normalize_qed_cycle_indices(qed_cycle_indices)

    circuit = stim.Circuit()
    record_count = 0
    qed_positions: dict[int, tuple[int, ...]] = {}

    def add_qed_cycle(index: int) -> None:
        nonlocal record_count
        append_11_site_patch_stabilizer_measurement(
            circuit,
            add_ticks=add_ticks,
        )
        qed_positions[index] = tuple(
            range(record_count, record_count + KT_STABILIZER_BITS_PER_CYCLE)
        )
        record_count += KT_STABILIZER_BITS_PER_CYCLE

    def add_cultivation_round(
        *,
        tomography: bool = False,
    ) -> KickbackCultivationRoundBits:
        nonlocal record_count
        append_y_projection_round(
            circuit,
            add_ticks=add_ticks,
            pin_folded_root=not tomography,
        )
        positions = tuple(
            range(
                record_count,
                record_count + CULTIVATION_MEASUREMENTS_PER_ROUND,
            )
        )
        bits = cultivation_round_bits_from_positions(positions)
        record_count += CULTIVATION_MEASUREMENTS_PER_ROUND
        return bits

    append_11_site_patch_rosenfeld_encoder(
        circuit,
        injected_state=injected_state,
        measure_flags=measure_encoder_flags,
        add_ticks=add_ticks,
    )
    encoder_flag_positions: tuple[int, ...] = ()
    if measure_encoder_flags:
        encoder_flag_positions = tuple(
            range(record_count, record_count + len(PATCH_11_SITE_HELPER_SITES))
        )
        record_count += len(PATCH_11_SITE_HELPER_SITES)

    if 1 in qed_indices:
        add_qed_cycle(1)

    preparation_bits = add_cultivation_round() if include_preparation_cultivation else None

    if 2 in qed_indices:
        add_qed_cycle(2)

    measurement_bits = add_cultivation_round(tomography=True)

    if 3 in qed_indices:
        add_qed_cycle(3)

    check_bits = add_cultivation_round(tomography=True)

    if 4 in qed_indices:
        add_qed_cycle(4)

    final_bit_count = final_stabilizer_bit_count(final_stabilizer_basis)
    final_positions: tuple[int, ...] = ()
    if final_bit_count:
        append_11_site_patch_stabilizer_measurement(
            circuit,
            basis=final_stabilizer_basis,
            add_ticks=add_ticks,
        )
        final_positions = tuple(range(record_count, record_count + final_bit_count))
        record_count += final_bit_count

    layout = KickbackTomographyBitLayout(
        preparation_round=preparation_bits,
        tomography_measure_round=measurement_bits,
        tomography_check_round=check_bits,
        qed_cycles=tuple(qed_positions[index] for index in qed_indices),
        qed_cycle_indices=qed_indices,
        final_stabilizers=final_positions,
        encoder_flags=encoder_flag_positions,
    )
    if circuit.num_measurements != record_count:
        raise RuntimeError("Kickback-tomography measurement layout does not match the circuit.")
    circuit_stim.append_observable_include(
        circuit,
        measurement_positions=(layout.tomography_measure_round.folded_root,),
    )
    return KickbackTomographyExperiment(circuit=circuit, layout=layout)


# =============================================================================
# Shot parsing and KT estimators
# =============================================================================


def _required_layout_positions(
    layout: KickbackTomographyBitLayout,
) -> tuple[int, ...]:
    """Return every measurement-record position read during analysis."""
    measurement_round = layout.tomography_measure_round
    check_round = layout.tomography_check_round
    return (
        *layout.detector_positions,
        measurement_round.folded_root,
        measurement_round.folded_root_double_check,
        check_round.folded_root,
        check_round.folded_root_double_check,
    )


def _validate_shots(
    shots: np.ndarray,
    layout: KickbackTomographyBitLayout,
) -> None:
    """Validate a sampled-shot array against a KT bit layout."""
    if shots.ndim != 2:
        raise ValueError("shots must be a two-dimensional array.")

    positions = _required_layout_positions(layout)
    if any(position < 0 or position >= shots.shape[1] for position in positions):
        raise ValueError("layout contains a position outside the shots array.")


def _retained_shot_mask(
    shots: np.ndarray,
    layout: KickbackTomographyBitLayout,
    *,
    retain_measure_check_disagreement: bool,
) -> np.ndarray:
    """Return the KT postselection mask for validated sampled shots."""
    detector_positions = layout.detector_positions
    retained = np.ones(shots.shape[0], dtype=bool)
    if detector_positions:
        retained &= ~shots[:, detector_positions].any(axis=1)

    measurement_round = layout.tomography_measure_round
    check_round = layout.tomography_check_round
    measurement_root = shots[:, measurement_round.folded_root]
    check_root = shots[:, check_round.folded_root]

    retained &= shots[:, measurement_round.folded_root_double_check] == measurement_root
    retained &= shots[:, check_round.folded_root_double_check] == check_root
    if not retain_measure_check_disagreement:
        retained &= measurement_root == check_root
    return retained


def kickback_tomography_shot_is_retained(
    shot: np.ndarray,
    layout: KickbackTomographyBitLayout,
    *,
    retain_measure_check_disagreement: bool = True,
) -> bool:
    """Return whether one Stim KT shot passes detector postselection."""
    shot_array = np.asarray(shot, dtype=bool)
    if shot_array.ndim != 1:
        raise ValueError("shot must be a one-dimensional array.")
    shots = shot_array[np.newaxis, :]
    _validate_shots(shots, layout)
    retained = _retained_shot_mask(
        shots,
        layout,
        retain_measure_check_disagreement=retain_measure_check_disagreement,
    )
    return bool(retained[0])


def summarize_kickback_tomography_shots(
    shots: np.ndarray,
    layout: KickbackTomographyBitLayout,
    *,
    retain_measure_check_disagreement: bool = True,
) -> KickbackTomographySummary:
    """Summarize KT retainment and logical error from a sampled shot array.

    ``shots`` is a boolean array of shape ``(num_shots, num_measurements)`` from
    ``circuit.compile_sampler().sample(...)``. The postselection and logical
    classification exactly mirror :func:`summarize_kickback_tomography_counts`,
    but read Stim measurement-record indices directly instead of parsing Qiskit
    bitstrings.
    """
    shots = np.asarray(shots, dtype=bool)
    _validate_shots(shots, layout)
    total_shots = int(shots.shape[0])

    measurement_round = layout.tomography_measure_round
    check_round = layout.tomography_check_round
    measurement_root = shots[:, measurement_round.folded_root]
    check_root = shots[:, check_round.folded_root]
    retained = _retained_shot_mask(
        shots,
        layout,
        retain_measure_check_disagreement=retain_measure_check_disagreement,
    )

    agree = measurement_root == check_root
    disagreement = retained & ~agree
    logical_error = retained & agree & measurement_root
    logical_success = retained & agree & ~measurement_root

    retained_shots = int(retained.sum())
    return KickbackTomographySummary(
        total_shots=total_shots,
        retained_shots=retained_shots,
        rejected_shots=total_shots - retained_shots,
        logical_success_shots=int(logical_success.sum()),
        logical_error_shots=int(logical_error.sum()),
        measure_check_disagreement_shots=int(disagreement.sum()),
    )
