"""Stim logical-Y Steane-patch kickback tomography (KT) experiment.

Clifford model of :mod:`kickback_tomography_qiskit`. The KT circuit reuses the
Stim injection encoder and logical-Y cultivation round with ``|S>`` as its
stabilizer proxy. Summaries report the native S-state logical error rate without
conversion to a magic-state estimate.

The public builders parallel their Qiskit counterparts for both the 11-site
and 14-site patches. Shared bit-layout and summary dataclasses store Stim
measurement-record indices, which are zero-based in append order.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

import numpy as np
import stim

from msc.steane_code.kickback_tomography_common import (
    KT_DEFAULT_QED_CYCLE_INDICES,
    FinalStabilizerMode,
    KickbackCultivationRoundBits,
    KickbackTomographyBitLayout,
    KickbackTomographyPatchSites,
    KickbackTomographyPatchSpec,
    KickbackTomographySummary,
    cultivation_round_bits_from_positions,
    final_stabilizer_bit_count,
    normalize_qed_cycle_indices,
    resolve_kickback_tomography_patch_spec,
)
from msc.steane_code.magic_state_cultivation_stim import (
    append_11_site_patch_y_projection_round,
    append_14_site_patch_projection_round,
)
from msc.steane_code.state_injection_common import (
    PatchStabilizerBasis,
    select_14_site_patch_superdense_measurement_layers,
)
from msc.steane_code.state_injection_stim import (
    DEFAULT_STIM_14_SITE_PATCH_QUBITS,
    StimInjectedStateName,
    append_11_site_patch_rosenfeld_encoder,
    append_11_site_patch_stabilizer_measurement,
    append_14_site_patch_abstract_routed_encoder,
    append_14_site_patch_superdense_stabilizer_measurement,
)
from msc.utils import circuit_stim


def build_kickback_tomography_experiment(
    injected_state: StimInjectedStateName = "S",
    *,
    patch_sites: KickbackTomographyPatchSites = 14,
    cultivated_state: Literal["S"] = "S",
    include_preparation_cultivation: bool = True,
    measure_encoder_flags: bool = False,
    qed_cycle_indices: Iterable[int] = KT_DEFAULT_QED_CYCLE_INDICES,
    final_stabilizer_basis: FinalStabilizerMode = "X",
    add_ticks: bool = True,
) -> tuple[stim.Circuit, KickbackTomographyBitLayout]:
    """Build a logical-Y KT experiment using the Clifford ``|S>`` proxy.

    The shared protocol order is

    ``encode -> QED1 -> [prep cultivation] -> QED2 -> KT measure -> QED3
    -> KT check -> QED4 -> final stabilizer check``.

    This parallels the Qiskit protocol schedule, but Stim uses logical Y and
    accepts only ``cultivated_state="S"``. This is an explicit Clifford proxy,
    not a simulation of a non-Clifford magic state.

    The 11-site path uses the Rosenfeld encoder, and the 14-site path uses the
    abstract-routed encoder. When ``measure_encoder_flags`` is true, their
    available flags are measured after encoding and postselected to zero.

    Both patches reset and re-measure the fold root after unfolding, and the
    repeated-root check accepts either logical branch when the two reads agree.

    ``qed_cycle_indices`` uses the paper's one-based numbering and defaults to
    ``(1, 3)``. An empty tuple omits the QED cycles. The final stabilizer basis
    may be ``"X"``, ``"Z"``, ``"BOTH"``, or ``"NONE"``. A 14-site X- or
    Z-only final block also records and postselects the three default
    Bell-partner flags.
    """
    if cultivated_state != "S":
        raise ValueError("cultivated_state must be 'S' in Stim.")
    spec = resolve_kickback_tomography_patch_spec(patch_sites)
    qed_indices = normalize_qed_cycle_indices(qed_cycle_indices)

    circuit = stim.Circuit()
    _append_patch_encoder(
        circuit,
        spec,
        injected_state=injected_state,
        measure_encoder_flags=measure_encoder_flags,
        add_ticks=add_ticks,
    )
    record_count = spec.encoder_flag_count if measure_encoder_flags else 0
    encoder_flag_positions = tuple(range(record_count))
    qed_positions: dict[int, tuple[int, ...]] = {}

    def add_qed_cycle(index: int) -> None:
        nonlocal record_count
        _append_patch_stabilizer_measurement(
            circuit,
            spec,
            basis="BOTH",
            add_ticks=add_ticks,
        )
        qed_positions[index] = tuple(range(record_count, record_count + spec.qed_measurement_count))
        record_count += spec.qed_measurement_count

    def add_cultivation_round(
        *,
        running_kt: bool = False,
    ) -> KickbackCultivationRoundBits:
        nonlocal record_count
        positions = _append_patch_projection_round(
            circuit,
            spec,
            add_ticks=add_ticks,
            running_kt=running_kt,
        )
        bits = cultivation_round_bits_from_positions(
            positions,
            root_check_index=spec.cultivation_root_check_index,
        )
        record_count += spec.cultivation_measurement_count
        return bits

    if 1 in qed_indices:
        add_qed_cycle(1)

    preparation_bits = add_cultivation_round() if include_preparation_cultivation else None

    if 2 in qed_indices:
        add_qed_cycle(2)

    measurement_bits = add_cultivation_round(running_kt=True)

    if 3 in qed_indices:
        add_qed_cycle(3)

    check_bits = add_cultivation_round(running_kt=True)

    if 4 in qed_indices:
        add_qed_cycle(4)

    final_bit_count = final_stabilizer_bit_count(final_stabilizer_basis)
    final_positions: tuple[int, ...] = ()
    final_bell_partner_positions: tuple[int, ...] = ()
    final_d2_copy_flag: int | None = None
    if final_stabilizer_basis in ("X", "Z", "BOTH"):
        _append_patch_stabilizer_measurement(
            circuit,
            spec,
            basis=final_stabilizer_basis,
            add_ticks=add_ticks,
        )
        if spec.patch_sites == 14:
            _, readouts = select_14_site_patch_superdense_measurement_layers(
                final_stabilizer_basis,
                None,
            )
            final_positions = tuple(
                record_count + offset
                for offset, (_, _, readout_role) in enumerate(readouts)
                if readout_role == "STABILIZER"
            )
            final_bell_partner_positions = tuple(
                record_count + offset
                for offset, (_, _, readout_role) in enumerate(readouts)
                if readout_role == "BELL_PARTNER_FLAG"
            )
            final_d2_copy_flag = next(
                (
                    record_count + offset
                    for offset, (_, _, readout_role) in enumerate(readouts)
                    if readout_role == "D2_COPY_FLAG"
                ),
                None,
            )
            final_record_width = len(readouts)
        elif spec.patch_sites == 11:
            final_record_width = final_bit_count
            final_positions = tuple(range(record_count, record_count + final_record_width))
        else:  # pragma: no cover - resolved patch specs support only 11/14 sites.
            raise ValueError(f"Unsupported patch size: {spec.patch_sites}.")
        record_count += final_record_width

    bit_layout = KickbackTomographyBitLayout(
        preparation_round=preparation_bits,
        tomography_measure_round=measurement_bits,
        tomography_check_round=check_bits,
        qed_cycles=tuple(qed_positions[index] for index in qed_indices),
        qed_cycle_indices=qed_indices,
        final_stabilizers=final_positions,
        final_bell_partner_flags=final_bell_partner_positions,
        final_d2_copy_flag=final_d2_copy_flag,
        encoder_flags=encoder_flag_positions,
    )
    if circuit.num_measurements != record_count:
        raise RuntimeError("Kickback-tomography measurement bit layout does not match the circuit.")
    circuit_stim.append_observable_include(
        circuit,
        measurement_positions=(bit_layout.tomography_measure_round.folded_root,),
    )
    return circuit, bit_layout


def _append_patch_encoder(
    circuit: stim.Circuit,
    spec: KickbackTomographyPatchSpec,
    *,
    injected_state: StimInjectedStateName,
    measure_encoder_flags: bool,
    add_ticks: bool,
) -> None:
    """Append the selected patch encoder and optional flag measurements."""
    if spec.patch_sites == 11:
        append_11_site_patch_rosenfeld_encoder(
            circuit,
            injected_state=injected_state,
            measure_flags=measure_encoder_flags,
            add_ticks=add_ticks,
        )
    elif spec.patch_sites == 14:
        append_14_site_patch_abstract_routed_encoder(
            circuit,
            injected_state=injected_state,
            measure_flags=measure_encoder_flags,
            add_ticks=add_ticks,
        )
    else:
        raise ValueError(f"Unsupported KT patch_sites: {spec.patch_sites}.")


def _append_patch_stabilizer_measurement(
    circuit: stim.Circuit,
    spec: KickbackTomographyPatchSpec,
    *,
    basis: PatchStabilizerBasis,
    add_ticks: bool,
) -> None:
    """Append one full-QED or final stabilizer measurement on the patch."""
    if spec.patch_sites == 11:
        append_11_site_patch_stabilizer_measurement(
            circuit,
            basis=basis,
            add_ticks=add_ticks,
        )
    elif spec.patch_sites == 14:
        append_14_site_patch_superdense_stabilizer_measurement(
            circuit,
            patch_qubits=DEFAULT_STIM_14_SITE_PATCH_QUBITS,
            basis=basis,
            add_ticks=add_ticks,
        )
    else:
        raise ValueError(f"Unsupported KT patch_sites: {spec.patch_sites}.")


def _append_patch_projection_round(
    circuit: stim.Circuit,
    spec: KickbackTomographyPatchSpec,
    *,
    add_ticks: bool,
    running_kt: bool,
) -> tuple[int, ...]:
    """Append one detector-annotated logical-Y proxy measurement."""
    if spec.patch_sites == 11:
        return append_11_site_patch_y_projection_round(
            circuit,
            add_ticks=add_ticks,
            running_kt=running_kt,
        )
    elif spec.patch_sites == 14:
        return append_14_site_patch_projection_round(
            circuit,
            patch_qubits=DEFAULT_STIM_14_SITE_PATCH_QUBITS,
            projection_axis="Y",
            add_ticks=add_ticks,
            running_kt=running_kt,
        )
    else:
        raise ValueError(f"Unsupported KT patch_sites: {spec.patch_sites}.")


# =============================================================================
# Shot parsing and KT estimators
# =============================================================================


def analyze_kickback_tomography_shots(
    shots: np.ndarray,
    bit_layout: KickbackTomographyBitLayout,
    *,
    retain_measure_check_disagreement: bool = True,
) -> KickbackTomographySummary:
    """Analyze Stim samples by postselecting and classifying the two KT logical readouts.

    ``shots`` is a boolean array of shape ``(num_shots, num_measurements)`` from
    ``circuit.compile_sampler().sample(...)``. Fixed-zero detector columns
    reject first. Each folded root must agree with its repeated-root check (the
    root re-measured after unfolding). Inter-round root disagreements are
    retained by default and reported separately; agreeing roots classify logical
    success (zero) or logical error (one).
    """
    shots = np.asarray(shots, dtype=bool)
    if shots.ndim != 2:
        raise ValueError("shots must be a two-dimensional array.")

    measurement_round = bit_layout.tomography_measure_round
    check_round = bit_layout.tomography_check_round
    required_positions = (
        *bit_layout.detector_positions,
        *measurement_round.all_positions,
        *check_round.all_positions,
    )
    if any(position < 0 or position >= shots.shape[1] for position in required_positions):
        raise ValueError("bit layout contains a position outside the shots array.")

    total_shots = int(shots.shape[0])

    # Fixed-zero encoder, preparation, QED, KT-flag, and final-check columns.
    retained = np.ones(total_shots, dtype=bool)
    if bit_layout.detector_positions:
        retained &= ~shots[:, bit_layout.detector_positions].any(axis=1)

    # Each repeated-root check must match its round's root.
    measurement_root = shots[:, measurement_round.folded_root]
    check_root = shots[:, check_round.folded_root]
    if measurement_round.folded_root_check is not None:
        retained &= shots[:, measurement_round.folded_root_check] == measurement_root
    if check_round.folded_root_check is not None:
        retained &= shots[:, check_round.folded_root_check] == check_root

    # Inter-round disagreement is diagnostic unless explicitly postselected.
    agree = measurement_root == check_root
    if not retain_measure_check_disagreement:
        retained &= agree

    # Only agreeing retained roots enter logical success/error numerators.
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
