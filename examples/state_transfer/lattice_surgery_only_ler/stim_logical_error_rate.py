"""Stim logical-error analysis of the noisy lattice-surgery stage alone.

The Steane preparation and QED prefix is ideal. Uniform operation noise covers
surface ``|+>`` preparation and its initial X/Z round, the three merge rounds,
Steane-X demolition and ``m_ab``, post-transfer stabilization, and destructive
surface readout. ``TICK`` intervals add no idle errors. The implementation keeps
the repository's serialized transfer schedule instead of Hirano's interleaved
Fig. 9 schedule.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import cast

import stim

from msc.noise.fault_injection_stim import add_circuit_depolarizing_noise
from msc.noise.fault_tolerance_stim import (
    DetectorErrorModelShotAnalyzer,
    LogicalErrorEstimate,
    sample_detector_logical_error_probability,
)
from msc.state_transfer import common
from msc.state_transfer import helper_stim as transfer_helper
from msc.state_transfer import to_d5_rotated_stim as transfer
from msc.steane_code.state_injection_stim import StimInjectedStateName


@dataclass(frozen=True)
class LogicalErrorExperiment:
    """One stabilizer-basis probe of lattice-surgery transfer errors."""

    name: str
    injected_state: StimInjectedStateName
    readout_basis: common.SurfaceReadoutBasis
    plot_label: str


TRANSFER_Z_ERROR = LogicalErrorExperiment(
    name="transfer_z_error",
    injected_state="+",
    readout_basis="X",
    plot_label=r"Injected $|+_L\rangle$ (detecting Z errors)",
)
TRANSFER_X_ERROR = LogicalErrorExperiment(
    name="transfer_x_error",
    injected_state="0",
    readout_basis="Z",
    plot_label=r"Injected $|0_L\rangle$ (detecting X errors)",
)

LATTICE_SURGERY_EXPERIMENTS = (TRANSFER_Z_ERROR, TRANSFER_X_ERROR)


# =============================================================================
# Circuit and detector-record analyzer
# =============================================================================


def build_lattice_surgery_only_circuit(
    physical_error_probability: float,
    *,
    experiment: LogicalErrorExperiment,
    patch_sites: common.SteanePatchSites = 14,
    merge_schedule: common.MergeSchedule = "hirano",
    measure_encoder_flags: bool = False,
    full_post_selection: bool = False,
) -> tuple[stim.Circuit, transfer_helper.TransferMeasurementLayout]:
    """Build the detector-annotated lattice-surgery-only experiment.

    The full-postselection branch uses one closing surface round; the correction
    branch uses five. ``minimal_z_only`` jointly extracts Z0145/Z0235 in merge
    round 1 and all three source Z checks in round 2, retaining each associated
    Bell-partner/copy flag. Final X checks are reconstructed from Steane-X
    demolition, and the logical observable includes transfer-frame records.

    Noise is applied only after the Steane-QED record boundary. It covers every
    active operation through the final all-data readout, including native-basis
    resets and measurements, but excludes idle noise between ticks.
    """
    if experiment not in LATTICE_SURGERY_EXPERIMENTS:
        raise ValueError("experiment must be one of LATTICE_SURGERY_EXPERIMENTS.")
    if not 0 <= physical_error_probability <= 1:
        raise ValueError("physical_error_probability must be in [0, 1].")

    circuit = transfer.build_steane_to_rotated_d5_lattice_surgery_circuit(
        patch_sites=patch_sites,
        merge_schedule=merge_schedule,
        injected_state=experiment.injected_state,
        include_cultivation=False,
        measure_encoder_flags=measure_encoder_flags,
        surface_readout_basis=None,
        add_ticks=True,
    )
    transfer_layout = transfer_helper.transfer_measurement_layout(
        transfer.TRANSFER_SPEC,
        patch_sites=patch_sites,
        merge_schedule=merge_schedule,
        include_cultivation=False,
        measure_encoder_flags=measure_encoder_flags,
    )
    if circuit.num_measurements != transfer_layout.num_measurements:
        raise RuntimeError("transfer measurement layout does not match the circuit.")

    layout = transfer.append_post_transfer_and_readout(
        circuit,
        transfer_layout=transfer_layout,
        readout_basis=experiment.readout_basis,
        full_post_selection=full_post_selection,
        add_ticks=True,
    )

    ideal_input, noisy_suffix = _split_after_ideal_input(circuit, transfer_layout)
    return (
        ideal_input
        + add_circuit_depolarizing_noise(
            noisy_suffix,
            p=physical_error_probability,
        ),
        layout,
    )


def _split_after_ideal_input(
    circuit: stim.Circuit,
    layout: transfer_helper.TransferMeasurementLayout,
) -> tuple[stim.Circuit, stim.Circuit]:
    """Split after the first tick following the ideal Steane-QED record."""
    ideal_input_measurement_count = layout.positions["steane_qed"][-1] + 1
    split_index: int | None = None
    observed_measurements = 0
    instructions = cast(Iterable[stim.CircuitInstruction], circuit)
    for index, instruction in enumerate(instructions):
        instruction_circuit = cast(stim.Circuit, circuit[index : index + 1])
        next_measurement_count = observed_measurements + instruction_circuit.num_measurements
        if observed_measurements < ideal_input_measurement_count < next_measurement_count:
            raise ValueError("the QED boundary splits a multi-result instruction.")
        observed_measurements = next_measurement_count
        if (
            split_index is None
            and instruction.name == "TICK"
            and observed_measurements >= ideal_input_measurement_count
        ):
            split_index = index + 1
    if split_index is None:
        raise RuntimeError("failed to find the lattice-surgery noise boundary.")
    if observed_measurements != circuit.num_measurements:
        raise RuntimeError("failed to traverse the complete circuit record.")
    return cast(stim.Circuit, circuit[:split_index]), cast(stim.Circuit, circuit[split_index:])


def _persistent_surface_x_detector_indices(
    layout: transfer_helper.TransferMeasurementLayout,
    *,
    protocol_postselection_count: int,
) -> tuple[int, ...]:
    """Locate the persistent X_bcgh/X_fgkl detector streams."""
    labels = transfer.HIRANO_POSTSELECTED_SURFACE_X_LABELS
    initial_x_start = protocol_postselection_count
    initial_offsets = {label: i for i, label in enumerate(layout.labels["surface_init_x"])}
    indices = [initial_x_start + initial_offsets[label] for label in labels]

    merged_x_start = (
        initial_x_start
        + len(layout.positions["surface_init_x"])
        + len(layout.positions["merged_surface_z"])
    )
    merged_offsets = {label: i for i, label in enumerate(transfer.TRANSFER_SPEC.merged_x_labels)}
    merged_width = len(transfer.TRANSFER_SPEC.merged_x_labels)
    indices.extend(
        merged_x_start + round_index * merged_width + merged_offsets[label]
        for round_index in range(common.LATTICE_SURGERY_ROUNDS)
        for label in labels
    )

    post_transfer_z_start = merged_x_start + len(layout.positions["merged_surface_x"])
    post_transfer_x_start = post_transfer_z_start + len(
        layout.positions[transfer.POST_TRANSFER_SURFACE_Z_FAMILY]
    )
    first_round_reference_labels = {
        *transfer.TRANSFER_SPEC.merged_x_labels,
        *layout.labels["surface_x_recovery"],
    }
    emitted_labels = tuple(
        label
        for label in layout.labels[transfer.POST_TRANSFER_SURFACE_X_FAMILY]
        if not label.startswith("round1_")
        or label.removeprefix("round1_") in first_round_reference_labels
    )
    post_transfer_offsets = {label: i for i, label in enumerate(emitted_labels)}
    indices.extend(
        post_transfer_x_start + post_transfer_offsets[f"round{round_index + 1}_{label}"]
        for round_index in range(transfer.TRANSFER_SPEC.distance)
        for label in labels
    )
    return tuple(indices)


class LatticeSurgeryOnlyShotAnalyzer(DetectorErrorModelShotAnalyzer):
    """Apply the selected hard postselection policy and surface decoder.

    With surface correction, color-code, interface, Bell-partner, and
    reconstructed final-X detectors reject. Optionally, the X_bcgh/X_fgkl
    streams also reject during initialization, all merge rounds, and all five
    post-transfer rounds; remaining surface detectors are decoded by PyMatching.
    Full postselection instead rejects every detector and uses one closing round.

    ``postselect_steane_input`` extends rejection into a noisy end-to-end prefix.
    A complementary-gap threshold adds the logical-sector detector and rejects
    ambiguous matching solutions. A separate decoder-model circuit may omit
    noise from a postselected prefix while retaining the same detector layout.
    """

    def __init__(
        self,
        circuit: stim.Circuit,
        *,
        layout: transfer_helper.TransferMeasurementLayout,
        full_post_selection: bool = False,
        persistent_surface_x_postselection: bool,
        postselect_steane_input: bool = False,
        decoder_model_circuit: stim.Circuit | None = None,
        complementary_gap_postselection_threshold: float | None = None,
    ) -> None:
        analyzer_circuit = circuit.copy()
        analyzer_decoder_model_circuit = (
            None if decoder_model_circuit is None else decoder_model_circuit.copy()
        )
        if full_post_selection:
            postselection_detector_indices = range(circuit.num_detectors)
        else:
            protocol_postselection_count = (
                transfer_helper.num_transfer_protocol_postselection_detectors(layout)
            )
            ideal_input_detector_count = sum(
                len(layout.positions.get(name, ())) for name in ("enc_flags", "steane_qed")
            )
            first_protocol_postselection_detector = (
                0 if postselect_steane_input else ideal_input_detector_count
            )
            postselection_detector_indices = (
                *range(
                    first_protocol_postselection_detector,
                    protocol_postselection_count,
                ),
                *(
                    _persistent_surface_x_detector_indices(
                        layout,
                        protocol_postselection_count=protocol_postselection_count,
                    )
                    if persistent_surface_x_postselection
                    else ()
                ),
            )
        if complementary_gap_postselection_threshold is not None:
            logical_families = tuple(
                name for name in layout.positions if name.startswith("surface_logical_")
            )
            if len(logical_families) != 1:
                raise ValueError(
                    "complementary-gap postselection requires one surface logical readout."
                )
            logical_family = logical_families[0]
            readout_basis_name = logical_family.removeprefix("surface_logical_").upper()
            if readout_basis_name == "X":
                frame_positions = transfer_helper.transfer_logical_frame_positions(layout, "X")
            elif readout_basis_name == "Y":
                frame_positions = transfer_helper.transfer_logical_frame_positions(layout, "Y")
            elif readout_basis_name == "Z":
                frame_positions = transfer_helper.transfer_logical_frame_positions(layout, "Z")
            else:
                raise ValueError(f"unsupported surface logical readout {logical_family!r}.")
            global_logical_measurement_positions = (
                layout.positions[logical_family] + frame_positions
            )
            detector_circuits = [analyzer_circuit]
            if analyzer_decoder_model_circuit is not None:
                detector_circuits.append(analyzer_decoder_model_circuit)
            for detector_circuit in detector_circuits:
                detector_circuit.append(
                    "DETECTOR",
                    [
                        stim.target_rec(position - detector_circuit.num_measurements)
                        for position in global_logical_measurement_positions
                    ],
                )
        super().__init__(
            analyzer_circuit,
            decoder_model_circuit=analyzer_decoder_model_circuit,
            postselection_detector_indices=postselection_detector_indices,
            complementary_gap_postselection_threshold=complementary_gap_postselection_threshold,
        )


# =============================================================================
# Monte Carlo sampling
# =============================================================================


def estimate_lattice_surgery_logical_error_probability(
    physical_error_probability: float,
    *,
    experiment: LogicalErrorExperiment,
    max_shots: int,
    target_logical_errors: int | None,
    patch_sites: common.SteanePatchSites = 14,
    merge_schedule: common.MergeSchedule = "hirano",
    measure_encoder_flags: bool = False,
    persistent_surface_x_postselection: bool,
    full_post_selection: bool = False,
    complementary_gap_postselection_threshold: float | None = None,
    batch_size: int = 100_000,
    seed: int | None = None,
) -> LogicalErrorEstimate:
    """Sample one lattice-surgery LER to a shot limit or exact error target.

    Full postselection uses the packed zero-event fast path. Otherwise, protocol
    detectors and optionally the persistent X_bcgh/X_fgkl streams reject before
    PyMatching decodes the remaining surface history. Pass
    ``target_logical_errors=None`` to consume all ``max_shots``.
    """
    circuit, layout = build_lattice_surgery_only_circuit(
        physical_error_probability,
        experiment=experiment,
        patch_sites=patch_sites,
        merge_schedule=merge_schedule,
        measure_encoder_flags=measure_encoder_flags,
        full_post_selection=full_post_selection,
    )
    return sample_detector_logical_error_probability(
        experiment_name=experiment.name,
        physical_error_probability=physical_error_probability,
        analyzer=LatticeSurgeryOnlyShotAnalyzer(
            circuit,
            layout=layout,
            full_post_selection=full_post_selection,
            persistent_surface_x_postselection=persistent_surface_x_postselection,
            complementary_gap_postselection_threshold=complementary_gap_postselection_threshold,
        ),
        max_shots=max_shots,
        target_logical_errors=target_logical_errors,
        batch_size=batch_size,
        seed=seed,
    )


def estimate_lattice_surgery_logical_error_curves(
    physical_error_probabilities: Iterable[float],
    *,
    experiments: Sequence[LogicalErrorExperiment] = LATTICE_SURGERY_EXPERIMENTS,
    max_shots: int,
    target_logical_errors: int | None,
    patch_sites: common.SteanePatchSites = 14,
    merge_schedule: common.MergeSchedule = "hirano",
    measure_encoder_flags: bool = False,
    persistent_surface_x_postselection: bool,
    full_post_selection: bool = False,
    complementary_gap_postselection_threshold: float | None = None,
    batch_size: int = 100_000,
    seed: int | None = None,
    max_workers: int = 1,
) -> list[LogicalErrorEstimate]:
    """Sample every experiment/probability pair in deterministic grid order.

    ``max_workers`` controls independent circuit-sampling jobs; it does not
    change result order or the seed assigned to any grid point.
    """
    if max_workers <= 0:
        raise ValueError("max_workers must be positive.")
    probabilities = tuple(physical_error_probabilities)
    jobs = [
        (
            experiment,
            probability,
            (
                None
                if seed is None
                else seed + experiment_index * len(probabilities) + probability_index
            ),
        )
        for experiment_index, experiment in enumerate(experiments)
        for probability_index, probability in enumerate(probabilities)
    ]

    if max_workers == 1:
        return [
            estimate_lattice_surgery_logical_error_probability(
                probability,
                experiment=experiment,
                max_shots=max_shots,
                target_logical_errors=target_logical_errors,
                patch_sites=patch_sites,
                merge_schedule=merge_schedule,
                measure_encoder_flags=measure_encoder_flags,
                full_post_selection=full_post_selection,
                persistent_surface_x_postselection=persistent_surface_x_postselection,
                complementary_gap_postselection_threshold=complementary_gap_postselection_threshold,
                batch_size=batch_size,
                seed=sample_seed,
            )
            for experiment, probability, sample_seed in jobs
        ]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                estimate_lattice_surgery_logical_error_probability,
                probability,
                experiment=experiment,
                max_shots=max_shots,
                target_logical_errors=target_logical_errors,
                patch_sites=patch_sites,
                merge_schedule=merge_schedule,
                measure_encoder_flags=measure_encoder_flags,
                full_post_selection=full_post_selection,
                persistent_surface_x_postselection=persistent_surface_x_postselection,
                complementary_gap_postselection_threshold=complementary_gap_postselection_threshold,
                batch_size=batch_size,
                seed=sample_seed,
            )
            for experiment, probability, sample_seed in jobs
        ]
        return [future.result() for future in futures]
