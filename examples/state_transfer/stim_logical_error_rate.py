"""Stim analysis of the noisy lattice-surgery stage alone.

The encoded Steane prefix is ideal. Noise covers surface ``|+>`` preparation
and its initial X/Z-check round; all interface, compatible surface, and
scheduled Steane checks in the three merge rounds; Steane-X demolition; and
the required ``m_ab`` lattice-surgery outcome. The non-full-postselection
branch then performs five noisy full surface rounds and a noisy destructive
all-data readout. Only operation locations are noisy; ``TICK`` intervals add
no idle errors.

This is not the exact noise boundary of Hirano's released Fig. 9 builder. That
code includes per-tick idle noise. This experiment preserves Hirano's branch-
dependent post-transfer rounds, destructive readout, and semantic
postselection mask while omitting idle noise and retaining the modular
serialized transfer schedule.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import stim

from msc.noise.fault_injection_stim import add_circuit_depolarizing_noise
from msc.noise.fault_tolerance_stim import (
    DetectorErrorModelShotAnalyzer,
    LogicalErrorEstimate,
    assess_circuit_single_fault_tolerance,
    sample_detector_logical_error_probability,
)
from msc.noise.fault_tolerance_stim import (
    SingleFaultAssessmentResult as CircuitSingleFaultAssessmentResult,
)
from msc.state_transfer import common
from msc.state_transfer import helper_stim as transfer_helper
from msc.state_transfer import to_d5_rotated_stim as transfer


@dataclass(frozen=True)
class LogicalErrorExperiment:
    """One stabilizer-basis probe of lattice-surgery transfer errors."""

    name: str
    injected_state: str
    readout_basis: str
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
    full_post_selection: bool = False,
) -> tuple[stim.Circuit, transfer_helper.TransferMeasurementLayout]:
    """Build the detector-annotated lattice-surgery-only experiment.

    This builder applies noise in the following places:

    1. Surface initialization: noisy native ``RX`` resets prepare the 25
       surface-data qubits in ``|+>``. One noisy combined X/Z-stabilizer round
       then establishes the Z reference and the known-``+1`` X detector layer.
       Its native-basis ancilla resets, four CNOT layers, and native-basis
       ancilla measurements are all noisy.
    2. All three merge rounds: the interface-Z checks, every surface-Z check,
       every compatible surface-X check, and every scheduled Steane check are
       noisy. Round 1 measures ``Z0145`` once and uses one simultaneous
       superdense extraction for the requested X/Z pairs on faces ``0235`` and
       ``0246``. Round 2 measures ``Z0145``, ``Z0235``, and ``Z0246`` and
       postselects the three unused Bell-partner readouts; round 3 has no Steane
       checks. The round-1 Z-only ``0145`` extraction likewise retains its
       unused Bell-partner flag. These flags test Bell correlations, not the
       omitted opposite-basis face operators, because their data-coupling
       phases are absent.
    3. Transfer completion: Steane-X demolition and the required ``m_ab``
       lattice-surgery outcome are noisy, including the fresh ``m_ab`` ancilla
       reset. This one-check ``X_ab`` extraction completes ``X0145ab``, and
       ``m_ab`` becomes the restored ``X_ab`` reference. There is no subsequent
       two-check ``X_ab``/``X_cd`` recovery round and no post-merge ``X_cd``
       result at this stage.
    4. Post-transfer continuation and readout: the non-full branch performs
       five noisy full X/Z rounds; the full branch performs one. A noisy
       destructive measurement of all 25 data qubits follows, with compatible
       readout-boundary detectors.

    The selected Steane encoder and post-injection QED before these phases are
    ideal. The final observable combines the appropriate five-qubit logical
    support within the all-data readout with the transfer-frame records.

    Hirano's released Fig. 9 builder likewise uses an ideal Steane preparation,
    a noisy complete X/Z surface round before the merge, and noisy scheduled
    Steane and surface operations during lattice surgery. Our operation noise
    covers the same phases. The remaining differences are that we omit idle
    noise and serialize the interface/surface/Steane groups instead of using
    the interleaved superdense schedule.

    The circuit itself only defines detector records. The persistent
    postselection of the ``X_bcgh`` and ``X_fgkl`` streams is applied by
    ``LatticeSurgeryOnlyShotAnalyzer``, which selects their initialization
    detectors and their detectors in all three merge rounds.
    """
    if experiment not in LATTICE_SURGERY_EXPERIMENTS:
        raise ValueError("experiment must be one of LATTICE_SURGERY_EXPERIMENTS.")
    if not 0 <= physical_error_probability <= 1:
        raise ValueError("physical_error_probability must be in [0, 1].")

    # Build the ideal input and transfer, then verify its measurement layout.
    circuit = transfer.build_steane_to_rotated_d5_lattice_surgery_circuit(
        patch_sites=patch_sites,
        injected_state=experiment.injected_state,
        include_cultivation=False,
        surface_readout_basis=None,
        add_ticks=True,
    )
    transfer_layout = transfer_helper.transfer_measurement_layout(
        transfer.TRANSFER_SPEC,
        patch_sites=patch_sites,
        include_cultivation=False,
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

    # Keep injection and QED ideal; noise starts at surface initialization.
    # `add_circuit_depolarizing_noise` noises every operation in whatever
    # circuit it is given, so to leave injection and QED ideal we must first
    # cut `circuit` into two pieces and noise only the second one: everything
    # up to (and including) the TICK that follows the last QED measurement
    # becomes `ideal_input`, and everything after that TICK becomes
    # `noisy_transfer` (surface init through lattice-surgery completion).
    #
    # `stim.Circuit` supports slicing (``circuit[:k]``), so cutting it is just
    # finding the right instruction index `split_index`. That index is not
    # known in advance because the cut is specified by a measurement-record
    # position (the last QED measurement), not an instruction index, and a
    # single instruction can add more than one measurement result at once. So
    # we scan the instructions in order, tallying how many measurement
    # results each one contributes, until the running total reaches the QED
    # measurement count, and record the index of the next TICK after that.
    ideal_input_measurement_count = transfer_layout.positions["steane_qed"][-1] + 1
    split_index: int | None = None
    observed_measurements = 0
    index = -1
    for instruction in circuit:
        index += 1
        instruction_circuit = stim.Circuit()
        instruction_circuit.append(instruction)
        next_measurement_count = observed_measurements + instruction_circuit.num_measurements
        if observed_measurements < ideal_input_measurement_count < next_measurement_count:
            raise ValueError("the QED boundary splits a multi-result instruction.")
        observed_measurements = next_measurement_count
        # The first TICK once all QED measurements have been emitted is the
        # cut point: `ideal_input` ends here, `noisy_transfer` starts next.
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
    ideal_input = circuit[:split_index]
    noisy_transfer_and_readout = circuit[split_index:]

    circuit = ideal_input + add_circuit_depolarizing_noise(
        noisy_transfer_and_readout,
        p=physical_error_probability,
    )
    return circuit, layout


class LatticeSurgeryOnlyShotAnalyzer(DetectorErrorModelShotAnalyzer):
    """Apply surface correction with two always-postselected X streams.

    ``full_post_selection=False`` is not pure surface-code QEC. The color-code,
    interface, and reconstructed final-X detectors reject. In addition, Hirano
    sets one persistent postselection flag on the ``X_bcgh`` and ``X_fgkl``
    check objects: their initial known-+1 detectors and their detector in every
    merge round reject. All surface-Z detectors and all other compatible
    surface-X detectors from the merge rounds are decoded by PyMatching.

    The two persistent X streams also reject during all five post-transfer
    rounds. Lattice-surgery completion retains only the required ``X_ab``
    outcome from a one-check ``X_ab`` extraction; there is no subsequent
    two-check ``X_ab``/``X_cd`` recovery round.
    Although the pre-merge round measures ``X_cd``, that value cannot cross the
    merge deformation. ``X_cd`` first reappears in noisy full round 1, where it
    establishes the new reference and has no change detector. Other post-
    transfer changes and the compatible destructive-readout boundary detectors
    are decoded. Setting
    ``full_post_selection=True`` uses Hirano's one-round closing circuit and
    rejects every ordinary detector. A numeric
    ``complementary_gap_postselection_threshold`` adds Hirano's final logical
    detector to the complete circuit and rejects a shot when the two global
    logical-sector matching weights differ by less than that threshold.
    ``postselect_steane_input=True`` additionally rejects injection-flag and
    Steane-QED detections for the end-to-end protocol.
    ``decoder_model_circuit`` may omit noise from that postselected prefix while
    retaining the same detector and observable layout as the sampling circuit.
    """

    def __init__(
        self,
        circuit: stim.Circuit,
        *,
        layout: transfer_helper.TransferMeasurementLayout,
        full_post_selection: bool = False,
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
                len(layout.positions[name]) for name in ("enc_flags", "steane_qed")
            )
            # Detector order is protocol prefix -> initial surface X -> merged
            # surface Z -> merged surface X -> post-transfer Z -> post-transfer
            # X -> destructive-readout boundary.
            initial_x_detector_start = protocol_postselection_count
            merged_x_detector_start = (
                initial_x_detector_start
                + len(layout.positions["surface_init_x"])
                + len(layout.positions["merged_surface_z"])
            )
            # Offsets within the pre-merge surface-X detector block.
            initial_x_detector_offset_by_label = {
                label: offset for offset, label in enumerate(layout.labels["surface_init_x"])
            }
            # Offsets within each merge round's surface-X detector block.
            merged_x_detector_offset_by_label = {
                label: offset for offset, label in enumerate(transfer.TRANSFER_SPEC.merged_x_labels)
            }
            # Hirano's two persistently postselected surface-X initialization
            # detectors; merge-round detectors from the same streams follow.
            postselected_surface_x_detector_indices = [
                initial_x_detector_start + initial_x_detector_offset_by_label[label]
                for label in transfer.HIRANO_POSTSELECTED_SURFACE_X_LABELS
            ]
            merged_x_round_width = len(transfer.TRANSFER_SPEC.merged_x_labels)
            # Reproduce Hirano's persistent object flag in the analyzer: reject
            # these two X streams at initialization and in every merge round.
            postselected_surface_x_detector_indices.extend(
                merged_x_detector_start
                + round_index * merged_x_round_width
                + merged_x_detector_offset_by_label[label]
                for round_index in range(common.LATTICE_SURGERY_ROUNDS)
                for label in transfer.HIRANO_POSTSELECTED_SURFACE_X_LABELS
            )
            post_transfer_z_detector_start = merged_x_detector_start + len(
                layout.positions["merged_surface_x"]
            )
            post_transfer_x_detector_start = post_transfer_z_detector_start + len(
                layout.positions[transfer.POST_TRANSFER_SURFACE_Z_FAMILY]
            )
            first_round_x_reference_labels = {
                *transfer.TRANSFER_SPEC.merged_x_labels,
                *layout.labels["surface_x_recovery"],
            }
            emitted_post_transfer_x_detector_labels = tuple(
                label
                for label in layout.labels[transfer.POST_TRANSFER_SURFACE_X_FAMILY]
                if not label.startswith("round1_")
                or label.removeprefix("round1_") in first_round_x_reference_labels
            )
            post_transfer_x_detector_offset_by_label = {
                label: offset
                for offset, label in enumerate(emitted_post_transfer_x_detector_labels)
            }
            postselected_surface_x_detector_indices.extend(
                post_transfer_x_detector_start
                + post_transfer_x_detector_offset_by_label[f"round{round_index + 1}_{label}"]
                for round_index in range(transfer.TRANSFER_SPEC.distance)
                for label in transfer.HIRANO_POSTSELECTED_SURFACE_X_LABELS
            )
            first_protocol_postselection_detector = (
                0 if postselect_steane_input else ideal_input_detector_count
            )
            postselection_detector_indices = (
                *range(
                    first_protocol_postselection_detector,
                    protocol_postselection_count,
                ),
                *postselected_surface_x_detector_indices,
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
            readout_basis = logical_family.removeprefix("surface_logical_").upper()
            global_logical_measurement_positions = layout.positions[
                logical_family
            ] + transfer_helper.transfer_logical_frame_positions(layout, readout_basis)
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


@dataclass(frozen=True)
class SingleFaultAssessmentResult(CircuitSingleFaultAssessmentResult):
    """Single-fault classification together with its postselection policy."""

    full_post_selection: bool


# =============================================================================
# Exact checks and Monte Carlo sampling
# =============================================================================


def assess_lattice_surgery_single_fault_tolerance(
    *,
    experiment: LogicalErrorExperiment,
    patch_sites: common.SteanePatchSites = 14,
    full_post_selection: bool = False,
    decoder_error_probability: float = 1e-3,
) -> SingleFaultAssessmentResult:
    """Classify every elementary fault and report graphlike distance.

    Stim supplies exact detector/observable effects for each single fault;
    PyMatching classifies accepted effects. ``decoder_error_probability`` only
    gives those detector-model edges nonzero weights.
    """
    if not 0 < decoder_error_probability < 1:
        raise ValueError("decoder_error_probability must be strictly between 0 and 1.")
    circuit, layout = build_lattice_surgery_only_circuit(
        decoder_error_probability,
        experiment=experiment,
        patch_sites=patch_sites,
        full_post_selection=full_post_selection,
    )
    result = assess_circuit_single_fault_tolerance(
        experiment_name=experiment.name,
        analyzer=LatticeSurgeryOnlyShotAnalyzer(
            circuit,
            layout=layout,
            full_post_selection=full_post_selection,
        ),
    )
    return SingleFaultAssessmentResult(
        **vars(result),
        full_post_selection=full_post_selection,
    )


def estimate_lattice_surgery_logical_error_probability(
    physical_error_probability: float,
    *,
    experiment: LogicalErrorExperiment,
    shots: int,
    patch_sites: common.SteanePatchSites = 14,
    full_post_selection: bool = False,
    complementary_gap_postselection_threshold: float | None = None,
    batch_size: int = 100_000,
    seed: int | None = None,
) -> LogicalErrorEstimate:
    """Sample one lattice-surgery LER conditioned on the selected policy.

    Full postselection uses the detector sampler's packed zero-event fast path.
    Surface correction, with ``X_bcgh`` and ``X_fgkl`` always postselected,
    rejects Hirano's persistent surface-X streams and sends accepted noisy
    surface-history records to PyMatching. A numeric complementary-gap
    threshold additionally selects shots using the two logical-sector weights
    from that one global matching graph; ``None`` disables this selection.
    """
    circuit, layout = build_lattice_surgery_only_circuit(
        physical_error_probability,
        experiment=experiment,
        patch_sites=patch_sites,
        full_post_selection=full_post_selection,
    )
    return sample_detector_logical_error_probability(
        experiment_name=experiment.name,
        physical_error_probability=physical_error_probability,
        analyzer=LatticeSurgeryOnlyShotAnalyzer(
            circuit,
            layout=layout,
            full_post_selection=full_post_selection,
            complementary_gap_postselection_threshold=complementary_gap_postselection_threshold,
        ),
        shots=shots,
        batch_size=batch_size,
        seed=seed,
    )


def estimate_lattice_surgery_logical_error_curves(
    physical_error_probabilities: Iterable[float],
    *,
    experiments: Sequence[LogicalErrorExperiment] = LATTICE_SURGERY_EXPERIMENTS,
    shots: int,
    patch_sites: common.SteanePatchSites = 14,
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
                shots=shots,
                patch_sites=patch_sites,
                full_post_selection=full_post_selection,
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
                shots=shots,
                patch_sites=patch_sites,
                full_post_selection=full_post_selection,
                complementary_gap_postselection_threshold=complementary_gap_postselection_threshold,
                batch_size=batch_size,
                seed=sample_seed,
            )
            for experiment, probability, sample_seed in jobs
        ]
        return [future.result() for future in futures]
