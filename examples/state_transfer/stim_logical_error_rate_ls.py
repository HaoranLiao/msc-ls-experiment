"""Stim analysis of the noisy lattice-surgery stage alone.

The encoded Steane input and the closing surface round/readout are ideal. Only
surface preparation, three merged rounds, demolition, and boundary recovery
are noisy. Transfer-local checks reject while PyMatching corrects surface-code
detector events, and the ideal closing-round changes are postselected. Rejecting
every detector remains optional. This hybrid mode follows the literal closing
policy in the Fig. 9 prose but does not reproduce the authors' fully
postselected Fig. 9 data or their concurrent superdense circuit.
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
    SingleFaultAssessmentResult as CircuitSingleFaultAssessmentResult,
    assess_circuit_single_fault_tolerance,
    sample_detector_logical_error_probability,
)
from msc.state_transfer import helper_stim as transfer_helper
from msc.state_transfer import to_d5_rotated_stim as transfer

from .stim_logical_error_rate import (
    TRANSFER_X_ERROR,
    TRANSFER_Z_ERROR,
    LogicalErrorExperiment,
)

LATTICE_SURGERY_EXPERIMENTS = (TRANSFER_Z_ERROR, TRANSFER_X_ERROR)


# =============================================================================
# Circuit and detector-record analyzer
# =============================================================================


def build_lattice_surgery_only_circuit(
    physical_error_probability: float,
    *,
    experiment: LogicalErrorExperiment,
) -> tuple[stim.Circuit, transfer_helper.TransferMeasurementLayout]:
    """Build the detector-annotated lattice-surgery-only experiment.

    The Steane encoder and post-injection QED are ideal. Noise begins with
    surface preparation and covers the three merge rounds, Steane demolition,
    and top-edge X-check recovery. One ideal full surface round then closes the
    detector history before an ideal destructive logical readout. All detector
    annotations remain available so ``LatticeSurgeryOnlyShotAnalyzer`` can use
    the protocol's hybrid postselection-plus-decoding policy or, for diagnostic
    comparisons only, reject every detector event.

    Call structure: build an ideal transfer -> split after Steane QED -> noise
    the surface/merge/demolition suffix -> append an ideal closing round and
    logical readout.
    """
    if experiment not in LATTICE_SURGERY_EXPERIMENTS:
        raise ValueError("experiment must be one of LATTICE_SURGERY_EXPERIMENTS.")
    if not 0 <= physical_error_probability <= 1:
        raise ValueError("physical_error_probability must be in [0, 1].")

    # Build the ideal input and transfer, then verify its measurement layout.
    circuit = transfer.build_steane_to_rotated_d5_lattice_surgery_circuit(
        injected_state=experiment.injected_state,
        include_cultivation=False,
        include_post_transfer_stabilizer_rounds=False,
        surface_readout_basis=None,
        add_ticks=True,
    )
    transfer_layout = transfer_helper.transfer_measurement_layout(
        transfer.TRANSFER_SPEC,
        include_cultivation=False,
        include_post_transfer_stabilizer_rounds=False,
        surface_readout_basis=None,
    )
    if circuit.num_measurements != transfer_layout.num_measurements:
        raise RuntimeError("transfer measurement layout does not match the circuit.")

    # Split exactly after QED so only the lattice-surgery suffix receives noise.
    ideal_input_measurement_count = transfer_layout.positions["steane_qed"][-1] + 1
    ideal_input = stim.Circuit()
    noisy_transfer = stim.Circuit()
    observed_measurements = 0
    for instruction in circuit:
        if isinstance(instruction, stim.CircuitRepeatBlock):
            raise ValueError(
                "the lattice-surgery noise boundary requires no repeat blocks."
            )
        instruction_circuit = stim.Circuit()
        instruction_circuit.append(instruction)
        next_measurement_count = (
            observed_measurements + instruction_circuit.num_measurements
        )
        if observed_measurements < ideal_input_measurement_count < next_measurement_count:
            raise ValueError("the QED boundary splits a multi-result instruction.")
        target = (
            ideal_input
            if observed_measurements < ideal_input_measurement_count
            else noisy_transfer
        )
        target.append(instruction)
        observed_measurements = next_measurement_count
    if observed_measurements != circuit.num_measurements:
        raise RuntimeError("failed to traverse the complete circuit record.")

    # Noise the transfer, then append the noise-free closing round and readout.
    circuit = ideal_input + add_circuit_depolarizing_noise(
        noisy_transfer,
        p_1q=physical_error_probability,
        p_2q=physical_error_probability,
        p_measure=physical_error_probability,
        p_reset=physical_error_probability,
    )
    layout = transfer.append_ideal_verification_and_readout(
        circuit,
        transfer_layout=transfer_layout,
        readout_basis=experiment.readout_basis,
        add_ticks=True,
    )
    return circuit, layout


class LatticeSurgeryOnlyShotAnalyzer(DetectorErrorModelShotAnalyzer):
    """Apply hybrid surface correction with an ideal closing postselection.

    By default, the color-code, interface, and reconstructed final-X prefix
    rejects, noisy surface-history events are decoded by PyMatching, and every
    detector from the noise-free closing surface round is postselected. Final
    destructive-readout boundary detectors remain decoder inputs. Setting
    ``full_post_selection=True`` additionally rejects every noisy surface and
    readout-boundary detector as a diagnostic comparison.
    """

    def __init__(
        self,
        circuit: stim.Circuit,
        *,
        layout: transfer_helper.TransferMeasurementLayout,
        full_post_selection: bool = False,
    ) -> None:
        if full_post_selection:
            postselection_detector_indices = range(circuit.num_detectors)
        else:
            protocol_postselection_count = (
                transfer_helper.num_transfer_protocol_postselection_detectors(layout)
            )
            noisy_surface_history_detector_count = sum(
                len(round_positions)
                for family in ("merged_surface_z", "merged_surface_x")
                for round_positions in layout.rounds(family)
            )
            closing_detector_start = (
                protocol_postselection_count + noisy_surface_history_detector_count
            )
            closing_detector_count = sum(
                len(layout.positions[family])
                for family in (
                    transfer.IDEAL_POST_TRANSFER_SURFACE_Z_FAMILY,
                    transfer.IDEAL_POST_TRANSFER_SURFACE_X_FAMILY,
                )
            )
            closing_detector_stop = closing_detector_start + closing_detector_count
            if closing_detector_stop > circuit.num_detectors:
                raise RuntimeError("ideal closing detector layout exceeds the circuit.")
            postselection_detector_indices = (
                *range(protocol_postselection_count),
                *range(closing_detector_start, closing_detector_stop),
            )
        super().__init__(
            circuit,
            postselection_detector_indices=postselection_detector_indices,
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
    full_post_selection: bool = False,
    batch_size: int = 100_000,
    seed: int | None = None,
) -> LogicalErrorEstimate:
    """Sample one lattice-surgery LER conditioned on the selected policy.

    Full postselection uses the detector sampler's packed zero-event fast path;
    Hybrid handling rejects the ideal closing-round syndrome changes and sends
    accepted noisy surface-history records to PyMatching.
    """
    circuit, layout = build_lattice_surgery_only_circuit(
        physical_error_probability,
        experiment=experiment,
    )
    return sample_detector_logical_error_probability(
        experiment_name=experiment.name,
        physical_error_probability=physical_error_probability,
        analyzer=LatticeSurgeryOnlyShotAnalyzer(
            circuit,
            layout=layout,
            full_post_selection=full_post_selection,
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
    full_post_selection: bool = False,
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
            None
            if seed is None
            else seed + experiment_index * len(probabilities) + probability_index,
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
                full_post_selection=full_post_selection,
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
                full_post_selection=full_post_selection,
                batch_size=batch_size,
                seed=sample_seed,
            )
            for experiment, probability, sample_seed in jobs
        ]
        return [future.result() for future in futures]
