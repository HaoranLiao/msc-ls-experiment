"""End-to-end Stim LER and success-rate analysis for 14-site MSC-LS.

Stim uses ``S|+>`` injection and logical-Y cultivation as the Clifford proxy
for the physical ``T|+>`` and logical-H_XY protocol. Uniform operation noise
covers injection, post-injection QED, cultivation, lattice surgery, and five
post-transfer d=5 surface-code rounds. An ideal diagnostic suffix closes the
surface syndrome history and reads logical Y.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import stim

from examples.state_transfer.lattice_surgery_only_ler.stim_logical_error_rate import (
    LatticeSurgeryOnlyShotAnalyzer,
)
from msc.noise.fault_injection_stim import add_circuit_depolarizing_noise
from msc.noise.fault_tolerance_stim import LogicalErrorEstimate
from msc.state_transfer import helper_stim as transfer_helper
from msc.state_transfer import to_d5_rotated_stim as transfer

POST_TRANSFER_STABILIZATION_ROUNDS = transfer.TRANSFER_SPEC.distance


def build_end_to_end_distillation_circuit(
    physical_error_probability: float,
) -> tuple[
    stim.Circuit,
    stim.Circuit,
    transfer_helper.TransferMeasurementLayout,
]:
    """Build sampling, decoder-model, and layout objects for 14-site MSC-LS.

    All operations from physical-state injection through the fifth d=5
    post-transfer round receive uniform circuit noise. The ideal final syndrome
    round and logical-Y measurement diagnose the noisy output and are not part
    of the distillation gadget. Samples come from the fully noisy circuit. A
    separate decoder-only circuit omits noise from the postselected injection,
    QED, and cultivation prefix so PyMatching can build a graphlike detector
    model; it never supplies samples.
    """
    if not 0 <= physical_error_probability <= 1:
        raise ValueError("physical_error_probability must be in [0, 1].")

    ideal_circuit = transfer.build_steane_to_rotated_d5_lattice_surgery_circuit(
        patch_sites=14,
        injected_state="S",
        include_cultivation=True,
        surface_readout_basis=None,
        add_ticks=True,
    )
    transfer_layout = transfer_helper.transfer_measurement_layout(
        transfer.TRANSFER_SPEC,
        patch_sites=14,
        include_cultivation=True,
    )
    post_transfer_layout = transfer.append_post_transfer_stabilization(
        ideal_circuit,
        transfer_layout=transfer_layout,
        stabilizer_rounds=POST_TRANSFER_STABILIZATION_ROUNDS,
        add_ticks=True,
    )

    # This is the physical experiment: every protocol operation is noisy.
    fully_noisy_sampling_circuit = add_circuit_depolarizing_noise(
        ideal_circuit,
        p=physical_error_probability,
    )

    # This circuit supplies decoder weights only and is never sampled.
    graphlike_decoder_model_circuit = _build_graphlike_decoder_model_circuit(
        ideal_circuit,
        transfer_layout=transfer_layout,
        physical_error_probability=physical_error_probability,
    )
    layout = transfer.append_ideal_y_verification(
        fully_noisy_sampling_circuit,
        post_transfer_layout=post_transfer_layout,
        stabilizer_rounds=POST_TRANSFER_STABILIZATION_ROUNDS,
        add_ticks=True,
    )
    transfer.append_ideal_y_verification(
        graphlike_decoder_model_circuit,
        post_transfer_layout=post_transfer_layout,
        stabilizer_rounds=POST_TRANSFER_STABILIZATION_ROUNDS,
        add_ticks=True,
    )
    return fully_noisy_sampling_circuit, graphlike_decoder_model_circuit, layout


def _build_graphlike_decoder_model_circuit(
    circuit: stim.Circuit,
    *,
    transfer_layout: transfer_helper.TransferMeasurementLayout,
    physical_error_probability: float,
) -> stim.Circuit:
    """Build the graphlike approximation used only to weight matching edges.

    The fully noisy circuit cannot directly supply PyMatching's detector error
    model: a single injection or cultivation fault can flip more than two
    detectors, producing a hyperedge rather than a graph edge. Those prefix
    detectors are hard-postselected, so matching is called only after they have
    all returned zero. Following Hirano, this decoder-only circuit removes the
    offending prefix noise while preserving the same measurements, detectors,
    and observable as the sampling circuit.

    Noise remains on lattice surgery and surface QEC. Stim analytically
    propagates each such noise mechanism to the detector pair it can flip and
    records its probability in the detector error model. PyMatching converts
    that probability into an edge weight, approximately ``log((1-p) / p)``.
    Thus this circuit supplies likelihood weights without supplying samples;
    all samples still come from the fully noisy circuit.

    This is an approximation, not a complete noise model. A prefix fault that
    escapes every hard-postselection detector is present in the sampled data
    but absent from these matching weights. Such a fault can therefore cause a
    logical error or a later syndrome that the decoder misweights. The method
    assumes these accepted prefix faults are sufficiently rare after injection,
    QED, and cultivation postselection. The measured logical-error rate still
    includes them, but the decoder is not maximum-likelihood for them. In the
    Fig. 9 experiment the physical prefix is already ideal, so its sampling
    circuit and decoder-model circuit are identical and no second circuit is
    needed.
    """
    cultivation_measurement_count = transfer_layout.positions["cult"][-1] + 1
    observed_measurements = 0
    split_index: int | None = None
    for instruction_index, instruction in enumerate(circuit):
        instruction_circuit = stim.Circuit()
        instruction_circuit.append(instruction)
        next_measurement_count = observed_measurements + instruction_circuit.num_measurements
        if observed_measurements < cultivation_measurement_count < next_measurement_count:
            raise ValueError("the cultivation boundary splits a multi-result instruction.")
        observed_measurements = next_measurement_count
        if (
            split_index is None
            and instruction.name == "TICK"
            and observed_measurements >= cultivation_measurement_count
        ):
            split_index = instruction_index + 1
    if split_index is None:
        raise RuntimeError("failed to find the post-cultivation noise boundary.")
    if observed_measurements != circuit.num_measurements:
        raise RuntimeError("failed to traverse the complete circuit record.")
    return circuit[:split_index] + add_circuit_depolarizing_noise(
        circuit[split_index:],
        p=physical_error_probability,
    )


def estimate_end_to_end_complementary_gap_curve(
    physical_error_probability: float,
    *,
    complementary_gap_postselection_thresholds: Iterable[float],
    shots: int,
    batch_size: int = 100_000,
    seed: int | None = None,
) -> list[tuple[float, LogicalErrorEstimate]]:
    """Estimate LER versus success rate from one shared Monte Carlo sample.

    Every shot first undergoes the protocol's ordinary hard postselection. The
    accepted syndrome is decoded once in each global logical sector. Applying
    several thresholds to the resulting complementary gap then produces the
    Figure 10-style tradeoff without resampling for every threshold.
    """
    gap_thresholds = tuple(
        float(threshold) for threshold in complementary_gap_postselection_thresholds
    )
    if not gap_thresholds:
        raise ValueError("at least one complementary-gap threshold is required.")
    if any(not math.isfinite(threshold) or threshold < 0 for threshold in gap_thresholds):
        raise ValueError("complementary-gap thresholds must be finite and nonnegative.")
    if shots <= 0:
        raise ValueError("shots must be positive.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    fully_noisy_sampling_circuit, graphlike_decoder_model_circuit, layout = (
        build_end_to_end_distillation_circuit(physical_error_probability)
    )
    analyzer = LatticeSurgeryOnlyShotAnalyzer(
        fully_noisy_sampling_circuit,
        layout=layout,
        postselect_steane_input=True,
        decoder_model_circuit=graphlike_decoder_model_circuit,
        complementary_gap_postselection_threshold=0.0,
    )
    sampler = analyzer._circuit.compile_detector_sampler(seed=seed)
    accepted_shot_counts = np.zeros(len(gap_thresholds), dtype=np.int64)
    logical_error_counts = np.zeros(len(gap_thresholds), dtype=np.int64)
    shots_remaining = shots
    while shots_remaining:
        selected_batch_size = min(batch_size, shots_remaining)
        detection_events, observable_flips = sampler.sample(
            selected_batch_size,
            separate_observables=True,
        )
        accepted_before_gap_selection, logical_errors, complementary_gaps = (
            analyzer.decode_complementary_gap_samples(
                detection_events,
                observable_flips,
            )
        )
        for threshold_index, threshold in enumerate(gap_thresholds):
            accepted = accepted_before_gap_selection & (complementary_gaps >= threshold)
            accepted_shot_counts[threshold_index] += np.count_nonzero(accepted)
            logical_error_counts[threshold_index] += np.count_nonzero(accepted & logical_errors)
        shots_remaining -= selected_batch_size

    return [
        (
            threshold,
            LogicalErrorEstimate(
                experiment_name="end_to_end_magic_state",
                physical_error_probability=physical_error_probability,
                sampled_shots=shots,
                accepted_shots=int(accepted_shot_counts[threshold_index]),
                logical_errors=int(logical_error_counts[threshold_index]),
            ),
        )
        for threshold_index, threshold in enumerate(gap_thresholds)
    ]
