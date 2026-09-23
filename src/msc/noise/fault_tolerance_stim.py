"""Logical-error-rate and fault-tolerance analysis for Stim detector circuits.

Call structure::

    analyzer = DetectorErrorModelShotAnalyzer(circuit, ...)
    ├── sample_detector_logical_error_probability(analyzer=analyzer, ...)
    │   ├── analyzer.postselect_and_decode_samples(sampled detector records)
    │   └── LogicalErrorEstimate
    └── analyzer.decode_complementary_gap_samples(...)

    fit_power_law(sequence[LogicalErrorEstimate])
    └── (coefficient, exponent)

    enumerate_circuit_physical_fault_signatures(circuit)
    └── CircuitPhysicalFaultInventory

    assess_fully_postselected_circuit_fault_distance(circuit=circuit, ...)
    ├── enumerate_circuit_physical_fault_signatures(circuit)
    ├── exhaustive weight-1 and weight-2 search
    ├── weight-3 witness search
    └── PhysicalFaultDistanceAssessment

"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Hashable, Sequence
from dataclasses import dataclass

import numpy as np
import stim

# ---------------------------------------------------------------------------
# Shared detector-record analyzer
# ---------------------------------------------------------------------------


class DetectorErrorModelShotAnalyzer:
    """Postselect and globally PyMatching-decode detector samples.

    The sampling circuit must have detectors and one observable. Selected
    detectors reject shots; accepted shots are logical errors when the decoded
    and sampled observable flips disagree. ``decoder_model_circuit`` may
    provide a graphlike detector model with the same detector and observable
    layout while samples still come from the fully noisy circuit. When a
    complementary-gap threshold is supplied, the circuit's final detector must
    be the complementary-gap detector.

    Toggling that detector forces the opposite global logical sector, and the
    analyzer rejects the shot when the two decoding weights differ by less than
    the user-supplied threshold.

    Call structure::

        analyzer.postselect_and_decode_samples(detection_events, observable_flips)
        ├── ordinary decoding
        │   ├── _normalize_samples(...)
        │   ├── _postselection_acceptance(...)
        │   └── matching.decode_batch(...)
        └── complementary-gap decoding
            └── decode_complementary_gap_samples(...)
                ├── _normalize_samples(...)
                ├── _postselection_acceptance(...)
                └── _decode_with_complementary_gap(...)
                    └── matching.decode(...) for both logical sectors
    """

    def __init__(
        self,
        circuit: stim.Circuit,
        *,
        decoder_model_circuit: stim.Circuit | None = None,
        postselection_detector_indices: Sequence[int] = (),
        complementary_gap_postselection_threshold: float | None = None,
    ) -> None:
        """Compile the sampling contract and its PyMatching decoder.

        ``circuit`` is copied and later sampled. ``decoder_model_circuit`` is
        never sampled; when supplied, only its graphlike detector error model
        provides matching edges and weights. The two circuits must therefore
        have identical detector and observable counts.

        ``postselection_detector_indices`` identifies hard-rejection
        detectors. A non-``None`` complementary-gap threshold additionally
        interprets the final detector as the logical-sector constraint and
        rejects decoded shots whose sector-weight gap is below the threshold.
        """
        import pymatching

        postselection_indices = tuple(postselection_detector_indices)
        if len(postselection_indices) != len(set(postselection_indices)):
            raise ValueError("postselection_detector_indices must be unique.")
        if any(not 0 <= index < circuit.num_detectors for index in postselection_indices):
            raise ValueError("a postselection detector index is outside the circuit.")
        gap_threshold = complementary_gap_postselection_threshold
        if gap_threshold is not None:
            if not math.isfinite(gap_threshold) or gap_threshold < 0:
                raise ValueError(
                    "complementary_gap_postselection_threshold must be None or a finite "
                    "nonnegative value."
                )
            if not circuit.num_detectors:
                raise ValueError("complementary-gap postselection requires a final detector.")
        if circuit.num_observables != 1:
            raise ValueError("logical-error sampling requires exactly one observable.")
        self._circuit = circuit.copy()
        if decoder_model_circuit is None:
            decoder_model_circuit = circuit
        if decoder_model_circuit.num_detectors != circuit.num_detectors:
            raise ValueError("decoder_model_circuit must have the same number of detectors.")
        if decoder_model_circuit.num_observables != circuit.num_observables:
            raise ValueError("decoder_model_circuit must have the same number of observables.")
        detector_error_model = decoder_model_circuit.detector_error_model(decompose_errors=True)
        self._matching = pymatching.Matching.from_detector_error_model(detector_error_model)
        self._num_detectors = self._circuit.num_detectors
        self._postselection_detector_indices = np.asarray(
            postselection_indices,
            dtype=np.intp,
        )
        self._complementary_gap_postselection_threshold = gap_threshold

    def postselect_and_decode_samples(
        self,
        detection_events: np.ndarray,
        observable_flips: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply detector postselection and classify accepted logical errors.

        The two returned Boolean arrays have one entry per shot. The first
        marks shots accepted by every enabled postselection rule. The second
        marks accepted shots whose decoded observable disagrees with the
        sampled observable; rejected shots are always false in this array.

        Ordinary shots are decoded together with ``decode_batch``. Enabling a
        complementary-gap threshold instead decodes both logical sectors per
        ordinarily accepted shot and then applies the gap threshold.
        """
        gap_threshold = self._complementary_gap_postselection_threshold
        if gap_threshold is not None:
            gap_analysis = self.decode_complementary_gap_samples(
                detection_events,
                observable_flips,
            )
            acceptance_mask, logical_error_mask, complementary_gaps = gap_analysis
            acceptance_mask &= complementary_gaps >= gap_threshold
            logical_error_mask &= acceptance_mask
            return acceptance_mask, logical_error_mask
        else:
            detection_events, observable_flips = self._normalize_samples(
                detection_events,
                observable_flips,
            )
            acceptance_mask = self._postselection_acceptance(detection_events)
            logical_error_mask = np.zeros(len(detection_events), dtype=bool)
            accepted_shot_indices = np.flatnonzero(acceptance_mask)
            if accepted_shot_indices.size == 0:
                return acceptance_mask, logical_error_mask
            predictions = np.asarray(
                self._matching.decode_batch(detection_events[accepted_shot_indices]),
                dtype=bool,
            ).reshape(-1)
            logical_error_mask[accepted_shot_indices] = (
                predictions != observable_flips[accepted_shot_indices, 0]
            )
            return acceptance_mask, logical_error_mask

    def decode_complementary_gap_samples(
        self,
        detection_events: np.ndarray,
        observable_flips: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Decode samples into pre-gap acceptance, logical-error, and gap arrays.

        Hard detector postselection is applied first. Each surviving shot is
        decoded in both logical sectors; the lower-weight sector supplies the
        logical prediction and the absolute weight difference supplies the
        complementary gap. This threshold-independent output lets one Monte
        Carlo sample be evaluated at several gap thresholds.

        The returned arrays contain pre-gap acceptance, accepted logical-error
        status, and complementary gap respectively. Gaps remain ``nan`` for
        shots rejected by hard detector postselection.
        """
        if self._complementary_gap_postselection_threshold is None:
            raise ValueError(
                "construct the analyzer with a complementary-gap threshold before "
                "requesting complementary-gap samples."
            )
        detection_events, observable_flips = self._normalize_samples(
            detection_events,
            observable_flips,
        )
        accepted_before_gap_selection = self._postselection_acceptance(detection_events)
        logical_error_mask = np.zeros(len(detection_events), dtype=bool)
        complementary_gaps = np.full(len(detection_events), np.nan)
        for shot_index in np.flatnonzero(accepted_before_gap_selection):
            prediction, complementary_gap = self._decode_with_complementary_gap(
                detection_events[shot_index]
            )
            logical_error_mask[shot_index] = prediction[0] != observable_flips[shot_index, 0]
            complementary_gaps[shot_index] = complementary_gap
        return accepted_before_gap_selection, logical_error_mask, complementary_gaps

    def _normalize_samples(
        self,
        detection_events: np.ndarray,
        observable_flips: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert one sampler batch to Boolean arrays and enforce its shape.

        PyMatching receives one detector column per circuit detector. This
        analyzer requires exactly one observable, so observable records must
        have shape ``(shot_count, 1)``.
        """
        detection_events = np.asarray(detection_events, dtype=bool)
        observable_flips = np.asarray(observable_flips, dtype=bool)
        if detection_events.ndim != 2 or detection_events.shape[1] != self._num_detectors:
            raise ValueError(
                "detection_events must have shape "
                f"(n, {self._num_detectors}), got {detection_events.shape}."
            )
        sample_count = detection_events.shape[0]
        expected_observable_shape = (sample_count, 1)
        if observable_flips.shape != expected_observable_shape:
            raise ValueError(
                "observable_flips must have shape "
                f"{expected_observable_shape}, got {observable_flips.shape}."
            )
        return detection_events, observable_flips

    def _postselection_acceptance(self, detection_events: np.ndarray) -> np.ndarray:
        """Accept shots with no events at any hard-postselection detector."""
        if not self._postselection_detector_indices.size:
            return np.ones(len(detection_events), dtype=bool)
        selected_detection_events = detection_events[:, self._postselection_detector_indices]
        return ~selected_detection_events.any(axis=1)

    def _decode_with_complementary_gap(
        self,
        shot_detection_events: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """Decode both global logical sectors and return the best prediction.

        The circuit's final detector is constructed from the same logical
        parity used by its observable. Holding every physical detector fixed
        while toggling this one detector therefore imposes the opposite logical
        parity constraint on MWPM. The two calls return the minimum-weight
        explanations in the two logical sectors; their weight difference is
        the complementary gap.

        PyMatching does not expose a ``return_complementary`` option. This
        method owns that two-decode operation without modifying the caller's
        detector-event array.
        """
        decode_sector = self._matching.decode
        prediction, decoded_weight = decode_sector(
            shot_detection_events,
            return_weight=True,
        )
        complementary_detection_events = shot_detection_events.copy()
        complementary_detection_events[-1] ^= True
        try:
            complementary_prediction, complementary_decoded_weight = decode_sector(
                complementary_detection_events,
                return_weight=True,
            )
        except ValueError:
            # No competing sector exists, so this shot has unbounded separation.
            return prediction, math.inf
        if complementary_decoded_weight < decoded_weight:
            prediction = complementary_prediction
        complementary_gap = abs(decoded_weight - complementary_decoded_weight)
        return prediction, float(complementary_gap)


# ---------------------------------------------------------------------------
# Logical-error-rate sampling and fitting
# ---------------------------------------------------------------------------


def sample_detector_logical_error_probability(
    *,
    experiment_name: str,
    physical_error_probability: float,
    analyzer: DetectorErrorModelShotAnalyzer,
    max_shots: int,
    target_logical_errors: int | None,
    batch_size: int = 100_000,
    seed: int | None = None,
) -> LogicalErrorEstimate:
    """Estimate a LER up to a shot limit and optional exact error target.

    Pass ``target_logical_errors=None`` for a fixed-shot estimate. Otherwise,
    sampling stops at the target-th logical error or ``max_shots``, whichever
    comes first. The final batch is truncated at that error, preventing target
    overshoot. With partial postselection, accepted records are PyMatching-
    decoded.

    Experiments that postselect every detector use a sampling-only fast path:
    Stim returns bit-packed records, an accepted row must have zero detector
    events, and no matching call or unpacked Boolean detector matrix is needed.
    This optimization does not change the postselection or decoding policy.

    Call structure::

        sample_detector_logical_error_probability(...)
        ├── sampler = analyzer._circuit.compile_detector_sampler(...)
        │   ├── all detectors postselected, complementary gap disabled
        │   │   ├── sampler.sample(..., bit_packed=True)
        │   │   └── zero-syndrome acceptance and observable count
        │   └── otherwise
        │       ├── sampler.sample(...)
        │       └── analyzer.postselect_and_decode_samples(...)
        ├── optionally truncate at the target-th logical error
        └── LogicalErrorEstimate
    """
    if max_shots <= 0:
        raise ValueError("max_shots must be positive.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if target_logical_errors is not None and target_logical_errors <= 0:
        raise ValueError("target_logical_errors must be positive or None.")
    sampler = analyzer._circuit.compile_detector_sampler(seed=seed)
    complementary_gap_enabled = analyzer._complementary_gap_postselection_threshold is not None
    all_detectors_postselected = (
        analyzer._postselection_detector_indices.size == analyzer._num_detectors
    )
    can_use_zero_syndrome_fast_path = all_detectors_postselected and not complementary_gap_enabled
    accepted_shots = 0
    logical_errors = 0
    sampled_shots = 0
    shots_remaining = max_shots
    while shots_remaining and (
        target_logical_errors is None or logical_errors < target_logical_errors
    ):
        current_batch_size = min(batch_size, shots_remaining)
        if can_use_zero_syndrome_fast_path:
            packed_detectors, packed_observables = sampler.sample(
                current_batch_size,
                separate_observables=True,
                bit_packed=True,
            )
            acceptance_mask = ~packed_detectors.any(axis=1)
            packed_observable_flips = (packed_observables[:, 0] & 1) != 0
            logical_error_mask = acceptance_mask & packed_observable_flips
        else:
            detection_events, observable_flips = sampler.sample(
                current_batch_size,
                separate_observables=True,
            )
            acceptance_mask, logical_error_mask = analyzer.postselect_and_decode_samples(
                detection_events,
                observable_flips,
            )
        if target_logical_errors is not None:
            remaining_errors = target_logical_errors - logical_errors
            error_indices = np.flatnonzero(logical_error_mask)
            if len(error_indices) >= remaining_errors:
                current_batch_size = int(error_indices[remaining_errors - 1]) + 1
                acceptance_mask = acceptance_mask[:current_batch_size]
                logical_error_mask = logical_error_mask[:current_batch_size]
        accepted_shots += int(np.count_nonzero(acceptance_mask))
        logical_errors += int(np.count_nonzero(logical_error_mask))
        sampled_shots += current_batch_size
        shots_remaining -= current_batch_size
    return LogicalErrorEstimate(
        experiment_name=experiment_name,
        physical_error_probability=physical_error_probability,
        sampled_shots=sampled_shots,
        accepted_shots=accepted_shots,
        logical_errors=logical_errors,
    )


@dataclass(frozen=True)
class LogicalErrorEstimate:
    """Store the counts underlying one postselected LER estimate.

    ``sampled_shots`` counts every generated shot, ``accepted_shots`` counts
    shots surviving postselection, and ``logical_errors`` counts accepted shots
    that the decoder classified incorrectly. Keeping counts instead of only
    ratios supports acceptance reporting and binomial confidence intervals.
    """

    experiment_name: str
    physical_error_probability: float
    sampled_shots: int
    accepted_shots: int
    logical_errors: int

    @property
    def acceptance_probability(self) -> float:
        """Return accepted shots divided by all sampled shots."""
        return self.accepted_shots / self.sampled_shots

    @property
    def logical_error_probability(self) -> float:
        """Return logical failures conditioned on protocol acceptance.

        The result is ``nan`` when postselection rejects every sampled shot.
        """
        if self.accepted_shots == 0:
            return float("nan")
        return self.logical_errors / self.accepted_shots

    def wilson_interval(
        self,
        *,
        z_score: float = 1.959963984540054,
    ) -> tuple[float, float]:
        """Return the Wilson score interval for the conditional LER.

        ``z_score`` selects the confidence level and defaults to approximately
        95%. The result is ``(nan, nan)`` when no shot was accepted.
        """
        if self.accepted_shots == 0:
            return float("nan"), float("nan")
        probability = self.logical_error_probability
        denominator = 1 + z_score**2 / self.accepted_shots
        center = (probability + z_score**2 / (2 * self.accepted_shots)) / denominator
        half_width = (
            z_score
            * np.sqrt(
                probability * (1 - probability) / self.accepted_shots
                + z_score**2 / (4 * self.accepted_shots**2)
            )
            / denominator
        )
        return max(0.0, center - half_width), min(1.0, center + half_width)


def fit_power_law(estimates: Sequence[LogicalErrorEstimate]) -> tuple[float, float]:
    """Fit ``logical_error_probability = coefficient * p**exponent``.

    Zero-error estimates are excluded. At least two nonzero estimates are
    required. The unweighted nonlinear least-squares fit is performed in
    linear probability space, matching the Hirano Figure 9 audit. A log-linear
    estimate supplies only the optimizer's initial parameters. The fitted
    exponent summarizes finite-window LER scaling; it is not independently a
    circuit-distance proof.

    Call structure::

        fit_power_law(estimates)
        ├── select nonzero LogicalErrorEstimate values
        ├── numpy.polyfit(...) for initial parameters
        ├── scipy.optimize.curve_fit(...)
        └── (coefficient, exponent)
    """
    selected_estimates = [
        estimate
        for estimate in estimates
        if estimate.logical_errors > 0 and np.isfinite(estimate.logical_error_probability)
    ]
    if len(selected_estimates) < 2:
        raise ValueError("at least two nonzero logical-error estimates are required.")
    physical_probabilities = np.asarray(
        [estimate.physical_error_probability for estimate in selected_estimates]
    )
    logical_probabilities = np.asarray(
        [estimate.logical_error_probability for estimate in selected_estimates]
    )
    initial_exponent, initial_log_coefficient = np.polyfit(
        np.log(physical_probabilities),
        np.log(logical_probabilities),
        deg=1,
    )
    from scipy.optimize import curve_fit

    fit, _ = curve_fit(
        lambda probability, coefficient, exponent: coefficient * probability**exponent,
        physical_probabilities,
        logical_probabilities,
        p0=(np.exp(initial_log_coefficient), initial_exponent),
        maxfev=100_000,
    )
    return float(fit[0]), float(fit[1])


# ---------------------------------------------------------------------------
# Shared physical-fault-signature inventory
# ---------------------------------------------------------------------------


def enumerate_circuit_physical_fault_signatures(
    circuit: stim.Circuit,
) -> CircuitPhysicalFaultInventory:
    """Inventory physical Pauli faults by detector and observable signature.

    One pass counts every elementary alternative in the supported noise model.
    Stim then maps each nonzero ``(detector, observable)`` effect back to its
    physical location. Unsupported active noise fails explicitly.

    Stim omits zero-effect faults. The inventory counts them as benign instead
    of storing them because they cannot affect a combination's signature.

    Call structure::

        enumerate_circuit_physical_fault_signatures(circuit)
        ├── count elementary alternatives in supported noisy instructions
        ├── circuit.explain_detector_error_model_errors()
        └── CircuitPhysicalFaultInventory
    """
    total_elementary_fault_cases = 0
    total_physical_locations = 0
    pending_circuit_blocks = [(1, circuit)]
    while pending_circuit_blocks:
        repeat_multiplicity, circuit_block = pending_circuit_blocks.pop()
        for instruction in circuit_block:
            if isinstance(instruction, stim.CircuitRepeatBlock):
                pending_circuit_blocks.append(
                    (repeat_multiplicity * instruction.repeat_count, instruction.body_copy())
                )
                continue
            gate = stim.gate_data(instruction.name)
            if not gate.is_noisy_gate:
                continue
            probabilities = instruction.gate_args_copy()
            if not (probabilities and probabilities[0] > 0):
                continue
            targets = instruction.targets_copy()
            if instruction.name == "DEPOLARIZE1":
                total_elementary_fault_cases += repeat_multiplicity * 3 * len(targets)
                total_physical_locations += repeat_multiplicity * len(targets)
            elif instruction.name == "DEPOLARIZE2":
                if len(targets) % 2:
                    raise ValueError("DEPOLARIZE2 targets must form qubit pairs.")
                target_pair_count = len(targets) // 2
                total_elementary_fault_cases += repeat_multiplicity * 15 * target_pair_count
                total_physical_locations += repeat_multiplicity * target_pair_count
            elif instruction.name in ("X_ERROR", "Y_ERROR", "Z_ERROR"):
                total_elementary_fault_cases += repeat_multiplicity * len(targets)
                total_physical_locations += repeat_multiplicity * len(targets)
            elif gate.produces_measurements:
                total_elementary_fault_cases += repeat_multiplicity * instruction.num_measurements
                total_physical_locations += repeat_multiplicity * instruction.num_measurements
            else:
                raise NotImplementedError(f"Unsupported physical fault channel: {instruction.name}")

    effectful_faults: list[PhysicalFaultCase] = []
    for explained_error in circuit.explain_detector_error_model_errors():
        detector_mask = 0
        observable_mask = 0
        for term in explained_error.dem_error_terms:
            target = term.dem_target
            if target.is_relative_detector_id():
                detector_mask ^= 1 << target.val
            elif target.is_logical_observable_id():
                observable_mask ^= 1 << target.val
            elif not target.is_separator():
                raise ValueError(f"unsupported detector-error-model target {target}.")
        for location in explained_error.circuit_error_locations:
            effectful_faults.append(
                PhysicalFaultCase(
                    location_key=_physical_fault_location_key(location),
                    location_description=_describe_physical_fault_location(location),
                    pauli_description=_describe_pauli_alternative(location),
                    detector_mask=detector_mask,
                    observable_mask=observable_mask,
                )
            )

    if len(effectful_faults) > total_elementary_fault_cases:
        raise RuntimeError("Stim explained more circuit faults than the noise model enumerates.")
    return CircuitPhysicalFaultInventory(
        effectful_faults=tuple(effectful_faults),
        total_elementary_fault_cases=total_elementary_fault_cases,
        total_physical_locations=total_physical_locations,
    )


@dataclass(frozen=True)
class PhysicalFaultCase:
    """One Pauli alternative and its packed detector/observable effect.

    ``location_key`` excludes the Pauli alternative, preventing alternatives at
    one circuit location from being counted as simultaneous faults.
    """

    location_key: Hashable
    location_description: str
    pauli_description: str
    detector_mask: int
    observable_mask: int


@dataclass(frozen=True)
class CircuitPhysicalFaultInventory:
    """Stored nonzero fault signatures and complete noise-model counts."""

    effectful_faults: tuple[PhysicalFaultCase, ...]
    total_elementary_fault_cases: int
    total_physical_locations: int

    @property
    def benign_fault_cases(self) -> int:
        """Return alternatives omitted by Stim because their effect is zero."""
        return self.total_elementary_fault_cases - len(self.effectful_faults)


def _physical_fault_location_key(location: stim.CircuitErrorLocation) -> Hashable:
    """Identify a circuit location independently of its Pauli alternative."""
    stack_frames = tuple(
        (frame.instruction_offset, frame.iteration_index) for frame in location.stack_frames
    )
    instruction_targets = location.instruction_targets
    return (
        stack_frames,
        instruction_targets.target_range_start,
        instruction_targets.target_range_end,
    )


def _describe_physical_fault_location(location: stim.CircuitErrorLocation) -> str:
    """Return a concise tick/instruction/qubit description."""
    instruction_targets = location.instruction_targets
    qubits = [target.gate_target.qubit_value for target in instruction_targets.targets_in_range]
    stack_frame = location.stack_frames[0]
    return (
        f"tick {location.tick_offset}, {instruction_targets.gate} on qubit(s) {qubits} "
        f"(instruction {stack_frame.instruction_offset}, iteration {stack_frame.iteration_index})"
    )


def _describe_pauli_alternative(location: stim.CircuitErrorLocation) -> str:
    """Describe the Pauli alternative or measurement flip."""
    flipped_measurement = location.flipped_measurement
    if flipped_measurement is not None and flipped_measurement.record_index is not None:
        return f"measurement result flip (record {flipped_measurement.record_index})"
    return "*".join(
        f"{target.gate_target.pauli_type}{target.gate_target.qubit_value}"
        for target in location.flipped_pauli_product
    )


# ---------------------------------------------------------------------------
# Physical-fault-distance assessment
# ---------------------------------------------------------------------------


def assess_fully_postselected_circuit_fault_distance(
    *,
    experiment_name: str,
    circuit: stim.Circuit,
    max_exhaustive_weight: int = 2,
    find_witness_weight: int = 3,
) -> PhysicalFaultDistanceAssessment:
    """Certify physical fault distance when every detector is postselected.

    A combination is malignant exactly when its detector mask is zero and its
    observable mask is nonzero. Singles and distinct-location pairs are
    exhausted; if both pass, a meet-in-the-middle search finds a triple witness.

    Call structure::

        assess_fully_postselected_circuit_fault_distance(circuit=circuit, ...)
        ├── enumerate_circuit_physical_fault_signatures(circuit)
        ├── classify single faults
        ├── _summarize_malignant_pairs(...)             (exhaustive weight 2)
        ├── _find_malignant_triple_by_signature(...)    (weight-3 witness)
        └── PhysicalFaultDistanceAssessment
    """
    if max_exhaustive_weight not in (1, 2):
        raise NotImplementedError("exhaustive search is implemented for weight 1 and 2 only.")
    if find_witness_weight not in (0, 3):
        raise NotImplementedError("malignant-witness search is implemented for weight 3 only.")
    if circuit.num_observables != 1:
        raise ValueError("physical-fault-distance certification requires exactly one observable.")
    inventory = enumerate_circuit_physical_fault_signatures(circuit)
    faults = inventory.effectful_faults

    malignant_single_faults = tuple(
        FaultCombinationWitness((fault,))
        for fault in faults
        if fault.detector_mask == 0 and fault.observable_mask != 0
    )
    rejected_single_faults = sum(fault.detector_mask != 0 for fault in faults)
    benign_single_faults = inventory.benign_fault_cases + sum(
        fault.detector_mask == 0 and fault.observable_mask == 0 for fault in faults
    )
    if (
        benign_single_faults + rejected_single_faults + len(malignant_single_faults)
        != inventory.total_elementary_fault_cases
    ):
        raise RuntimeError("single-fault accounting is not exhaustive.")

    malignant_pair_count = 0
    first_malignant_pair: FaultCombinationWitness | None = None
    pairs_exhaustively_checked = False
    if not malignant_single_faults and max_exhaustive_weight >= 2:
        malignant_pair_count, first_malignant_pair = _summarize_malignant_pairs(faults)
        pairs_exhaustively_checked = True

    first_malignant_triple: FaultCombinationWitness | None = None
    if pairs_exhaustively_checked and malignant_pair_count == 0 and find_witness_weight == 3:
        first_malignant_triple = _find_malignant_triple_by_signature(faults)

    return PhysicalFaultDistanceAssessment(
        experiment_name=experiment_name,
        total_elementary_fault_cases=inventory.total_elementary_fault_cases,
        total_physical_locations=inventory.total_physical_locations,
        malignant_single_faults=malignant_single_faults,
        malignant_pair_count=malignant_pair_count,
        first_malignant_pair=first_malignant_pair,
        first_malignant_triple=first_malignant_triple,
    )


def _summarize_malignant_pairs(
    faults: Sequence[PhysicalFaultCase],
) -> tuple[int, FaultCombinationWitness | None]:
    """Count distinct-location malignant pairs and return the first.

    Equal detector masks cancel. For one observable, the pair is then malignant
    exactly when its observable masks differ. Grouping by detector mask reduces
    the exhaustive search to those candidate pairs.
    """
    faults_by_detector_mask: dict[int, list[PhysicalFaultCase]] = defaultdict(list)
    for fault in faults:
        faults_by_detector_mask[fault.detector_mask].append(fault)

    malignant_pair_count = 0
    first_malignant_pair: FaultCombinationWitness | None = None
    for grouped_faults in faults_by_detector_mask.values():
        observable_silent = [fault for fault in grouped_faults if fault.observable_mask == 0]
        observable_flipping = [fault for fault in grouped_faults if fault.observable_mask != 0]
        if not observable_silent or not observable_flipping:
            continue
        silent_location_counts = Counter(fault.location_key for fault in observable_silent)
        same_location_pairs = sum(
            silent_location_counts[fault.location_key] for fault in observable_flipping
        )
        malignant_pair_count += (
            len(observable_silent) * len(observable_flipping) - same_location_pairs
        )
        if first_malignant_pair is None:
            first_malignant_pair = next(
                (
                    FaultCombinationWitness((silent_fault, flipping_fault))
                    for silent_fault in observable_silent
                    for flipping_fault in observable_flipping
                    if silent_fault.location_key != flipping_fault.location_key
                ),
                None,
            )
    return malignant_pair_count, first_malignant_pair


def _find_malignant_triple_by_signature(
    faults: Sequence[PhysicalFaultCase],
) -> FaultCombinationWitness | None:
    """Find one weight-three witness without materializing every candidate pair.

    For each distinct-location pair, look up a third fault that cancels its
    detector mask and flips its logical parity. Return as soon as all three
    locations differ; exhausting other pairs cannot strengthen this witness.
    """
    faults_by_signature: dict[tuple[int, int], list[PhysicalFaultCase]] = defaultdict(list)
    for fault in faults:
        faults_by_signature[(fault.detector_mask, fault.observable_mask)].append(fault)

    for left_index, left_fault in enumerate(faults):
        for right_index in range(left_index + 1, len(faults)):
            right_fault = faults[right_index]
            if left_fault.location_key == right_fault.location_key:
                continue
            required_signature = (
                left_fault.detector_mask ^ right_fault.detector_mask,
                left_fault.observable_mask ^ right_fault.observable_mask ^ 1,
            )
            for third_fault in faults_by_signature.get(required_signature, ()):
                if third_fault.location_key not in (
                    left_fault.location_key,
                    right_fault.location_key,
                ):
                    return FaultCombinationWitness((left_fault, right_fault, third_fault))
    return None


@dataclass(frozen=True)
class FaultCombinationWitness:
    """Distinct-location physical faults and their combined signature."""

    faults: tuple[PhysicalFaultCase, ...]

    @property
    def weight(self) -> int:
        """Return the number of fault locations."""
        return len(self.faults)

    @property
    def combined_detector_mask(self) -> int:
        """Return the XOR of all detector masks."""
        combined_mask = 0
        for fault in self.faults:
            combined_mask ^= fault.detector_mask
        return combined_mask

    @property
    def combined_observable_mask(self) -> int:
        """Return the XOR of all observable masks."""
        combined_mask = 0
        for fault in self.faults:
            combined_mask ^= fault.observable_mask
        return combined_mask

    def to_dataframe(self):
        """Return the witness as a pandas ``DataFrame``."""
        import pandas as pd

        return pd.DataFrame(
            {
                "fault": index,
                "location": fault.location_description,
                "pauli_alternative": fault.pauli_description,
                "detector_mask": fault.detector_mask,
                "observable_mask": fault.observable_mask,
            }
            for index, fault in enumerate(self.faults, start=1)
        )


@dataclass(frozen=True)
class PhysicalFaultDistanceAssessment:
    """Result of an exact physical-fault-distance assessment.

    ``certified_distance`` is the first malignant weight found, or ``None`` in
    the requested search. A distance-3 certificate exhausts singles and pairs
    and includes ``first_malignant_triple``.
    """

    experiment_name: str
    total_elementary_fault_cases: int
    total_physical_locations: int
    malignant_single_faults: tuple[FaultCombinationWitness, ...]
    malignant_pair_count: int
    first_malignant_pair: FaultCombinationWitness | None
    first_malignant_triple: FaultCombinationWitness | None

    @property
    def certified_distance(self) -> int | None:
        """Return the first malignant weight found by the requested search."""
        if self.malignant_single_faults:
            return 1
        if self.malignant_pair_count:
            return 2
        if self.first_malignant_triple is not None:
            return 3
        return None
