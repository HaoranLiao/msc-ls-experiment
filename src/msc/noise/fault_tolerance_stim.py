"""Logical-error-rate and fault-tolerance analysis for Stim detector circuits.

Call structure::

    analyzer = DetectorErrorModelShotAnalyzer(circuit, ...)
    ├── sample_detector_logical_error_probability(analyzer=analyzer, ...)
    │   ├── analyzer.postselect_and_decode_samples(sampled detector records)
    │   └── LogicalErrorEstimate
    ├── analyzer.decode_complementary_gap_samples(...)
    └── assess_circuit_single_fault_tolerance(..., analyzer=analyzer)
        ├── count represented single-fault cases
        └── SingleFaultAssessmentResult

    fit_power_law(sequence[LogicalErrorEstimate])
    └── (coefficient, exponent)
"""

from __future__ import annotations

import math
from collections.abc import Sequence
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
    shots: int,
    batch_size: int = 100_000,
    seed: int | None = None,
) -> LogicalErrorEstimate:
    """Estimate logical error rate conditioned on detector-based acceptance.

    Detector and observable records are sampled in batches. With partial or no
    postselection, ``analyzer.postselect_and_decode_samples`` applies the
    requested rejection policy and PyMatching-decodes accepted detector events.

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
        └── LogicalErrorEstimate
    """
    if shots <= 0:
        raise ValueError("shots must be positive.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    sampler = analyzer._circuit.compile_detector_sampler(seed=seed)
    complementary_gap_enabled = analyzer._complementary_gap_postselection_threshold is not None
    all_detectors_postselected = (
        analyzer._postselection_detector_indices.size == analyzer._num_detectors
    )
    can_use_zero_syndrome_fast_path = all_detectors_postselected and not complementary_gap_enabled
    accepted_shots = 0
    logical_errors = 0
    shots_remaining = shots
    while shots_remaining:
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
        accepted_shots += int(np.count_nonzero(acceptance_mask))
        logical_errors += int(np.count_nonzero(logical_error_mask))
        shots_remaining -= current_batch_size
    return LogicalErrorEstimate(
        experiment_name=experiment_name,
        physical_error_probability=physical_error_probability,
        sampled_shots=shots,
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
# Exhaustive single-fault assessment
# ---------------------------------------------------------------------------


def assess_circuit_single_fault_tolerance(
    *,
    experiment_name: str,
    analyzer: DetectorErrorModelShotAnalyzer,
) -> SingleFaultAssessmentResult:
    """Assess single-fault coverage and classify every represented fault case.

    The total case count makes the coverage auditable: it detects whether any
    fault mechanism was omitted from the benign, rejected, correctly decoded,
    and malignant classifications. Each active ``DEPOLARIZE1`` contributes
    three cases per qubit, each ``DEPOLARIZE2`` contributes 15 per qubit pair,
    and each noisy measurement contributes one result-flip case. This total is
    not the number of faults sampled in one stochastic shot or the number of
    logical errors. Faults absent from Stim's detector model are benign.

    Call structure::

        assess_circuit_single_fault_tolerance(analyzer=analyzer, ...)
        ├── scan circuit instructions to count all single-fault cases
        ├── circuit.explain_detector_error_model_errors()
        │   └── analyzer.postselect_and_decode_samples(each fault effect)
        ├── validate exhaustive fault classification
        ├── circuit.shortest_graphlike_error()
        └── SingleFaultAssessmentResult
    """
    circuit = analyzer._circuit

    # Count every elementary fault represented by the supported noise channels.
    # Repeat blocks are traversed without flattening the circuit.
    total_single_fault_cases = 0
    pending_circuit_blocks = [(1, circuit)]
    while pending_circuit_blocks:
        repeat_multiplicity, circuit_block = pending_circuit_blocks.pop()
        for instruction in circuit_block:
            if isinstance(instruction, stim.CircuitRepeatBlock):
                pending_circuit_blocks.append(
                    (
                        repeat_multiplicity * instruction.repeat_count,
                        instruction.body_copy(),
                    )
                )
                continue

            targets = instruction.targets_copy()
            probabilities = instruction.gate_args_copy()
            active_noise_channel = bool(probabilities) and probabilities[0] > 0
            if instruction.name == "DEPOLARIZE1":
                if active_noise_channel:
                    total_single_fault_cases += repeat_multiplicity * 3 * len(targets)
            elif instruction.name == "DEPOLARIZE2":
                if len(targets) % 2:
                    raise ValueError("DEPOLARIZE2 targets must form qubit pairs.")
                if active_noise_channel:
                    target_pair_count = len(targets) // 2
                    total_single_fault_cases += repeat_multiplicity * 15 * target_pair_count
            elif stim.gate_data(instruction.name).produces_measurements and active_noise_channel:
                total_single_fault_cases += repeat_multiplicity * instruction.num_measurements

    # Classify every fault with a nontrivial detector or observable effect.
    explained_fault_cases = 0
    rejected_faults = 0
    correctly_decoded_faults = 0
    malignant_fault_locations: list[str] = []

    for explained_error in circuit.explain_detector_error_model_errors():
        locations = explained_error.circuit_error_locations
        if not locations:
            continue
        detection_events = np.zeros((1, circuit.num_detectors), dtype=bool)
        observable_flips = np.zeros((1, circuit.num_observables), dtype=bool)
        for term in explained_error.dem_error_terms:
            target = term.dem_target
            if target.is_relative_detector_id():
                detection_events[0, target.val] ^= True
            elif target.is_logical_observable_id():
                observable_flips[0, target.val] ^= True
            elif not target.is_separator():
                raise ValueError(f"unsupported detector-error-model target {target}.")
        acceptance_mask, logical_error_mask = analyzer.postselect_and_decode_samples(
            detection_events,
            observable_flips,
        )
        location_count = len(locations)
        explained_fault_cases += location_count
        if not acceptance_mask[0]:
            rejected_faults += location_count
        elif logical_error_mask[0]:
            malignant_fault_locations.extend(str(location) for location in locations)
        else:
            correctly_decoded_faults += location_count

    # Infer benign faults and check that the classification is exhaustive.
    classified_effectful_fault_cases = (
        rejected_faults + correctly_decoded_faults + len(malignant_fault_locations)
    )
    if classified_effectful_fault_cases != explained_fault_cases:
        raise RuntimeError("single-fault classification missed an effectful fault.")
    if explained_fault_cases > total_single_fault_cases:
        raise RuntimeError("Stim explained more circuit faults than the noise model enumerates.")
    benign_faults = total_single_fault_cases - explained_fault_cases

    # Report the classification together with Stim's detector-model distance.
    return SingleFaultAssessmentResult(
        experiment_name=experiment_name,
        total_single_fault_cases=total_single_fault_cases,
        benign_faults=benign_faults,
        rejected_faults=rejected_faults,
        correctly_decoded_faults=correctly_decoded_faults,
        graphlike_distance=len(circuit.shortest_graphlike_error()),
        malignant_fault_locations=tuple(malignant_fault_locations),
    )


@dataclass(frozen=True)
class SingleFaultAssessmentResult:
    """Classification of every single-fault case in the circuit noise model.

    ``total_single_fault_cases`` is useful as a coverage check: it must equal
    benign + rejected + correctly decoded + malignant, proving that every fault
    case represented by the stochastic noise channels was classified. It does
    not count logical errors, sampled fault occurrences, or deterministic faults
    appended by ``append_pauli_faults``.
    """

    experiment_name: str
    total_single_fault_cases: int
    benign_faults: int
    rejected_faults: int
    correctly_decoded_faults: int
    graphlike_distance: int
    malignant_fault_locations: tuple[str, ...]

    @property
    def malignant_faults(self) -> int:
        """Count accepted single faults that produce a logical failure."""
        return len(self.malignant_fault_locations)

    @property
    def passes_single_fault_check(self) -> bool:
        """Return whether no represented single fault is malignant."""
        return self.malignant_faults == 0
