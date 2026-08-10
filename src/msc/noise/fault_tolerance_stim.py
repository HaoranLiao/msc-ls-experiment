"""Logical-error-rate and fault-tolerance analysis for Stim detector circuits.

This module has one detector-record workflow::

    analyzer = DetectorErrorModelShotAnalyzer(circuit, ...)
    ├── sample_detector_logical_error_probability(analyzer=analyzer, ...)
    │   └── LogicalErrorEstimate
    │       └── fit_power_law(estimates)
    └── assess_circuit_single_fault_tolerance(..., analyzer=analyzer)
        └── SingleFaultAssessmentResult
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import stim

# ---------------------------------------------------------------------------
# Shared detector-record analyzer
# ---------------------------------------------------------------------------


class DetectorErrorModelShotAnalyzer:
    """Postselect and PyMatching-decode detector samples.

    The circuit must have detectors and one observable. Selected detectors
    reject shots; accepted shots are logical errors when the decoded and sampled
    observable flips disagree.

    Call structure::

        analyzer = DetectorErrorModelShotAnalyzer(circuit, ...)
        ├── sample_detector_logical_error_probability(analyzer=analyzer, ...)
        │   └── analyzer.analyze(sampled detector records)
        └── assess_circuit_single_fault_tolerance(..., analyzer=analyzer)
            └── analyzer.analyze(each explained single-fault effect)
    """

    def __init__(
        self,
        circuit: stim.Circuit,
        *,
        postselection_detector_indices: Sequence[int] = (),
    ) -> None:
        import pymatching

        selected_indices = tuple(postselection_detector_indices)
        if len(selected_indices) != len(set(selected_indices)):
            raise ValueError("postselection_detector_indices must be unique.")
        if any(not 0 <= index < circuit.num_detectors for index in selected_indices):
            raise ValueError("a postselection detector index is outside the circuit.")
        if circuit.num_observables != 1:
            raise ValueError("logical-error sampling requires exactly one observable.")
        self._circuit = circuit.copy()
        detector_error_model = self._circuit.detector_error_model(decompose_errors=True)
        self._matching = pymatching.Matching.from_detector_error_model(detector_error_model)
        self._num_detectors = self._circuit.num_detectors
        self._postselection_detector_indices = np.asarray(
            selected_indices,
            dtype=np.intp,
        )

    def analyze(
        self,
        detection_events: np.ndarray,
        observable_flips: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(accepted, accepted_but_incorrectly_decoded)`` masks."""
        detection_events = np.asarray(detection_events, dtype=bool)
        observable_flips = np.asarray(observable_flips, dtype=bool)
        if detection_events.ndim != 2 or detection_events.shape[1] != self._num_detectors:
            raise ValueError(
                "detection_events must have shape "
                f"(n, {self._num_detectors}), got {detection_events.shape}."
            )
        if observable_flips.shape != (len(detection_events), 1):
            raise ValueError(
                "observable_flips must have shape "
                f"({len(detection_events)}, 1), got {observable_flips.shape}."
            )
        accepted = (
            ~detection_events[:, self._postselection_detector_indices].any(axis=1)
            if self._postselection_detector_indices.size
            else np.ones(len(detection_events), dtype=bool)
        )
        logical_errors = np.zeros(len(detection_events), dtype=bool)
        selected_indices = np.flatnonzero(accepted)
        if selected_indices.size == 0:
            return accepted, logical_errors
        predictions = np.asarray(
            self._matching.decode_batch(detection_events[selected_indices]),
            dtype=bool,
        ).reshape(len(selected_indices), 1)
        logical_errors[selected_indices] = np.any(
            predictions != observable_flips[selected_indices],
            axis=1,
        )
        return accepted, logical_errors


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
    postselection, ``analyzer.analyze`` applies the requested rejection policy
    and PyMatching-decodes accepted detector events.

    Experiments that postselect every detector use a sampling-only fast path:
    Stim returns bit-packed records, an accepted row must have zero detector
    events, and no matching call or unpacked Boolean detector matrix is needed.
    This optimization does not change the postselection or decoding policy.

    Call structure::

        analyzer._circuit.compile_detector_sampler().sample(...)
        ├── all detectors postselected
        │   └── bit-packed zero-syndrome acceptance and observable count
        ├── partial or no postselection
        │   └── analyzer.analyze(...)
        │       └── PyMatching.decode_batch(...)
        └── LogicalErrorEstimate
    """
    if shots <= 0:
        raise ValueError("shots must be positive.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    sampler = analyzer._circuit.compile_detector_sampler(seed=seed)
    postselects_all = (
        analyzer._postselection_detector_indices.size == analyzer._num_detectors
    )
    accepted_shots = 0
    logical_errors = 0
    shots_remaining = shots
    while shots_remaining:
        selected_batch_size = min(batch_size, shots_remaining)
        if postselects_all:
            packed_detectors, packed_observables = sampler.sample(
                selected_batch_size,
                separate_observables=True,
                bit_packed=True,
            )
            accepted = ~packed_detectors.any(axis=1)
            error_mask = accepted & ((packed_observables[:, 0] & 1) != 0)
        else:
            detection_events, observable_flips = sampler.sample(
                selected_batch_size,
                separate_observables=True,
            )
            accepted, error_mask = analyzer.analyze(detection_events, observable_flips)
        accepted_shots += int(np.count_nonzero(accepted))
        logical_errors += int(np.count_nonzero(error_mask))
        shots_remaining -= selected_batch_size
    return LogicalErrorEstimate(
        experiment_name=experiment_name,
        physical_error_probability=physical_error_probability,
        sampled_shots=shots,
        accepted_shots=accepted_shots,
        logical_errors=logical_errors,
    )


@dataclass(frozen=True)
class LogicalErrorEstimate:
    """Postselected LER counts returned by the sampling function above."""

    experiment_name: str
    physical_error_probability: float
    sampled_shots: int
    accepted_shots: int
    logical_errors: int

    @property
    def acceptance_probability(self) -> float:
        """Return the fraction of sampled shots accepted by the protocol."""
        return self.accepted_shots / self.sampled_shots

    @property
    def logical_error_probability(self) -> float:
        """Return logical failures divided by accepted shots."""
        if self.accepted_shots == 0:
            return float("nan")
        return self.logical_errors / self.accepted_shots

    def wilson_interval(self, *, z_score: float = 1.959963984540054) -> tuple[float, float]:
        """Return a Wilson interval using the stored accepted-shot counts."""
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
    required. The exponent summarizes the observed leading-order LER scaling;
    it is not independently a circuit-distance proof.

    Call structure::

        sample_detector_logical_error_probability(...) at several p values
        └── sequence[LogicalErrorEstimate]
            └── fit_power_law(estimates)
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
    exponent, log_coefficient = np.polyfit(
        np.log(physical_probabilities),
        np.log(logical_probabilities),
        deg=1,
    )
    return float(np.exp(log_coefficient)), float(exponent)


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

        analyzer = DetectorErrorModelShotAnalyzer(circuit, ...)
        └── assess_circuit_single_fault_tolerance(analyzer=analyzer, ...)
            ├── count single-fault cases represented by stochastic noise
            ├── circuit.explain_detector_error_model_errors()
            ├── analyzer.analyze(each nontrivial fault effect)
            ├── circuit.shortest_graphlike_error()
            └── SingleFaultAssessmentResult
    """
    circuit = analyzer._circuit

    # Count the complete single-fault classification denominator.
    total_single_fault_cases = 0
    circuit_blocks = [(1, circuit)]
    while circuit_blocks:
        repeat_multiplicity, circuit_block = circuit_blocks.pop()
        for instruction in circuit_block:
            if isinstance(instruction, stim.CircuitRepeatBlock):
                circuit_blocks.append(
                    (
                        repeat_multiplicity * instruction.repeat_count,
                        instruction.body_copy(),
                    )
                )
                continue
            targets = instruction.targets_copy()
            probabilities = instruction.gate_args_copy()
            is_active_channel = bool(probabilities) and probabilities[0] > 0
            if instruction.name == "DEPOLARIZE1" and is_active_channel:
                total_single_fault_cases += repeat_multiplicity * 3 * len(targets)
            elif instruction.name == "DEPOLARIZE2":
                if len(targets) % 2:
                    raise ValueError("DEPOLARIZE2 targets must form qubit pairs.")
                if is_active_channel:
                    total_single_fault_cases += repeat_multiplicity * 15 * (len(targets) // 2)
            elif stim.gate_data(instruction.name).produces_measurements and is_active_channel:
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
        accepted, logical_errors = analyzer.analyze(
            detection_events,
            observable_flips,
        )
        location_count = len(locations)
        explained_fault_cases += location_count
        if not accepted[0]:
            rejected_faults += location_count
        elif logical_errors[0]:
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
        """Return the number of accepted, incorrectly decoded single faults."""
        return len(self.malignant_fault_locations)

    @property
    def passes_single_fault_check(self) -> bool:
        """Return whether every single fault is benign, rejected, or decoded."""
        return self.malignant_faults == 0
