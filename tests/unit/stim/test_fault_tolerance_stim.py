from __future__ import annotations

import numpy as np
import pytest
import stim
from scipy.optimize import curve_fit

from msc.noise.fault_tolerance_stim import (
    DetectorErrorModelShotAnalyzer,
    LogicalErrorEstimate,
    PhysicalFaultDistanceAssessment,
    assess_fully_postselected_circuit_fault_distance,
    enumerate_circuit_physical_fault_signatures,
    fit_power_law,
)


def test_fit_power_law_matches_hirano_curve_fit_reference() -> None:
    """Match Hirano's independent ``curve_fit`` convention on the same data.

    Hirano's linked Figure 9 workbook identifies its original fit as SciPy
    ``curve_fit`` of ``a * p**b`` in linear probability space. The reference
    call below spells out that fit independently of our production helper and
    uses a fixed ``(1000, 3)`` starting point instead of our helper's log-linear
    initializer. Comparing the two on the fixed eight-point no-idle Z-probe
    counts guards against changing our helper to log-space or weighted
    regression, using sampled rather than accepted shots, or selecting a
    different fitting window.

    Hirano's plotting script also calls ``sinter.plot_error_rate``, but leaves
    its ``line_fits`` argument at the default ``None``. Sinter therefore fits a
    binomial rate and uncertainty interval independently at each point; it does
    not fit or return a power-law exponent. Using ``sinter.fit_line_slope`` on
    log-transformed points here would instead define a different regression
    from Hirano's workbook fit.
    """
    probabilities = (5e-4, 6e-4, 8e-4, 1e-3, 1.2e-3, 1.4e-3, 1.7e-3, 2e-3)
    accepted_shots = (
        25_674_705,
        24_888_720,
        23_387_259,
        21_975_607,
        20_654_637,
        19_409_954,
        17_688_923,
        16_116_543,
    )
    logical_errors = (447, 633, 1_281, 2_062, 3_086, 4_289, 6_821, 9_455)
    estimates = [
        LogicalErrorEstimate(
            experiment_name="hirano_no_idle_z_error",
            physical_error_probability=probability,
            sampled_shots=30_000_000,
            accepted_shots=accepted,
            logical_errors=errors,
        )
        for probability, accepted, errors in zip(
            probabilities,
            accepted_shots,
            logical_errors,
            strict=True,
        )
    ]

    reference_fit, _ = curve_fit(
        lambda probability, coefficient, exponent: (coefficient * probability**exponent),
        np.asarray(probabilities),
        np.asarray(logical_errors) / np.asarray(accepted_shots),
        p0=(1_000, 3),
        maxfev=100_000,
    )
    coefficient, exponent = fit_power_law(estimates)

    assert coefficient == pytest.approx(reference_fit[0], rel=1e-7)
    assert exponent == pytest.approx(reference_fit[1], rel=1e-7)


# =============================================================================
# Exact physical-fault-distance certification
# =============================================================================
#
# These circuits control each elementary fault's (detector, observable)
# signature directly so the certified distance is known by construction. Formal
# failure is defined by detector parity under full postselection, not decoding:
# a fault combination is malignant exactly when its combined detector mask is
# zero while its combined observable mask is nonzero.


def _certify(circuit: stim.Circuit) -> PhysicalFaultDistanceAssessment:
    """Audit a fully postselected single-observable circuit's fault distance."""
    return assess_fully_postselected_circuit_fault_distance(
        experiment_name="synthetic",
        circuit=circuit,
    )


def test_distance_one_logical_only_single_fault_is_certified() -> None:
    """A single fault flipping only the observable certifies distance 1."""
    circuit = stim.Circuit("""
        R 0
        X_ERROR(0.001) 0
        M 0
        OBSERVABLE_INCLUDE(0) rec[-1]
    """)

    assessment = _certify(circuit)

    assert len(assessment.malignant_single_faults) == 1
    assert assessment.certified_distance == 1
    single_witness = assessment.malignant_single_faults[0]
    assert single_witness.combined_detector_mask == 0
    assert single_witness.combined_observable_mask == 1


def test_distance_two_detector_canceling_pair_is_certified() -> None:
    """Two faults whose detector effects cancel while one flips L0 give d=2.

    ``X_ERROR 0`` flips ``M0`` (detector only); ``X_ERROR 1`` flips ``M1``
    (detector and observable). The shared detector ``M0 xor M1`` cancels under
    combination while the observable ``M1`` survives.
    """
    circuit = stim.Circuit("""
        R 0 1
        X_ERROR(0.001) 0
        X_ERROR(0.001) 1
        M 0 1
        DETECTOR rec[-2] rec[-1]
        OBSERVABLE_INCLUDE(0) rec[-1]
    """)

    assessment = _certify(circuit)

    assert not assessment.malignant_single_faults
    assert assessment.malignant_pair_count > 0
    assert assessment.first_malignant_pair is not None
    assert assessment.first_malignant_pair.combined_detector_mask == 0
    assert assessment.first_malignant_pair.combined_observable_mask == 1
    assert assessment.certified_distance == 2


def test_distance_three_triple_witness_is_certified() -> None:
    """Three faults combine to L0 while every single and pair is detected.

    Each fault flips one measurement; the three detectors ``M0^M2``, ``M0^M1``,
    ``M1^M2`` make each single a two-detector event and each pair a nonzero
    detector event, so only the weight-three combination cancels detectors.
    """
    circuit = stim.Circuit("""
        R 0 1 2
        X_ERROR(0.001) 0
        X_ERROR(0.001) 1
        X_ERROR(0.001) 2
        M 0 1 2
        DETECTOR rec[-3] rec[-1]
        DETECTOR rec[-3] rec[-2]
        DETECTOR rec[-2] rec[-1]
        OBSERVABLE_INCLUDE(0) rec[-3]
    """)

    assessment = _certify(circuit)

    assert not assessment.malignant_single_faults
    assert assessment.malignant_pair_count == 0
    assert assessment.first_malignant_triple is not None
    assert assessment.certified_distance == 3

    witness = assessment.first_malignant_triple
    assert witness.weight == 3
    assert len({fault.location_key for fault in witness.faults}) == 3
    assert witness.combined_detector_mask == 0
    assert witness.combined_observable_mask == 1


def test_same_location_pauli_alternatives_share_one_location_key() -> None:
    """Every Pauli alternative of one ``DEPOLARIZE2`` shares one location key.

    Because pair and triple searches require distinct ``location_key`` values,
    grouping all 15 alternatives of a physical two-qubit gate under one key is
    what prevents two alternatives of the same location from being combined as a
    two-fault event. A batched two-pair ``DEPOLARIZE2`` still yields one key per
    pair, so distinct physical pairs stay separable.
    """
    circuit = stim.Circuit("""
        R 0 1 2 3
        DEPOLARIZE2(0.001) 0 1 2 3
        M 0 1 2 3
        DETECTOR rec[-4]
        DETECTOR rec[-3]
        DETECTOR rec[-2]
        DETECTOR rec[-1]
        OBSERVABLE_INCLUDE(0) rec[-1]
    """)

    inventory = enumerate_circuit_physical_fault_signatures(circuit)
    alternatives_per_location: dict[object, set[str]] = {}
    for fault in inventory.effectful_faults:
        alternatives_per_location.setdefault(fault.location_key, set()).add(fault.pauli_description)

    assert len(alternatives_per_location) == 2
    for pauli_alternatives in alternatives_per_location.values():
        assert len(pauli_alternatives) > 1


def test_hyperedge_pair_is_certified() -> None:
    """A malignant pair built from detector hyperedges is certified d=2.

    Each single fault triggers three coincident detectors. The physical signature
    enumerator combines the two faults into a detector-canceling,
    observable-flipping pair.
    """
    circuit = stim.Circuit("""
        R 0 1
        X_ERROR(0.001) 0
        X_ERROR(0.001) 1
        M 0 1
        DETECTOR rec[-2] rec[-1]
        DETECTOR rec[-2] rec[-1]
        DETECTOR rec[-2] rec[-1]
        OBSERVABLE_INCLUDE(0) rec[-1]
    """)

    assessment = _certify(circuit)

    assert assessment.malignant_pair_count >= 1
    assert assessment.certified_distance == 2


def test_noise_channel_coverage_counts_every_elementary_alternative() -> None:
    """Every supported stochastic channel contributes its exact case count.

    ``DEPOLARIZE1`` gives 3 alternatives, ``DEPOLARIZE2`` gives 15,
    ``X_ERROR`` / ``Z_ERROR`` give 1 each (reset and ``MPP`` faults use these),
    and a noisy measurement gives one result flip per measurement.
    """
    circuit = stim.Circuit("""
        R 0 1 2
        DEPOLARIZE1(0.001) 0
        DEPOLARIZE2(0.001) 0 1
        X_ERROR(0.001) 2
        Z_ERROR(0.001) 0
        M(0.001) 0 1 2
    """)

    inventory = enumerate_circuit_physical_fault_signatures(circuit)

    assert inventory.total_elementary_fault_cases == 3 + 15 + 1 + 1 + 3
    assert inventory.total_physical_locations == 1 + 1 + 1 + 1 + 3
    # No detectors or observables, so every alternative is a benign (0, 0) case.
    assert inventory.effectful_faults == ()
    assert inventory.benign_fault_cases == inventory.total_elementary_fault_cases


def test_unsupported_noise_channel_raises_rather_than_undercounting() -> None:
    """An active unsupported noise channel fails loudly, never silently skipped."""
    circuit = stim.Circuit()
    circuit.append("R", [0])
    circuit.append("PAULI_CHANNEL_1", [0], [0.001, 0.001, 0.001])
    circuit.append("M", [0])

    with pytest.raises(NotImplementedError, match="Unsupported physical fault channel"):
        enumerate_circuit_physical_fault_signatures(circuit)


# ---------------------------------------------------------------------------
# Tesseract decoding
# ---------------------------------------------------------------------------


def test_tesseract_decodes_like_matching_and_never_worse() -> None:
    """Tesseract agrees with matching on nearly every shot, in ordinary and two-sector decoding.

    On a matchable d = 3 memory, Tesseract's most likely errors keep correlated
    faults that matching splits, so it makes no more logical errors, and its
    sector-weight gaps track matching's.
    """
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=3,
        rounds=3,
        after_clifford_depolarization=0.01,
        before_measure_flip_probability=0.01,
        after_reset_flip_probability=0.01,
    )
    detection_events, observable_flips = circuit.compile_detector_sampler(seed=2).sample(
        2_000, separate_observables=True
    )
    logical_errors = {
        decoder: DetectorErrorModelShotAnalyzer(
            circuit, decoder=decoder
        ).postselect_and_decode_samples(detection_events, observable_flips)[1]
        for decoder in ("matching", "tesseract")
    }
    assert np.mean(logical_errors["matching"] == logical_errors["tesseract"]) > 0.95
    assert 0 < logical_errors["tesseract"].mean() <= logical_errors["matching"].mean()

    # A final detector over the observable's records constrains the logical sector.
    (observable,) = [
        instruction for instruction in circuit if instruction.name == "OBSERVABLE_INCLUDE"
    ]
    circuit.append("DETECTOR", observable.targets_copy())
    detection_events, observable_flips = circuit.compile_detector_sampler(seed=1).sample(
        2_000, separate_observables=True
    )
    gap_samples = {
        decoder: DetectorErrorModelShotAnalyzer(
            circuit, complementary_gap_postselection_threshold=0.0, decoder=decoder
        ).decode_complementary_gap_samples(detection_events, observable_flips)
        for decoder in ("matching", "tesseract")
    }
    (_, matching_errors, matching_gaps), (accepted, tesseract_errors, tesseract_gaps) = (
        gap_samples.values()
    )
    assert accepted.all() and np.isfinite(tesseract_gaps).all() and (tesseract_gaps >= 0).all()
    assert np.mean(matching_errors == tesseract_errors) > 0.95
    assert tesseract_errors.mean() <= matching_errors.mean()
    assert np.corrcoef(matching_gaps, tesseract_gaps)[0, 1] > 0.9


@pytest.mark.parametrize("decoder", ["matching", "tesseract"])
def test_sector_ties_are_broken_without_the_sampled_observable(decoder: str) -> None:
    """A tied shot must not decode correctly by construction.

    Equal-probability flips of the two data qubits explain a fired check in
    opposite logical sectors at equal weight. The primary decode is given the
    sampled sector as the gap detector's event, so keeping its prediction on a
    tie would always be right; ties go to the no-flip sector instead, so about
    half the tied shots are logical errors.
    """
    circuit = stim.Circuit(
        """
        X_ERROR(0.1) 0 1
        M 0 1
        DETECTOR rec[-1] rec[-2]
        OBSERVABLE_INCLUDE(0) rec[-2]
        DETECTOR rec[-2]
        """
    )
    detection_events, observable_flips = circuit.compile_detector_sampler(seed=4).sample(
        4_000, separate_observables=True
    )
    accepted, logical_errors, gaps = DetectorErrorModelShotAnalyzer(
        circuit, complementary_gap_postselection_threshold=0.0, decoder=decoder
    ).decode_complementary_gap_samples(detection_events, observable_flips)
    is_tied = detection_events[:, 0]
    assert accepted.all() and (gaps[is_tied] == 0).all() and (gaps[~is_tied] > 0).all()
    assert np.array_equal(logical_errors[is_tied], observable_flips[is_tied, 0])
    assert 0.4 < logical_errors[is_tied].mean() < 0.6
    # Untied shots decode as no flip; their errors are the undetectable double flips.
    assert np.array_equal(logical_errors[~is_tied], observable_flips[~is_tied, 0])
