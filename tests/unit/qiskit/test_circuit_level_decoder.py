"""Circuit-level decoding of the saved ibm_phoenix Ty tomography job."""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pytest
import stim

from msc.experiment.results_io import SavedHardwareJob, load_saved_hardware_job
from msc.paths import HARDWARE_RUNS_DIR
from msc.rotated_surface_code.decoding.circuit_level_decoder import CircuitLevelDecoder
from msc.state_transfer import common
from msc.state_transfer import to_d5_rotated_qiskit as d5
from msc.state_transfer.tomography_qiskit import TOMOGRAPHY_BASES, analyze_ty_tomography_counts
from msc.utils.normalize import qiskit_bit_values

SAVED_JOB_DIRECTORY = HARDWARE_RUNS_DIR / "tomography_20260923T202524"


@pytest.fixture(scope="module")
def saved_job() -> SavedHardwareJob:
    return load_saved_hardware_job(SAVED_JOB_DIRECTORY)


@pytest.fixture(scope="module")
def decoders(saved_job: SavedHardwareJob) -> dict[str, CircuitLevelDecoder]:
    return {
        basis: CircuitLevelDecoder(
            saved_job.circuits_by_basis[basis],
            calibration=saved_job.calibrations_by_basis[basis],
            tomography_config=saved_job.config,
            measurement_schema=saved_job.measurement_schemas_by_basis[basis],
            readout_basis=basis,
        )
        for basis in TOMOGRAPHY_BASES
    }


def _counts_from_records(
    decoder: CircuitLevelDecoder,
    records: np.ndarray,
) -> dict[str, int]:
    """Write Stim-ordered measurement rows as Qiskit count keys (bit 0 last)."""
    width = len(decoder.record_clbits)
    characters = np.full((len(records), width), ord("0"), dtype=np.uint8)
    characters[:, width - 1 - np.asarray(decoder.record_clbits)] += records.astype(np.uint8)
    return dict(Counter(row.tobytes().decode("ascii") for row in characters))


def _flip_records(
    counts: dict[str, int],
    schema: dict[str, tuple[str, ...]],
    records: tuple[tuple[str, str], ...],
) -> dict[str, int]:
    """Flip the classical bits of labeled records in every Qiskit count key."""
    clbit_by_record = {
        record: clbit
        for clbit, record in enumerate(
            (family, label) for family, labels in schema.items() for label in labels
        )
    }
    width = len(next(iter(counts)))
    key_positions = [width - 1 - clbit_by_record[record] for record in records]
    flipped_counts = {}
    for key, count in counts.items():
        characters = list(key)
        for position in key_positions:
            characters[position] = "1" if characters[position] == "0" else "0"
        flipped_counts["".join(characters)] = count
    return flipped_counts


# =============================================================================
# Calibrated Stim model
# =============================================================================


def test_noise_channels_follow_the_calibrated_submitted_operations(
    saved_job: SavedHardwareJob,
    decoders: dict[str, CircuitLevelDecoder],
) -> None:
    calibration = saved_job.calibrations_by_basis["X"]
    instructions = list(decoders["X"].circuit)
    checked = Counter()
    for instruction, next_instruction, third_instruction in zip(
        instructions,
        instructions[1:],
        instructions[2:],
        strict=False,
    ):
        targets = tuple(target.value for target in instruction.targets_copy())
        if instruction.name == "CZ" and next_instruction.name == "DEPOLARIZE2":
            assert next_instruction.gate_args_copy() == [
                pytest.approx(1.25 * calibration.operation_error("cz", targets))
            ]
            checked["cz"] += 1
        elif instruction.name == "SQRT_X" and next_instruction.name == "DEPOLARIZE1":
            assert next_instruction.gate_args_copy() == [
                pytest.approx(1.5 * calibration.operation_error("sx", targets))
            ]
            checked["sx"] += 1
        elif instruction.name == "M" and next_instruction.name == "R":
            # Composite readout and its independent reset-error proxy share measure_2.
            readout_error = calibration.operation_error("measure_2", targets)
            assert instruction.gate_args_copy() == [pytest.approx(readout_error)]
            assert third_instruction.name == "X_ERROR"
            assert third_instruction.gate_args_copy() == [pytest.approx(readout_error)]
            checked["measure_reset_2"] += 1
        elif (
            instruction.name == "X_ERROR"
            and next_instruction.name == "Y_ERROR"
            and third_instruction.name == "Z_ERROR"
        ):
            # Independent flips must compose to the Pauli-twirled T1/T2 channel.
            (x_flip,), (y_flip,), (z_flip,) = (
                channel.gate_args_copy()
                for channel in (instruction, next_instruction, third_instruction)
            )
            t1, t2 = calibration.relaxation_times[targets[0]]
            duration = -2 * t1 * math.log(1 - 2 * x_flip)
            assert y_flip == pytest.approx(x_flip)
            assert (duration / calibration.dt) == pytest.approx(round(duration / calibration.dt))
            composed_x = x_flip * (1 - y_flip) * (1 - z_flip) + (1 - x_flip) * y_flip * z_flip
            composed_z = z_flip * (1 - x_flip) * (1 - y_flip) + x_flip * y_flip * (1 - z_flip)
            assert composed_x == pytest.approx((1 - math.exp(-duration / t1)) / 4)
            assert composed_z == pytest.approx(
                (1 + math.exp(-duration / t1) - 2 * math.exp(-duration / t2)) / 4
            )
            checked["delay"] += 1
    assert min(checked.values()) > 0 and len(checked) == 4


def test_complementary_gap_detector_changes_only_the_logical_sector(
    saved_job: SavedHardwareJob,
    decoders: dict[str, CircuitLevelDecoder],
) -> None:
    gap_decoder = CircuitLevelDecoder(
        saved_job.circuits_by_basis["X"],
        calibration=saved_job.calibrations_by_basis["X"],
        tomography_config=saved_job.config,
        measurement_schema=saved_job.measurement_schemas_by_basis["X"],
        readout_basis="X",
        complementary_gap_threshold=0.0,
    )
    circuit = gap_decoder.circuit
    gap_detector = circuit.num_detectors - 1
    assert circuit.num_detectors == decoders["X"].circuit.num_detectors + 1
    assert gap_detector not in gap_decoder.analyzer._postselection_detector_indices
    for error in circuit.detector_error_model().flattened():
        if error.type == "error":
            targets = error.targets_copy()
            assert (stim.DemTarget.relative_detector_id(gap_detector) in targets) == (
                stim.DemTarget.logical_observable_id(0) in targets
            )

    # No standalone boundary edge lets matching switch sectors cheaply.
    assert all(
        (node, partner) != (gap_detector, None)
        for node, partner, _ in gap_decoder.analyzer._matching.edges()
    )

    # Threshold zero adds no rejection to the saved hardware counts.
    counts = saved_job.counts_by_basis["X"]
    assert sum(gap_decoder.decode_counts(counts)) == sum(decoders["X"].decode_counts(counts))


# =============================================================================
# Decoding under postselection
# =============================================================================


@pytest.mark.parametrize("basis", TOMOGRAPHY_BASES)
def test_accepted_single_faults_never_cause_logical_errors(
    decoders: dict[str, CircuitLevelDecoder],
    basis: common.SurfaceReadoutBasis,
) -> None:
    """Every modeled elementary suffix fault is rejected or decoded correctly.

    This is decoding correctness under postselection, distinct from the
    undetected-error distance: faults with no detection event must also leave
    the logical observable unchanged.
    """
    decoder = decoders[basis]
    detector_error_model = decoder.circuit.detector_error_model()
    fault_signatures = [
        error.targets_copy() for error in detector_error_model.flattened() if error.type == "error"
    ]
    detection_events = np.zeros((len(fault_signatures), decoder.circuit.num_detectors), dtype=bool)
    observable_flips = np.zeros((len(fault_signatures), 1), dtype=bool)
    for row, targets in enumerate(fault_signatures):
        for target in targets:
            if target.is_relative_detector_id():
                detection_events[row, target.val] = True
            elif target.is_logical_observable_id():
                observable_flips[row, 0] = True
    assert not (observable_flips[:, 0] & ~detection_events.any(axis=1)).any()

    accepted, logical_errors = decoder.analyzer.postselect_and_decode_samples(
        detection_events,
        observable_flips,
    )
    assert accepted.any() and not logical_errors.any()


# =============================================================================
# Hardware counts
# =============================================================================


@pytest.mark.parametrize("basis", TOMOGRAPHY_BASES)
def test_hardware_style_counts_round_trip_through_the_record_mapping(
    saved_job: SavedHardwareJob,
    decoders: dict[str, CircuitLevelDecoder],
    basis: common.SurfaceReadoutBasis,
) -> None:
    """Weighted keys decode exactly like the analyzer on the sampled Stim records."""
    decoder = decoders[basis]
    sampled_records = decoder.circuit.compile_sampler(seed=5).sample(20_000)
    records = np.concatenate([sampled_records, sampled_records[:5_000]])
    counts = _counts_from_records(decoder, records)
    assert max(counts.values()) > 1  # Multiplicities are exercised.

    detection_events, observable_flips = decoder.circuit.compile_m2d_converter().convert(
        measurements=records,
        separate_observables=True,
    )
    accepted, logical_errors = decoder.analyzer.postselect_and_decode_samples(
        detection_events,
        observable_flips,
    )
    expected = (
        int(np.count_nonzero(accepted & ~logical_errors)),
        int(np.count_nonzero(accepted & logical_errors)),
    )
    assert decoder.decode_counts(counts) == expected
    assert decoder.decode_counts(dict(reversed(counts.items()))) == expected
    assert expected[0] > expected[1] > 0

    # Flipping readout bits on the conjugate logical's support changes only the
    # logical outcome, and each transfer byproduct record enters it exactly once.
    schema = dict(saved_job.measurement_schemas_by_basis[basis])
    conjugate_support = (
        d5.TRANSFER_SPEC.logical_z if basis == "X" else d5.TRANSFER_SPEC.logical_x
    )
    readout_flip = tuple((f"surface_data_{basis.lower()}", label) for label in conjugate_support)
    if basis == "X":
        byproduct_flip = common.transfer_logical_frame_records(d5.TRANSFER_SPEC, "X")
    elif basis == "Z":
        byproduct_flip = tuple(
            ("interface_z", f"round{round_index}_Z1a")
            for round_index in range(1, common.LATTICE_SURGERY_ROUNDS + 1)
        )
    else:  # pragma: no cover - TOMOGRAPHY_BASES contains only X and Z.
        raise ValueError(basis)
    for flipped_records in (readout_flip, byproduct_flip):
        flipped_counts = _flip_records(counts, schema, flipped_records)
        assert decoder.decode_counts(flipped_counts) == expected[::-1]

    # A Steane input flag rejects every shot.
    rejected_counts = _flip_records(counts, schema, (("cult", "root_x"),))
    assert decoder.decode_counts(rejected_counts) == (0, 0)


def test_saved_policy_matches_phenomenological_acceptance_shot_by_shot(
    saved_job: SavedHardwareJob,
    decoders: dict[str, CircuitLevelDecoder],
) -> None:
    """Both decoders accept the same hardware shots: 5,015 X and 5,126 Z in total.

    Phenomenological parsing is slow, so the shot-by-shot comparison covers
    every circuit-level-accepted key plus the first 20,000 keys of each basis.
    """
    sampled_counts = {}
    circuit_level_accepted_shots = {}
    for basis, expected_accepted_shots in (("X", 5_015), ("Z", 5_126)):
        decoder = decoders[basis]
        counts = saved_job.counts_by_basis[basis]
        detection_events, observable_flips = decoder.circuit.compile_m2d_converter().convert(
            measurements=qiskit_bit_values(list(counts))[:, list(decoder.record_clbits)],
            separate_observables=True,
        )
        accepted, _ = decoder.analyzer.postselect_and_decode_samples(
            detection_events,
            observable_flips,
        )
        sampled_counts[basis] = {
            key: count
            for index, (key, count, is_accepted) in enumerate(
                zip(counts, counts.values(), accepted, strict=True)
            )
            if is_accepted or index < 20_000
        }
        circuit_level_accepted_shots[basis] = sum(
            count
            for count, is_accepted in zip(counts.values(), accepted, strict=True)
            if is_accepted
        )
        assert circuit_level_accepted_shots[basis] == expected_accepted_shots

    # The phenomenological decoder accepts exactly the circuit-level-accepted keys.
    phenomenological = analyze_ty_tomography_counts(
        sampled_counts,
        tomography_config=saved_job.config,
        measurement_schemas=saved_job.measurement_schemas_by_basis,
    )
    assert (phenomenological.accepted_x_shots, phenomenological.accepted_z_shots) == (
        circuit_level_accepted_shots["X"],
        circuit_level_accepted_shots["Z"],
    )
