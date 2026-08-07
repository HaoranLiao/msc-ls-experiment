"""Tests for the Fig. 9-style lattice-surgery-only Stim example."""

from __future__ import annotations

import numpy as np
import pytest
import stim

from examples.state_transfer_by_lattice_surgery import stim_logical_error_rate_ls as ls

# ===========================================================================
# Helpers
# ===========================================================================


def _is_noise_instruction(instruction: stim.CircuitInstruction) -> bool:
    return instruction.name.endswith("_ERROR") or instruction.name.startswith("DEPOLARIZE")


def _noise_record_positions(circuit: stim.Circuit) -> tuple[int, ...]:
    positions: list[int] = []
    measurement_count = 0
    for instruction in circuit:
        assert isinstance(instruction, stim.CircuitInstruction)
        gate = stim.gate_data(instruction.name)
        if _is_noise_instruction(instruction) or (
            gate.produces_measurements and instruction.gate_args_copy()
        ):
            positions.append(measurement_count)
        instruction_circuit = stim.Circuit()
        instruction_circuit.append(instruction)
        measurement_count += instruction_circuit.num_measurements
    return tuple(positions)


# ===========================================================================
# SET 1: Circuit partition, annotations, and record layout
# ===========================================================================


class TestLatticeSurgeryOnlyCircuit:
    @pytest.mark.parametrize("experiment", ls.LATTICE_SURGERY_EXPERIMENTS)
    def test_noiseless_experiment_has_no_detection_or_logical_events(
        self,
        experiment,
    ) -> None:
        circuit, layout = ls.build_lattice_surgery_only_circuit(
            0,
            experiment=experiment,
        )
        detection_events, observable_flips = circuit.compile_detector_sampler(seed=1).sample(
            32, separate_observables=True
        )
        accepted, logical_errors = ls.LatticeSurgeryOnlyShotAnalyzer(
            circuit,
            layout=layout,
        ).analyze(detection_events, observable_flips)

        assert circuit.num_measurements == layout.num_measurements == 163
        assert circuit.num_detectors == 129
        assert circuit.num_observables == 1
        assert accepted.all()
        assert not logical_errors.any()

    def test_noise_is_confined_to_surface_preparation_and_transfer(self) -> None:
        circuit, layout = ls.build_lattice_surgery_only_circuit(
            0.01,
            experiment=ls.TRANSFER_Z_ERROR,
        )
        positions = _noise_record_positions(circuit)
        ideal_input_end = layout.record("steane_qed").positions[-1] + 1
        noisy_transfer_end = layout.record("surface_x_recovery").positions[-1] + 1

        assert positions
        assert min(positions) == ideal_input_end
        assert max(positions) < noisy_transfer_end

    def test_layout_has_one_full_ideal_round_and_all_data_readout(self) -> None:
        _, layout = ls.build_lattice_surgery_only_circuit(
            0,
            experiment=ls.TRANSFER_X_ERROR,
        )

        assert len(layout.record("ideal_post_transfer_surface_z").positions) == 12
        assert len(layout.record("ideal_post_transfer_surface_x").positions) == 12
        assert len(layout.record("ideal_surface_data_z").positions) == 25


# ===========================================================================
# SET 2: Selectable detector postselection and circuit-level matching
# ===========================================================================


class TestDetectorErrorModelAnalysis:
    def test_full_postselection_flag_selects_surface_detectors(self) -> None:
        circuit, layout = ls.build_lattice_surgery_only_circuit(
            0,
            experiment=ls.TRANSFER_Z_ERROR,
        )

        full_analyzer = ls.LatticeSurgeryOnlyShotAnalyzer(
            circuit,
            layout=layout,
            full_post_selection=True,
        )
        correction_analyzer = ls.LatticeSurgeryOnlyShotAnalyzer(
            circuit,
            layout=layout,
            full_post_selection=False,
        )

        assert full_analyzer.num_postselection_detectors == 129
        assert correction_analyzer.num_postselection_detectors == 27

    def test_any_detector_event_is_rejected(self) -> None:
        circuit, layout = ls.build_lattice_surgery_only_circuit(
            0,
            experiment=ls.TRANSFER_Z_ERROR,
        )
        detection_events = np.zeros((1, circuit.num_detectors), dtype=bool)
        observable_flips = np.zeros((1, 1), dtype=bool)
        detection_events[0, 0] = True

        accepted, logical_errors = ls.LatticeSurgeryOnlyShotAnalyzer(
            circuit,
            layout=layout,
        ).analyze(detection_events, observable_flips)

        assert np.array_equal(accepted, [False])
        assert np.array_equal(logical_errors, [False])

    def test_undetected_observable_flip_is_a_logical_error(self) -> None:
        circuit, layout = ls.build_lattice_surgery_only_circuit(
            0,
            experiment=ls.TRANSFER_Z_ERROR,
        )
        detection_events = np.zeros((1, circuit.num_detectors), dtype=bool)
        observable_flips = np.ones((1, 1), dtype=bool)

        accepted, logical_errors = ls.LatticeSurgeryOnlyShotAnalyzer(
            circuit,
            layout=layout,
        ).analyze(detection_events, observable_flips)

        assert np.array_equal(accepted, [True])
        assert np.array_equal(logical_errors, [True])


# ===========================================================================
# SET 3: Exhaustive single-fault verification
# ===========================================================================


class TestSingleFaultVerification:
    @pytest.mark.parametrize(
        ("experiment", "expected_graphlike_distance"),
        tuple(zip(ls.LATTICE_SURGERY_EXPERIMENTS, (2, 3), strict=True)),
    )
    @pytest.mark.parametrize(
        ("full_post_selection", "expected_rejected", "expected_decoded"),
        ((True, 5137, 0), (False, 1055, 4082)),
    )
    def test_every_single_fault_is_benign_or_rejected(
        self,
        experiment,
        expected_graphlike_distance,
        full_post_selection,
        expected_rejected,
        expected_decoded,
    ) -> None:
        result = ls.verify_lattice_surgery_single_faults(
            experiment=experiment,
            full_post_selection=full_post_selection,
        )

        assert result.checked_faults == 5588
        assert result.benign_faults == 451
        assert result.rejected_faults == expected_rejected
        assert result.correctly_decoded_faults == expected_decoded
        assert result.malignant_faults == 0
        assert result.graphlike_distance == expected_graphlike_distance
        assert result.passes_single_fault_check
