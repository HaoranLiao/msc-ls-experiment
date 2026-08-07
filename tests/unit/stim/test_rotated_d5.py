"""Tests for the rotated distance-5 surface-code patch in Stim."""

from __future__ import annotations

import pytest
import stim

from msc.surface_code import rotated_d5_stim as d5
from msc.surface_code import rotated_helper_stim as rotated_helper
from msc.utils.circuit_stim import two_qubit_depth

NUM_DATA_QUBITS = 25
NUM_Z_ANCILLAS = 12
NUM_X_ANCILLAS = 12
Z_NAMES = tuple(d5.Z_STABILIZERS)
X_NAMES = tuple(d5.X_STABILIZERS)
MERGE_X_NAMES = tuple(name for name in X_NAMES if name not in {"X_ab", "X_cd"})


# ===========================================================================
# Helpers
# ===========================================================================


def _scheduled_round_circuit(selected_names: tuple[str, ...]) -> stim.Circuit:
    circuit = stim.Circuit()
    d5.append_scheduled_stabilizer_round(
        circuit,
        surface_qubits=tuple(range(NUM_DATA_QUBITS)),
        selected_stabilizer_names=selected_names,
        ancillas=tuple(range(NUM_DATA_QUBITS, NUM_DATA_QUBITS + len(selected_names))),
        round_label="scheduled",
        add_ticks=True,
    )
    return circuit


# ===========================================================================
# SET 1: Patch geometry and canonical measurement schedule validation
# ===========================================================================


class TestPatchGeometry:
    def test_expected_geometry(self) -> None:
        assert d5.SURFACE_ORDER == tuple("abcdefghijklmnopqrstuvwxy")
        assert len(d5.Z_STABILIZERS) == NUM_Z_ANCILLAS
        assert len(d5.X_STABILIZERS) == NUM_X_ANCILLAS
        assert d5.DEFAULT_LOGICAL_Z == ("a", "b", "c", "d", "e")
        assert d5.DEFAULT_LOGICAL_X == ("a", "f", "k", "p", "u")
        assert d5.X_STABILIZERS["X_ab"] == ("a", "b")
        assert d5.X_STABILIZERS["X_cd"] == ("c", "d")

    def test_logical_representatives_match_stabilizer_products(self) -> None:
        assert len(d5.LOGICAL_Z_REPRESENTATIVES) == 2 ** len(d5.Z_STABILIZERS)
        assert len(d5.LOGICAL_X_REPRESENTATIVES) == 2 ** len(d5.X_STABILIZERS)
        assert d5.DEFAULT_LOGICAL_Z in d5.LOGICAL_Z_REPRESENTATIVES
        assert d5.DEFAULT_LOGICAL_X in d5.LOGICAL_X_REPRESENTATIVES


class TestCanonicalStabilizerSchedule:
    def test_paper_top_and_bottom_boundary_rows(self) -> None:
        assert d5.X_STABILIZER_SCHEDULE["X_ab"] == ("", "", "a", "b")
        assert d5.X_STABILIZER_SCHEDULE["X_cd"] == ("", "", "c", "d")
        assert d5.X_STABILIZER_SCHEDULE["X_vw"] == ("v", "w", "", "")
        assert d5.X_STABILIZER_SCHEDULE["X_xy"] == ("x", "y", "", "")

    @pytest.mark.parametrize("selected_names", (Z_NAMES, X_NAMES, Z_NAMES + X_NAMES))
    def test_full_selections_have_four_two_qubit_layers(
        self,
        selected_names: tuple[str, ...],
    ) -> None:
        assert two_qubit_depth(_scheduled_round_circuit(selected_names)) == 4

    def test_merge_compatible_round_has_expected_interactions(self) -> None:
        circuit = _scheduled_round_circuit(Z_NAMES + MERGE_X_NAMES)
        expected_interactions = sum(map(len, d5.Z_STABILIZERS.values())) + sum(
            len(d5.X_STABILIZERS[name]) for name in MERGE_X_NAMES
        )
        assert {"X_ab", "X_cd"}.isdisjoint(MERGE_X_NAMES)
        assert two_qubit_depth(circuit) == 4
        assert (
            sum(
                len(instruction.targets_copy()) // 2
                for instruction in circuit
                if instruction.name == "CX"
            )
            == expected_interactions
        )

    def test_practical_subsets_have_at_most_four_two_qubit_layers(self) -> None:
        selections = (
            (),
            Z_NAMES,
            X_NAMES,
            Z_NAMES + X_NAMES,
            Z_NAMES + MERGE_X_NAMES,
            ("Z_abfg", "Z_mnrs"),
            ("X_fgkl", "X_rswx"),
            ("Z_ghlm", "X_nost"),
            ("Z_cdhi", "X_himn", "Z_qrvw"),
        )
        assert all(two_qubit_depth(_scheduled_round_circuit(names)) <= 4 for names in selections)


# ===========================================================================
# SET 2: Logical-zero preparation and Z checks
# ===========================================================================


class TestLogicalZeroPreparation:
    def test_z_stabilizers_and_logical_z_are_plus_one(self) -> None:
        circuit = stim.Circuit()
        d5.append_scheduled_stabilizer_round(
            circuit,
            surface_qubits=tuple(range(25)),
            selected_stabilizer_names=X_NAMES,
            ancillas=tuple(range(25, 37)),
            round_label="initialize",
            add_ticks=False,
        )
        d5.append_scheduled_stabilizer_round(
            circuit,
            surface_qubits=tuple(range(25)),
            selected_stabilizer_names=Z_NAMES,
            ancillas=tuple(range(37, 49)),
            round_label="check",
            add_ticks=False,
        )
        d5.append_logical_measurement(
            circuit,
            "Z",
            surface_qubits=tuple(range(25)),
            destructive=True,
        )
        shots = circuit.compile_sampler(seed=1).sample(32)
        assert not shots[:, 12:24].any()
        assert not (shots[:, -5:].sum(axis=1) % 2).any()


# ===========================================================================
# SET 3: Logical-plus preparation and X checks
# ===========================================================================


class TestLogicalPlusPreparation:
    def test_x_stabilizers_and_logical_x_are_plus_one(self) -> None:
        circuit = stim.Circuit()
        d5.prepare_product_plus(circuit, qubits=tuple(range(25)))
        d5.append_scheduled_stabilizer_round(
            circuit,
            surface_qubits=tuple(range(25)),
            selected_stabilizer_names=Z_NAMES,
            ancillas=tuple(range(25, 37)),
            round_label="initialize",
            add_ticks=False,
        )
        d5.append_scheduled_stabilizer_round(
            circuit,
            surface_qubits=tuple(range(25)),
            selected_stabilizer_names=X_NAMES,
            ancillas=tuple(range(37, 49)),
            round_label="check",
            add_ticks=False,
        )
        d5.append_logical_measurement(
            circuit,
            "X",
            surface_qubits=tuple(range(25)),
            destructive=True,
        )
        shots = circuit.compile_sampler(seed=1).sample(32)
        assert not shots[:, 12:24].any()
        assert not (shots[:, -5:].sum(axis=1) % 2).any()

    def test_known_stabilizers_and_logical_readout_are_annotated(self) -> None:
        circuit = stim.Circuit()
        d5.prepare_product_plus(circuit, qubits=tuple(range(NUM_DATA_QUBITS)))
        d5.append_scheduled_stabilizer_round(
            circuit,
            surface_qubits=tuple(range(NUM_DATA_QUBITS)),
            selected_stabilizer_names=Z_NAMES,
            ancillas=tuple(range(25, 37)),
            round_label="initialize",
            add_ticks=False,
        )
        final_x_positions = d5.append_scheduled_stabilizer_round(
            circuit,
            surface_qubits=tuple(range(NUM_DATA_QUBITS)),
            selected_stabilizer_names=X_NAMES,
            ancillas=tuple(range(37, 49)),
            round_label="final X checks",
            add_ticks=False,
        )
        rotated_helper.append_stabilizer_detectors(circuit, rounds=(final_x_positions,))
        d5.append_logical_measurement(
            circuit,
            "X",
            surface_qubits=tuple(range(NUM_DATA_QUBITS)),
            observable_index=0,
        )
        detection_events, observable_flips = circuit.compile_detector_sampler(seed=3).sample(
            32, separate_observables=True
        )

        assert circuit.num_detectors == len(X_NAMES)
        assert circuit.num_observables == 1
        assert not detection_events.any()
        assert not observable_flips.any()
