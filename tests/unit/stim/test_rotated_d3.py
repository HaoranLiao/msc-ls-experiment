"""Tests for the rotated distance-3 surface-code patch in Stim."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest
import stim

from msc.rotated_surface_code import d3_stim as d3
from msc.rotated_surface_code import helper_stim as helper
from msc.utils.circuit_stim import two_qubit_depth

# ===========================================================================
# Shared constants
# ===========================================================================

NUM_DATA_QUBITS = 9
NUM_Z_ANCILLAS = 4
NUM_X_ANCILLAS = 4
Z_NAMES = tuple(d3.Z_STABILIZERS)
X_NAMES = tuple(d3.X_STABILIZERS)
MERGE_X_NAMES = tuple(name for name in X_NAMES if name != "X_ab")


# ===========================================================================
# Helpers
# ===========================================================================


def _pauli_string(basis: str, support: tuple[str, ...], *, num_qubits: int) -> stim.PauliString:
    chars = ["_"] * num_qubits
    for label in support:
        chars[d3.SURFACE_INDEX[label]] = basis
    return stim.PauliString("".join(chars))


def _peek(circuit: stim.Circuit, observable: stim.PauliString) -> int:
    simulator = stim.TableauSimulator()
    simulator.do_circuit(circuit)
    return simulator.peek_observable_expectation(observable)


def _scheduled_round_circuit(selected_stabilizer_names: tuple[str, ...]) -> stim.Circuit:
    circuit = stim.Circuit()
    d3.append_scheduled_stabilizer_round(
        circuit,
        surface_qubits=tuple(range(NUM_DATA_QUBITS)),
        selected_stabilizer_names=selected_stabilizer_names,
        ancillas=tuple(range(NUM_DATA_QUBITS, NUM_DATA_QUBITS + len(selected_stabilizer_names))),
        round_label="scheduled",
        add_ticks=True,
    )
    return circuit


def _logical_zero_circuit() -> stim.Circuit:
    circuit = stim.Circuit()
    d3.append_scheduled_stabilizer_round(
        circuit,
        surface_qubits=tuple(range(NUM_DATA_QUBITS)),
        selected_stabilizer_names=X_NAMES,
        ancillas=tuple(range(9, 13)),
        round_label="initialize |0_L>",
        add_ticks=False,
    )
    return circuit


def _logical_plus_circuit() -> stim.Circuit:
    circuit = stim.Circuit()
    d3.prepare_product_plus(circuit, qubits=tuple(range(NUM_DATA_QUBITS)))
    d3.append_scheduled_stabilizer_round(
        circuit,
        surface_qubits=tuple(range(NUM_DATA_QUBITS)),
        selected_stabilizer_names=Z_NAMES,
        ancillas=tuple(range(9, 13)),
        round_label="initialize |+_L>",
        add_ticks=False,
    )
    return circuit


# ===========================================================================
# SET 1: Patch geometry and canonical measurement schedule validation
# ===========================================================================


class TestPatchGeometry:
    def test_data_and_stabilizer_counts(self) -> None:
        assert len(d3.SURFACE_ORDER) == NUM_DATA_QUBITS
        assert len(d3.Z_STABILIZERS) == NUM_Z_ANCILLAS
        assert len(d3.X_STABILIZERS) == NUM_X_ANCILLAS

    def test_logical_operators_have_distance_three_weight(self) -> None:
        assert len(d3.DEFAULT_LOGICAL_X) == 3
        assert len(d3.DEFAULT_LOGICAL_Z) == 3
        assert set(d3.DEFAULT_LOGICAL_X) <= set(d3.SURFACE_ORDER)
        assert set(d3.DEFAULT_LOGICAL_Z) <= set(d3.SURFACE_ORDER)

    @pytest.mark.parametrize(
        ("default", "stabilizers", "representatives"),
        (
            (d3.DEFAULT_LOGICAL_Z, d3.Z_STABILIZERS, d3.LOGICAL_Z_REPRESENTATIVES),
            (d3.DEFAULT_LOGICAL_X, d3.X_STABILIZERS, d3.LOGICAL_X_REPRESENTATIVES),
        ),
    )
    def test_representatives_are_stabilizer_products(
        self,
        default: tuple[str, ...],
        stabilizers: dict[str, tuple[str, ...]],
        representatives: tuple[tuple[str, ...], ...],
    ) -> None:
        stabilizer_supports = tuple(frozenset(support) for support in stabilizers.values())
        generated: set[frozenset[str]] = set()
        for count in range(len(stabilizer_supports) + 1):
            for selected_supports in combinations(stabilizer_supports, count):
                representative = frozenset(default)
                for support in selected_supports:
                    representative = representative.symmetric_difference(support)
                generated.add(representative)
        assert generated == {frozenset(support) for support in representatives}


class TestCanonicalStabilizerSchedule:
    def test_schedule_matches_canonical_order(self) -> None:
        assert d3.X_STABILIZER_SCHEDULE == {
            "X_ab": ("", "", "a", "b"),
            "X_bcef": ("b", "c", "e", "f"),
            "X_degh": ("d", "e", "g", "h"),
            "X_hi": ("h", "i", "", ""),
        }
        assert d3.Z_STABILIZER_SCHEDULE == {
            "Z_abde": ("a", "d", "b", "e"),
            "Z_cf": ("c", "f", "", ""),
            "Z_dg": ("", "", "d", "g"),
            "Z_efhi": ("e", "h", "f", "i"),
        }

    @pytest.mark.parametrize("selected_names", (Z_NAMES, X_NAMES, Z_NAMES + X_NAMES))
    def test_full_round_selections_have_four_two_qubit_layers(
        self,
        selected_names: tuple[str, ...],
    ) -> None:
        assert two_qubit_depth(_scheduled_round_circuit(selected_names)) == 4

    def test_merge_round_omits_top_x_and_preserves_interaction_count(self) -> None:
        circuit = _scheduled_round_circuit(Z_NAMES + MERGE_X_NAMES)
        expected_interactions = sum(map(len, d3.Z_STABILIZERS.values())) + sum(
            len(d3.X_STABILIZERS[name]) for name in MERGE_X_NAMES
        )
        actual_interactions = sum(
            len(instruction.targets_copy()) // 2
            for instruction in circuit
            if instruction.name == "CX"
        )
        assert actual_interactions == expected_interactions
        assert two_qubit_depth(circuit) == 4

    @pytest.mark.slow
    def test_every_subset_has_at_most_four_two_qubit_layers(self) -> None:
        names = Z_NAMES + X_NAMES
        for count in range(len(names) + 1):
            for selected_names in combinations(names, count):
                assert two_qubit_depth(_scheduled_round_circuit(selected_names)) <= 4

    def test_duplicate_stabilizers_and_ancillas_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be unique"):
            _scheduled_round_circuit(("X_ab", "X_ab"))
        with pytest.raises(ValueError, match="unique ancilla"):
            d3.append_scheduled_stabilizer_round(
                stim.Circuit(),
                surface_qubits=tuple(range(NUM_DATA_QUBITS)),
                selected_stabilizer_names=("X_ab", "Z_cf"),
                ancillas=(9, 9),
                round_label="invalid",
                add_ticks=False,
            )


# ===========================================================================
# SET 2: Logical zero preparation and eigenvalue checks
# ===========================================================================


class TestLogicalZeroPreparation:
    @pytest.mark.parametrize("support", d3.LOGICAL_Z_REPRESENTATIVES)
    def test_logical_z_expectation_is_plus_one(self, support: tuple[str, ...]) -> None:
        assert (
            _peek(
                _logical_zero_circuit(),
                _pauli_string("Z", support, num_qubits=13),
            )
            == 1
        )

    @pytest.mark.parametrize("support", d3.LOGICAL_X_REPRESENTATIVES)
    def test_logical_x_expectation_is_zero(self, support: tuple[str, ...]) -> None:
        assert (
            _peek(
                _logical_zero_circuit(),
                _pauli_string("X", support, num_qubits=13),
            )
            == 0
        )

    def test_destructive_logical_z_and_z_checks_are_plus_one(self) -> None:
        circuit = _logical_zero_circuit()
        d3.append_scheduled_stabilizer_round(
            circuit,
            surface_qubits=tuple(range(9)),
            selected_stabilizer_names=Z_NAMES,
            ancillas=tuple(range(13, 17)),
            round_label="check",
            add_ticks=False,
        )
        d3.append_logical_measurement(
            circuit,
            "Z",
            surface_qubits=tuple(range(9)),
            destructive=True,
        )
        shots = circuit.compile_sampler(seed=1).sample(32)
        assert not shots[:, 4:8].any()
        assert not (shots[:, -3:].sum(axis=1) % 2).any()


# ===========================================================================
# SET 3: Logical-plus preparation and eigenvalue checks
# ===========================================================================


class TestLogicalPlusPreparation:
    @pytest.mark.parametrize("support", d3.LOGICAL_X_REPRESENTATIVES)
    def test_logical_x_expectation_is_plus_one(self, support: tuple[str, ...]) -> None:
        assert (
            _peek(
                _logical_plus_circuit(),
                _pauli_string("X", support, num_qubits=13),
            )
            == 1
        )

    @pytest.mark.parametrize("support", d3.LOGICAL_Z_REPRESENTATIVES)
    def test_logical_z_expectation_is_zero(self, support: tuple[str, ...]) -> None:
        assert (
            _peek(
                _logical_plus_circuit(),
                _pauli_string("Z", support, num_qubits=13),
            )
            == 0
        )

    def test_destructive_logical_x_and_x_checks_are_plus_one(self) -> None:
        circuit = _logical_plus_circuit()
        d3.append_scheduled_stabilizer_round(
            circuit,
            surface_qubits=tuple(range(9)),
            selected_stabilizer_names=X_NAMES,
            ancillas=tuple(range(13, 17)),
            round_label="check",
            add_ticks=False,
        )
        d3.append_logical_measurement(
            circuit,
            "X",
            surface_qubits=tuple(range(9)),
            destructive=True,
        )
        shots = circuit.compile_sampler(seed=1).sample(32)
        assert not shots[:, 4:8].any()
        assert not (shots[:, -3:].sum(axis=1) % 2).any()

    def test_surface_logical_y_uses_union_of_distinct_x_and_z_supports(self) -> None:
        circuit = _logical_plus_circuit()
        d3.append_logical_measurement(
            circuit,
            "Y",
            surface_qubits=tuple(range(9)),
            destructive=True,
        )
        assert circuit.num_measurements == NUM_Z_ANCILLAS + 5
        parities = circuit.compile_sampler(seed=2).sample(512)[:, -5:].sum(axis=1) % 2
        assert np.mean(parities) == pytest.approx(0.5, abs=0.08)

    def test_known_stabilizers_and_logical_readout_are_annotated(self) -> None:
        circuit = _logical_plus_circuit()
        final_x_positions = d3.append_scheduled_stabilizer_round(
            circuit,
            surface_qubits=tuple(range(NUM_DATA_QUBITS)),
            selected_stabilizer_names=X_NAMES,
            ancillas=tuple(range(13, 17)),
            round_label="final X checks",
            add_ticks=False,
        )
        helper.append_stabilizer_detectors(circuit, rounds=(final_x_positions,))
        d3.append_logical_measurement(
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
