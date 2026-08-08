"""Tests for the rotated distance-5 surface-code patch (Qiskit workflow)."""

from __future__ import annotations

import pytest
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit_aer import AerSimulator

import msc.rotated_surface_code.d5_qiskit as d5
from msc.utils.circuit_qiskit import two_qubit_depth
from msc.utils.normalize import parse_qiskit_registers

NUM_DATA_QUBITS = 25
NUM_Z_ANCILLAS = 12
NUM_X_ANCILLAS = 12
Z_NAMES = tuple(d5.Z_STABILIZERS)
X_NAMES = tuple(d5.X_STABILIZERS)
MERGE_X_NAMES = tuple(name for name in X_NAMES if name not in {"X_ab", "X_cd"})


# ===========================================================================
# Helpers
# ===========================================================================


def _chebyshev_distance(c1: tuple[float, float], c2: tuple[float, float]) -> float:
    return max(abs(c1[0] - c2[0]), abs(c1[1] - c2[1]))


def _one_shot(circuit: QuantumCircuit, register_labels):
    sim = AerSimulator(method="stabilizer")
    raw = next(iter(sim.run(circuit, shots=1).result().get_counts()))
    return parse_qiskit_registers(raw, register_labels)


# ===========================================================================
# SET 1: Patch geometry and canonical measurement schedule validation
# ===========================================================================


class TestPatchGeometry:
    """Verify the d=5 rotated patch has the expected geometry."""

    def test_data_qubit_count(self) -> None:
        assert len(d5.SURFACE_ORDER) == NUM_DATA_QUBITS
        assert d5.SURFACE_ORDER == tuple("abcdefghijklmnopqrstuvwxy")

    def test_stabilizer_counts(self) -> None:
        assert len(d5.Z_STABILIZERS) == NUM_Z_ANCILLAS
        assert len(d5.X_STABILIZERS) == NUM_X_ANCILLAS

    def test_paper_top_edge_labels_are_present(self) -> None:
        assert d5.DEFAULT_LOGICAL_Z == ("a", "b", "c", "d", "e")
        assert d5.DEFAULT_LOGICAL_X == ("a", "f", "k", "p", "u")
        assert d5.X_STABILIZERS["X_ab"] == ("a", "b")
        assert d5.X_STABILIZERS["X_cd"] == ("c", "d")

    def test_all_data_qubits_have_coordinates(self) -> None:
        for label in d5.SURFACE_ORDER:
            assert label in d5.SURFACE_COORDS

    def test_stabilizer_centers_are_nearest_neighbors_to_support(self) -> None:
        for name, support in d5.Z_STABILIZERS.items():
            center = d5.Z_STABILIZER_CENTERS[name]
            for qubit_label in support:
                dist = _chebyshev_distance(center, d5.SURFACE_COORDS[qubit_label])
                assert dist <= 1.0

        for name, support in d5.X_STABILIZERS.items():
            center = d5.X_STABILIZER_CENTERS[name]
            for qubit_label in support:
                dist = _chebyshev_distance(center, d5.SURFACE_COORDS[qubit_label])
                assert dist <= 1.0

    def test_logical_operators_have_correct_weight(self) -> None:
        assert len(d5.DEFAULT_LOGICAL_X) == 5
        assert len(d5.DEFAULT_LOGICAL_Z) == 5

    def test_logical_representatives_match_stabilizer_products(self) -> None:
        assert len(d5.LOGICAL_Z_REPRESENTATIVES) == 2 ** len(d5.Z_STABILIZERS)
        assert len(d5.LOGICAL_X_REPRESENTATIVES) == 2 ** len(d5.X_STABILIZERS)
        assert d5.DEFAULT_LOGICAL_Z in d5.LOGICAL_Z_REPRESENTATIVES
        assert d5.DEFAULT_LOGICAL_X in d5.LOGICAL_X_REPRESENTATIVES


class TestCanonicalStabilizerSchedule:
    """Verify the hook-safe four-step interaction schedule."""

    @staticmethod
    def _scheduled_round_circuit(
        selected_stabilizer_names: tuple[str, ...],
    ) -> QuantumCircuit:
        data = QuantumRegister(NUM_DATA_QUBITS, "data")
        circuit = QuantumCircuit(data)
        ancillas = ()
        bits = ()
        if selected_stabilizer_names:
            ancilla_register = QuantumRegister(
                len(selected_stabilizer_names),
                "ancillas",
            )
            syndrome_register = ClassicalRegister(
                len(selected_stabilizer_names),
                "syndrome",
            )
            circuit.add_register(ancilla_register, syndrome_register)
            ancillas = tuple(ancilla_register)
            bits = tuple(syndrome_register)

        d5.append_scheduled_stabilizer_round(
            circuit,
            surface_qubits=list(data),
            selected_stabilizer_names=selected_stabilizer_names,
            ancillas=ancillas,
            classical_bits=bits,
            round_label="scheduled",
            add_barriers=False,
        )
        return circuit

    def test_schedule_matches_canonical_corner_and_boundary_order(self) -> None:
        assert d5.X_STABILIZER_SCHEDULE == {
            "X_ab": ("", "", "a", "b"),
            "X_cd": ("", "", "c", "d"),
            "X_bcgh": ("b", "c", "g", "h"),
            "X_deij": ("d", "e", "i", "j"),
            "X_fgkl": ("f", "g", "k", "l"),
            "X_himn": ("h", "i", "m", "n"),
            "X_lmqr": ("l", "m", "q", "r"),
            "X_nost": ("n", "o", "s", "t"),
            "X_pquv": ("p", "q", "u", "v"),
            "X_rswx": ("r", "s", "w", "x"),
            "X_vw": ("v", "w", "", ""),
            "X_xy": ("x", "y", "", ""),
        }
        assert d5.Z_STABILIZER_SCHEDULE == {
            "Z_abfg": ("a", "f", "b", "g"),
            "Z_cdhi": ("c", "h", "d", "i"),
            "Z_ghlm": ("g", "l", "h", "m"),
            "Z_ijno": ("i", "n", "j", "o"),
            "Z_klpq": ("k", "p", "l", "q"),
            "Z_mnrs": ("m", "r", "n", "s"),
            "Z_qrvw": ("q", "v", "r", "w"),
            "Z_stxy": ("s", "x", "t", "y"),
            "Z_ej": ("e", "j", "", ""),
            "Z_ot": ("o", "t", "", ""),
            "Z_fk": ("", "", "f", "k"),
            "Z_pu": ("", "", "p", "u"),
        }

    def test_full_mixed_basis_schedule_has_no_data_qubit_collisions(self) -> None:
        """Each canonical step touches every data qubit at most once."""
        full_mixed_basis_schedule_rows = tuple(d5.Z_STABILIZER_SCHEDULE.values()) + tuple(
            d5.X_STABILIZER_SCHEDULE.values()
        )
        assert all(len(schedule) == 4 for schedule in full_mixed_basis_schedule_rows)
        for canonical_step in zip(
            *full_mixed_basis_schedule_rows,
            strict=True,
        ):
            active_data_qubit_labels = [label for label in canonical_step if label]
            assert len(active_data_qubit_labels) == len(set(active_data_qubit_labels))

    def test_all_z_preparation_checks_have_two_qubit_depth_four(self) -> None:
        assert two_qubit_depth(self._scheduled_round_circuit(Z_NAMES)) == 4

    def test_all_x_preparation_checks_have_two_qubit_depth_four(self) -> None:
        assert two_qubit_depth(self._scheduled_round_circuit(X_NAMES)) == 4

    def test_full_qec_round_has_two_qubit_depth_four(self) -> None:
        assert two_qubit_depth(self._scheduled_round_circuit(Z_NAMES + X_NAMES)) == 4

    def test_merge_compatible_round_omits_top_x_and_has_depth_four(self) -> None:
        circuit = self._scheduled_round_circuit(Z_NAMES + MERGE_X_NAMES)
        expected_cx_count = sum(map(len, d5.Z_STABILIZERS.values())) + sum(
            len(d5.X_STABILIZERS[name]) for name in MERGE_X_NAMES
        )

        assert {"X_ab", "X_cd"}.isdisjoint(MERGE_X_NAMES)
        assert circuit.count_ops()["cx"] == expected_cx_count
        assert two_qubit_depth(circuit) == 4

    def test_empty_stabilizer_subset_has_two_qubit_depth_zero(self) -> None:
        assert two_qubit_depth(self._scheduled_round_circuit(())) == 0

    def test_duplicate_stabilizer_names_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="selected_stabilizer_names must be unique"):
            self._scheduled_round_circuit(("X_ab", "X_ab"))

    def test_practical_stabilizer_selections_have_two_qubit_depth_at_most_four(
        self,
    ) -> None:
        """Exercise practical full, boundary-reduced, and local selections."""
        full_stabilizer_names = Z_NAMES + X_NAMES

        top_weight_two_x_names = ("X_ab", "X_cd")
        bottom_weight_two_x_names = ("X_vw", "X_xy")
        right_weight_two_z_names = ("Z_ej", "Z_ot")
        left_weight_two_z_names = ("Z_fk", "Z_pu")
        all_weight_two_names = (
            top_weight_two_x_names
            + right_weight_two_z_names
            + bottom_weight_two_x_names
            + left_weight_two_z_names
        )

        practical_stabilizer_selections = {
            # Empty and complete preparation/QEC selections.
            "empty": (),
            "all Z": Z_NAMES,
            "all X": X_NAMES,
            "all Z and X": full_stabilizer_names,
            # Full mixed-basis rounds with one boundary edge omitted.
            "without top weight-2 X checks": tuple(
                name for name in full_stabilizer_names if name not in top_weight_two_x_names
            ),
            "without right weight-2 Z checks": tuple(
                name for name in full_stabilizer_names if name not in right_weight_two_z_names
            ),
            "without bottom weight-2 X checks": tuple(
                name for name in full_stabilizer_names if name not in bottom_weight_two_x_names
            ),
            "without left weight-2 Z checks": tuple(
                name for name in full_stabilizer_names if name not in left_weight_two_z_names
            ),
            # Separate bulk-only and boundary-only rounds.
            "all weight-4 checks": tuple(
                name for name in full_stabilizer_names if name not in all_weight_two_names
            ),
            "all weight-2 checks": all_weight_two_names,
            # Fixed representative pairs and triples from one or both bases.
            "Z pair": ("Z_abfg", "Z_mnrs"),
            "X pair": ("X_fgkl", "X_rswx"),
            "mixed pair": ("Z_ghlm", "X_nost"),
            "Z triple": ("Z_cdhi", "Z_klpq", "Z_stxy"),
            "X triple": ("X_bcgh", "X_himn", "X_pquv"),
            "mixed triple": ("Z_abfg", "X_himn", "Z_qrvw"),
            # One X bulk row plus another Z bulk row and one boundary check.
            "X row, Z row, one weight-2 check": (
                "X_bcgh",
                "X_deij",
                "Z_klpq",
                "Z_mnrs",
                "X_ab",
            ),
            # One Z bulk row plus another X bulk row and no boundary checks.
            "Z row, X row, no weight-2 checks": (
                "Z_ghlm",
                "Z_ijno",
                "X_pquv",
                "X_rswx",
            ),
            # One X bulk column plus another Z bulk column and one boundary check.
            "X column, Z column, one weight-2 check": (
                "X_bcgh",
                "X_lmqr",
                "Z_cdhi",
                "Z_mnrs",
                "Z_ej",
            ),
            # One Z bulk column plus another X bulk column and no boundary checks.
            "Z column, X column, no weight-2 checks": (
                "Z_abfg",
                "Z_klpq",
                "X_deij",
                "X_nost",
            ),
        }

        for selection_name, selected_stabilizer_names in practical_stabilizer_selections.items():
            assert (
                two_qubit_depth(self._scheduled_round_circuit(selected_stabilizer_names)) <= 4
            ), selection_name


# ===========================================================================
# SET 2: Logical-zero preparation and Z checks
# ===========================================================================


class TestLogicalZeroPreparation:
    """Prepare |0_L> and verify deterministic Z data."""

    def test_z_stabilizers_and_logical_z_are_plus_one(self) -> None:
        data = QuantumRegister(NUM_DATA_QUBITS, "data")
        x_anc = QuantumRegister(NUM_X_ANCILLAS, "x_anc")
        z_anc = QuantumRegister(NUM_Z_ANCILLAS, "z_anc")
        x_cl = ClassicalRegister(NUM_X_ANCILLAS, "x_syn")
        z_cl = ClassicalRegister(NUM_Z_ANCILLAS, "z_syn")
        log_cl = ClassicalRegister(len(d5.DEFAULT_LOGICAL_Z), "log_z")
        circuit = QuantumCircuit(data, x_anc, z_anc, x_cl, z_cl, log_cl)

        d5.append_scheduled_stabilizer_round(
            circuit,
            surface_qubits=list(data),
            selected_stabilizer_names=X_NAMES,
            ancillas=list(x_anc),
            classical_bits=list(x_cl),
            round_label="init",
            reset_ancillas_initially=False,
            add_barriers=False,
        )
        d5.append_scheduled_stabilizer_round(
            circuit,
            surface_qubits=list(data),
            selected_stabilizer_names=Z_NAMES,
            ancillas=list(z_anc),
            classical_bits=list(z_cl),
            round_label="check",
            reset_ancillas_initially=False,
            add_barriers=False,
        )
        d5.append_logical_measurement(
            circuit,
            "Z",
            surface_qubits=list(data),
            classical_bits=list(log_cl),
            destructive=True,
        )

        parsed = _one_shot(
            circuit,
            (("x_syn", X_NAMES), ("z_syn", Z_NAMES), ("log_z", d5.DEFAULT_LOGICAL_Z)),
        )
        assert all(bit == 0 for bit in parsed["z_syn"].values())
        assert sum(parsed["log_z"].values()) % 2 == 0


# ===========================================================================
# SET 3: Logical-plus preparation and X checks
# ===========================================================================


class TestLogicalPlusPreparation:
    """Prepare |+_L> and verify deterministic X data."""

    def test_x_stabilizers_and_logical_x_are_plus_one(self) -> None:
        data = QuantumRegister(NUM_DATA_QUBITS, "data")
        z_anc = QuantumRegister(NUM_Z_ANCILLAS, "z_anc")
        x_anc = QuantumRegister(NUM_X_ANCILLAS, "x_anc")
        z_cl = ClassicalRegister(NUM_Z_ANCILLAS, "z_syn")
        x_cl = ClassicalRegister(NUM_X_ANCILLAS, "x_syn")
        log_cl = ClassicalRegister(len(d5.DEFAULT_LOGICAL_X), "log_x")
        circuit = QuantumCircuit(data, z_anc, x_anc, z_cl, x_cl, log_cl)

        d5.prepare_product_plus(circuit, qubits=list(data))
        d5.append_scheduled_stabilizer_round(
            circuit,
            surface_qubits=list(data),
            selected_stabilizer_names=Z_NAMES,
            ancillas=list(z_anc),
            classical_bits=list(z_cl),
            round_label="init",
            reset_ancillas_initially=False,
            add_barriers=False,
        )
        d5.append_scheduled_stabilizer_round(
            circuit,
            surface_qubits=list(data),
            selected_stabilizer_names=X_NAMES,
            ancillas=list(x_anc),
            classical_bits=list(x_cl),
            round_label="check",
            reset_ancillas_initially=False,
            add_barriers=False,
        )
        d5.append_logical_measurement(
            circuit,
            "X",
            surface_qubits=list(data),
            classical_bits=list(log_cl),
            destructive=True,
        )

        parsed = _one_shot(
            circuit,
            (("z_syn", Z_NAMES), ("x_syn", X_NAMES), ("log_x", d5.DEFAULT_LOGICAL_X)),
        )
        assert all(bit == 0 for bit in parsed["x_syn"].values())
        assert sum(parsed["log_x"].values()) % 2 == 0
