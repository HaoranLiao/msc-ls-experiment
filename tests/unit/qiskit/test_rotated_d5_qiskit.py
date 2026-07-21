"""Tests for the rotated distance-5 surface-code patch (Qiskit workflow)."""

from __future__ import annotations

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit_aer import AerSimulator

import magic_state_cultivation.surface_code.rotated_d5_qiskit as d5
from magic_state_cultivation.utils.normalize import parse_qiskit_registers

NUM_DATA_QUBITS = 25
NUM_Z_ANCILLAS = 12
NUM_X_ANCILLAS = 12
Z_NAMES = tuple(d5.Z_STABILIZERS)
X_NAMES = tuple(d5.X_STABILIZERS)


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
# SET 1: Patch geometry and logical operators
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

        d5.append_x_stabilizer_measurements(
            circuit,
            surface_qubits=list(data),
            x_ancillas=list(x_anc),
            classical_bits=list(x_cl),
            round_label="init",
            prepare_ancillas=False,
            add_barriers=False,
        )
        d5.append_z_stabilizer_measurements(
            circuit,
            surface_qubits=list(data),
            z_ancillas=list(z_anc),
            classical_bits=list(z_cl),
            round_label="check",
            prepare_ancillas=False,
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
        d5.append_z_stabilizer_measurements(
            circuit,
            surface_qubits=list(data),
            z_ancillas=list(z_anc),
            classical_bits=list(z_cl),
            round_label="init",
            prepare_ancillas=False,
            add_barriers=False,
        )
        d5.append_x_stabilizer_measurements(
            circuit,
            surface_qubits=list(data),
            x_ancillas=list(x_anc),
            classical_bits=list(x_cl),
            round_label="check",
            prepare_ancillas=False,
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
