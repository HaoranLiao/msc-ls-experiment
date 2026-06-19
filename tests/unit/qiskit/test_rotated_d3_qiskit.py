"""Tests for the rotated distance-3 surface-code patch (Qiskit workflow).

Organized into three test sets:
1. Patch geometry validation
2. Logical state preparation and stabilizer eigenvalue checks
3. Single-fault injection, syndrome extraction, and decoding
"""

from __future__ import annotations

import pytest
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer import AerSimulator

from research_qec_haoran.noise import append_pauli_faults
from research_qec_haoran.surface_code.decoding import PauliFrame, RotatedDecoder, StabilizerFrame
from research_qec_haoran.surface_code.rotated_d3_qiskit import (
    DEFAULT_LOGICAL_X,
    DEFAULT_LOGICAL_Z,
    LOGICAL_X_REPRESENTATIVES,
    LOGICAL_Z_REPRESENTATIVES,
    SURFACE_COORDS,
    SURFACE_INDEX,
    SURFACE_ORDER,
    X_STABILIZER_CENTERS,
    X_STABILIZERS,
    Z_STABILIZER_CENTERS,
    Z_STABILIZERS,
    append_logical_measurement,
    append_stabilizer_round,
    append_x_stabilizer_measurements,
    append_z_stabilizer_measurements,
    label2qubit,
    prepare_plus,
)
from research_qec_haoran.utils.normalize import parse_qiskit_registers

# ===========================================================================
# Shared constants
# ===========================================================================

NUM_DATA_QUBITS = 9
NUM_Z_ANCILLAS = 4
NUM_X_ANCILLAS = 4
Z_NAMES = tuple(Z_STABILIZERS)
X_NAMES = tuple(X_STABILIZERS)


# ===========================================================================
# Helpers
# ===========================================================================

def _chebyshev_distance(c1: tuple[float, float], c2: tuple[float, float]) -> float:
    return max(abs(c1[0] - c2[0]), abs(c1[1] - c2[1]))

def _build_logical_operator(basis: str, support: tuple[str, ...], n: int) -> SparsePauliOp:
    """Build an n-qubit Pauli operator acting on the given support."""
    chars = ["I"] * n
    for label in support:
        chars[SURFACE_INDEX[label]] = basis
    return SparsePauliOp("".join(reversed(chars)))


def _build_stabilizer_operator(basis: str, support: tuple[str, ...], n: int) -> SparsePauliOp:
    """Build an n-qubit stabilizer operator from its support."""
    return _build_logical_operator(basis, support, n)


def _expectation_from_counts(counts: dict[str, int], log_width: int) -> float:
    """
    Compute expectation value from measurement counts.

    Assumes the logical measurement bits are the leftmost ``log_width`` bits
    in each bitstring (i.e., the last classical register appended).
    The expectation is (+1)^(parity=0) weighted average: (N_even - N_odd) / N_total.
    """
    plus_count = 0
    minus_count = 0
    for bitstring, count in counts.items():
        raw = bitstring.replace(" ", "")
        log_bits = [int(b) for b in raw[:log_width]]
        parity = sum(log_bits) % 2
        if parity == 0:
            plus_count += count
        else:
            minus_count += count
    return (plus_count - minus_count) / (plus_count + minus_count)


def _expected_syndrome_for_fault(pauli: str, qubit_label: str):
    """Compute the expected (z_deltas, x_deltas) for a single-qubit Pauli fault."""
    z_deltas = {name: 0 for name in Z_NAMES}
    x_deltas = {name: 0 for name in X_NAMES}
    if pauli in {"X", "Y"}:
        for name, support in Z_STABILIZERS.items():
            z_deltas[name] = int(qubit_label in support)
    if pauli in {"Z", "Y"}:
        for name, support in X_STABILIZERS.items():
            x_deltas[name] = int(qubit_label in support)
    return z_deltas, x_deltas


# ===========================================================================
# SET 1: Patch geometry validation
# ===========================================================================


class TestPatchGeometry:
    """Verify the d=3 rotated patch has correct qubit counts and connectivity."""

    def test_data_qubit_count(self) -> None:
        assert len(SURFACE_ORDER) == NUM_DATA_QUBITS

    def test_stabilizer_counts(self) -> None:
        assert len(Z_STABILIZERS) == NUM_Z_ANCILLAS
        assert len(X_STABILIZERS) == NUM_X_ANCILLAS

    def test_all_data_qubits_have_coordinates(self) -> None:
        for label in SURFACE_ORDER:
            assert label in SURFACE_COORDS

    def test_stabilizer_centers_are_nearest_neighbors_to_support(self) -> None:
        """Each stabilizer ancilla center must be adjacent to all its data qubits."""
        for name, support in Z_STABILIZERS.items():
            center = Z_STABILIZER_CENTERS[name]
            for qubit_label in support:
                qubit_coord = SURFACE_COORDS[qubit_label]
                dist = _chebyshev_distance(center, qubit_coord)
                assert dist <= 1.0, (
                    f"Z-stabilizer {name} center {center} is too far "
                    f"from data qubit {qubit_label} at {qubit_coord} (dist={dist})"
                )

        for name, support in X_STABILIZERS.items():
            center = X_STABILIZER_CENTERS[name]
            for qubit_label in support:
                qubit_coord = SURFACE_COORDS[qubit_label]
                dist = _chebyshev_distance(center, qubit_coord)
                assert dist <= 1.0, (
                    f"X-stabilizer {name} center {center} is too far "
                    f"from data qubit {qubit_label} at {qubit_coord} (dist={dist})"
                )

    def test_logical_operators_have_correct_weight(self) -> None:
        """Logical operators for d=3 must have weight 3."""
        assert len(DEFAULT_LOGICAL_X) == 3
        assert len(DEFAULT_LOGICAL_Z) == 3

    def test_logical_operators_are_valid_data_qubits(self) -> None:
        for label in DEFAULT_LOGICAL_X:
            assert label in SURFACE_ORDER
        for label in DEFAULT_LOGICAL_Z:
            assert label in SURFACE_ORDER

    def test_logical_z_representatives_recovered_from_stabilizer_products(self) -> None:
        """Stabilizer products recover all logical-Z representatives."""
        from itertools import combinations
        z_stab_supports = [frozenset(s) for s in Z_STABILIZERS.values()]
        default = frozenset(DEFAULT_LOGICAL_Z)
        generated = set()
        for r in range(len(z_stab_supports) + 1):
            for combo in combinations(z_stab_supports, r):
                rep = default
                for stab in combo:
                    rep = rep.symmetric_difference(stab)
                generated.add(rep)
        expected = {frozenset(rep) for rep in LOGICAL_Z_REPRESENTATIVES}
        assert generated == expected

    def test_logical_x_representatives_recovered_from_stabilizer_products(self) -> None:
        """Stabilizer products recover all logical-X representatives."""
        from itertools import combinations
        x_stab_supports = [frozenset(s) for s in X_STABILIZERS.values()]
        default = frozenset(DEFAULT_LOGICAL_X)
        generated = set()
        for r in range(len(x_stab_supports) + 1):
            for combo in combinations(x_stab_supports, r):
                rep = default
                for stab in combo:
                    rep = rep.symmetric_difference(stab)
                generated.add(rep)
        expected = {frozenset(rep) for rep in LOGICAL_X_REPRESENTATIVES}
        assert generated == expected


# ===========================================================================
# SET 2: Logical state preparation and eigenvalue checks
# ===========================================================================


class TestLogicalZeroPreparation:
    """
    Prepare |0_L⟩ by initializing all data qubits to |0⟩ + X-stabilizer measurements.

    Expected properties:
      a) ⟨X_L⟩ = 0 (not an eigenstate of X_L)
      b) ⟨Z_L⟩ = +1 (eigenstate of Z_L with eigenvalue +1)
      c) All Z-stabilizers measure +1
    """

    def _prepare_logical_zero_statevector(self) -> Statevector:
        """Build |0_L⟩ circuit and return statevector (no measurements)."""
        s = QuantumRegister(NUM_DATA_QUBITS, "data")
        xa = QuantumRegister(NUM_X_ANCILLAS, "x_anc")
        x_cl = ClassicalRegister(NUM_X_ANCILLAS, "x_syn")
        qc = QuantumCircuit(s, xa, x_cl)

        append_x_stabilizer_measurements(
            qc, surface_qubits=list(s), x_ancillas=list(xa),
            classical_bits=list(x_cl), round_label="init",
            prepare_ancillas=False, add_barriers=False,
        )
        qc_no_meas = qc.remove_final_measurements(inplace=False)
        return Statevector.from_instruction(qc_no_meas)

    def _measure_logical_destructive(self, basis: str, support: tuple[str, ...]) -> float:
        """Prepare |0_L⟩, destructively measure logical, return expectation from counts."""
        sim = AerSimulator()
        s = QuantumRegister(NUM_DATA_QUBITS, "data")
        xa = QuantumRegister(NUM_X_ANCILLAS, "x_anc")
        x_cl = ClassicalRegister(NUM_X_ANCILLAS, "x_syn")
        log_cl = ClassicalRegister(len(support), "log")
        qc = QuantumCircuit(s, xa, x_cl, log_cl)

        append_x_stabilizer_measurements(
            qc, surface_qubits=list(s), x_ancillas=list(xa),
            classical_bits=list(x_cl), round_label="init",
            prepare_ancillas=False, add_barriers=False,
        )
        append_logical_measurement(
            qc, basis, surface_qubits=list(s), support=support,
            classical_bits=list(log_cl), destructive=True,
        )

        counts = sim.run(qc, shots=1024).result().get_counts()
        return _expectation_from_counts(counts, len(support))

    def _measure_logical_nondestructive(self, basis: str, support: tuple[str, ...]) -> float:
        """Prepare |0_L⟩, non-destructively measure logical, return ancilla Z expectation."""
        s = QuantumRegister(NUM_DATA_QUBITS, "data")
        xa = QuantumRegister(NUM_X_ANCILLAS, "x_anc")
        anc = QuantumRegister(1, "log_anc")
        x_cl = ClassicalRegister(NUM_X_ANCILLAS, "x_syn")
        log_cl = ClassicalRegister(1, "log")
        qc = QuantumCircuit(s, xa, anc, x_cl, log_cl)

        append_x_stabilizer_measurements(
            qc, surface_qubits=list(s), x_ancillas=list(xa),
            classical_bits=list(x_cl), round_label="init",
            prepare_ancillas=False, add_barriers=False,
        )
        append_logical_measurement(
            qc, basis, surface_qubits=list(s), support=support,
            ancilla_qubit=anc[0], classical_bits=list(log_cl), destructive=False,
        )

        qc_no_meas = qc.remove_final_measurements(inplace=False)
        sv = Statevector.from_instruction(qc_no_meas)
        anc_idx = NUM_DATA_QUBITS + NUM_X_ANCILLAS
        return sv.expectation_value(SparsePauliOp("Z"), [anc_idx]).real

    @pytest.mark.parametrize("support", LOGICAL_Z_REPRESENTATIVES)
    def test_logical_z_expectation_is_plus_one(self, support: tuple[str, ...]) -> None:
        sv = self._prepare_logical_zero_statevector()
        z_l = _build_logical_operator("Z", support, NUM_DATA_QUBITS + NUM_X_ANCILLAS)
        exp_val = sv.expectation_value(z_l).real
        assert exp_val == pytest.approx(1.0, abs=1e-10)

    @pytest.mark.parametrize("support", LOGICAL_X_REPRESENTATIVES)
    def test_logical_x_expectation_is_zero(self, support: tuple[str, ...]) -> None:
        sv = self._prepare_logical_zero_statevector()
        x_l = _build_logical_operator("X", support, NUM_DATA_QUBITS + NUM_X_ANCILLAS)
        exp_val = sv.expectation_value(x_l).real
        assert exp_val == pytest.approx(0.0, abs=1e-10)

    @pytest.mark.parametrize("support", LOGICAL_Z_REPRESENTATIVES)
    def test_destructive_z_measurement_is_plus_one(self, support: tuple[str, ...]) -> None:
        """Destructive Z_L measurement on |0_L⟩: expectation = +1 (eigenstate)."""
        assert self._measure_logical_destructive("Z", support) == pytest.approx(1.0, abs=1e-10)

    @pytest.mark.parametrize("support", LOGICAL_X_REPRESENTATIVES)
    def test_destructive_x_measurement_expectation_is_zero(self, support: tuple[str, ...]) -> None:
        """Destructive X_L measurement on |0_L⟩: expectation = 0 (not an eigenstate)."""
        assert self._measure_logical_destructive("X", support) == pytest.approx(0.0, abs=0.1)

    @pytest.mark.parametrize("support", LOGICAL_Z_REPRESENTATIVES)
    def test_nondestructive_z_measurement_is_plus_one(self, support: tuple[str, ...]) -> None:
        """Non-destructive Z_L on |0_L⟩: ancilla Z expectation = +1."""
        assert self._measure_logical_nondestructive("Z", support) == pytest.approx(1.0, abs=1e-10)

    @pytest.mark.parametrize("support", LOGICAL_X_REPRESENTATIVES)
    def test_nondestructive_x_measurement_expectation_is_zero(
        self,
        support: tuple[str, ...],
    ) -> None:
        """Non-destructive X_L on |0_L⟩: ancilla Z expectation = 0."""
        assert self._measure_logical_nondestructive("X", support) == pytest.approx(0.0, abs=1e-10)

    def test_all_z_stabilizers_are_plus_one(self) -> None:
        """Append Z-stabilizer measurement circuit; all Z ancillas have ⟨Z⟩ = +1."""
        s = QuantumRegister(NUM_DATA_QUBITS, "data")
        xa = QuantumRegister(NUM_X_ANCILLAS, "x_anc")
        za = QuantumRegister(NUM_Z_ANCILLAS, "z_anc")
        x_cl = ClassicalRegister(NUM_X_ANCILLAS, "x_syn")
        z_cl = ClassicalRegister(NUM_Z_ANCILLAS, "z_syn")
        qc = QuantumCircuit(s, xa, za, x_cl, z_cl)

        append_x_stabilizer_measurements(
            qc, surface_qubits=list(s), x_ancillas=list(xa),
            classical_bits=list(x_cl), round_label="init",
            prepare_ancillas=False, add_barriers=False,
        )
        append_z_stabilizer_measurements(
            qc, surface_qubits=list(s), z_ancillas=list(za),
            classical_bits=list(z_cl), round_label="check",
            prepare_ancillas=False, add_barriers=False,
        )

        qc_no_meas = qc.remove_final_measurements(inplace=False)
        sv = Statevector.from_instruction(qc_no_meas)
        z_op = SparsePauliOp("Z")
        for i, name in enumerate(Z_STABILIZERS):
            anc_idx = NUM_DATA_QUBITS + NUM_X_ANCILLAS + i
            exp_val = sv.expectation_value(z_op, [anc_idx]).real
            assert exp_val == pytest.approx(1.0, abs=1e-10), (
                f"Z-stabilizer {name} ancilla has ⟨Z⟩ = {exp_val}, expected +1"
            )


class TestLogicalPlusPreparation:
    """
    Prepare |+_L⟩ by initializing all data qubits to |+⟩ + Z-stabilizer measurements.

    Expected properties:
      a) ⟨Z_L⟩ = 0 (not an eigenstate of Z_L)
      b) ⟨X_L⟩ = +1 (eigenstate of X_L with eigenvalue +1)
      c) All X-stabilizers measure +1
    """

    def _prepare_logical_plus_statevector(self) -> Statevector:
        """Build |+_L⟩ circuit and return statevector (no measurements)."""
        s = QuantumRegister(NUM_DATA_QUBITS, "data")
        za = QuantumRegister(NUM_Z_ANCILLAS, "z_anc")
        z_cl = ClassicalRegister(NUM_Z_ANCILLAS, "z_syn")
        qc = QuantumCircuit(s, za, z_cl)

        prepare_plus(qc, qubits=list(s))
        append_z_stabilizer_measurements(
            qc, surface_qubits=list(s), z_ancillas=list(za),
            classical_bits=list(z_cl), round_label="init",
            prepare_ancillas=False, add_barriers=False,
        )
        qc_no_meas = qc.remove_final_measurements(inplace=False)
        return Statevector.from_instruction(qc_no_meas)

    def _measure_logical_destructive(self, basis: str, support: tuple[str, ...]) -> float:
        """Prepare |+_L⟩, destructively measure logical, return expectation from counts."""
        sim = AerSimulator()
        s = QuantumRegister(NUM_DATA_QUBITS, "data")
        za = QuantumRegister(NUM_Z_ANCILLAS, "z_anc")
        z_cl = ClassicalRegister(NUM_Z_ANCILLAS, "z_syn")
        log_cl = ClassicalRegister(len(support), "log")
        qc = QuantumCircuit(s, za, z_cl, log_cl)

        prepare_plus(qc, qubits=list(s))
        append_z_stabilizer_measurements(
            qc, surface_qubits=list(s), z_ancillas=list(za),
            classical_bits=list(z_cl), round_label="init",
            prepare_ancillas=False, add_barriers=False,
        )
        append_logical_measurement(
            qc, basis, surface_qubits=list(s), support=support,
            classical_bits=list(log_cl), destructive=True,
        )

        counts = sim.run(qc, shots=1024).result().get_counts()
        return _expectation_from_counts(counts, len(support))

    def _measure_logical_nondestructive(self, basis: str, support: tuple[str, ...]) -> float:
        """Prepare |+_L⟩, non-destructively measure logical, return ancilla Z expectation."""
        s = QuantumRegister(NUM_DATA_QUBITS, "data")
        za = QuantumRegister(NUM_Z_ANCILLAS, "z_anc")
        anc = QuantumRegister(1, "log_anc")
        z_cl = ClassicalRegister(NUM_Z_ANCILLAS, "z_syn")
        log_cl = ClassicalRegister(1, "log")
        qc = QuantumCircuit(s, za, anc, z_cl, log_cl)

        prepare_plus(qc, qubits=list(s))
        append_z_stabilizer_measurements(
            qc, surface_qubits=list(s), z_ancillas=list(za),
            classical_bits=list(z_cl), round_label="init",
            prepare_ancillas=False, add_barriers=False,
        )
        append_logical_measurement(
            qc, basis, surface_qubits=list(s), support=support,
            ancilla_qubit=anc[0], classical_bits=list(log_cl), destructive=False,
        )

        qc_no_meas = qc.remove_final_measurements(inplace=False)
        sv = Statevector.from_instruction(qc_no_meas)
        anc_idx = NUM_DATA_QUBITS + NUM_Z_ANCILLAS
        return sv.expectation_value(SparsePauliOp("Z"), [anc_idx]).real

    @pytest.mark.parametrize("support", LOGICAL_X_REPRESENTATIVES)
    def test_logical_x_expectation_is_plus_one(self, support: tuple[str, ...]) -> None:
        sv = self._prepare_logical_plus_statevector()
        x_l = _build_logical_operator("X", support, NUM_DATA_QUBITS + NUM_Z_ANCILLAS)
        exp_val = sv.expectation_value(x_l).real
        assert exp_val == pytest.approx(1.0, abs=1e-10)

    @pytest.mark.parametrize("support", LOGICAL_Z_REPRESENTATIVES)
    def test_logical_z_expectation_is_zero(self, support: tuple[str, ...]) -> None:
        sv = self._prepare_logical_plus_statevector()
        z_l = _build_logical_operator("Z", support, NUM_DATA_QUBITS + NUM_Z_ANCILLAS)
        exp_val = sv.expectation_value(z_l).real
        assert exp_val == pytest.approx(0.0, abs=1e-10)

    @pytest.mark.parametrize("support", LOGICAL_X_REPRESENTATIVES)
    def test_destructive_x_measurement_is_plus_one(self, support: tuple[str, ...]) -> None:
        """Destructive X_L measurement on |+_L⟩: expectation = +1 (eigenstate)."""
        assert self._measure_logical_destructive("X", support) == pytest.approx(1.0, abs=1e-10)

    @pytest.mark.parametrize("support", LOGICAL_Z_REPRESENTATIVES)
    def test_destructive_z_measurement_expectation_is_zero(self, support: tuple[str, ...]) -> None:
        """Destructive Z_L measurement on |+_L⟩: expectation = 0 (not an eigenstate)."""
        assert self._measure_logical_destructive("Z", support) == pytest.approx(0.0, abs=0.1)

    @pytest.mark.parametrize("support", LOGICAL_X_REPRESENTATIVES)
    def test_nondestructive_x_measurement_is_plus_one(self, support: tuple[str, ...]) -> None:
        """Non-destructive X_L on |+_L⟩: ancilla Z expectation = +1."""
        assert self._measure_logical_nondestructive("X", support) == pytest.approx(1.0, abs=1e-10)

    @pytest.mark.parametrize("support", LOGICAL_Z_REPRESENTATIVES)
    def test_nondestructive_z_measurement_expectation_is_zero(
        self,
        support: tuple[str, ...],
    ) -> None:
        """Non-destructive Z_L on |+_L⟩: ancilla Z expectation = 0."""
        assert self._measure_logical_nondestructive("Z", support) == pytest.approx(0.0, abs=1e-10)

    def test_all_x_stabilizers_are_plus_one(self) -> None:
        """Append X-stabilizer measurement circuit; all X ancillas have ⟨Z⟩ = +1."""
        s = QuantumRegister(NUM_DATA_QUBITS, "data")
        za = QuantumRegister(NUM_Z_ANCILLAS, "z_anc")
        xa = QuantumRegister(NUM_X_ANCILLAS, "x_anc")
        z_cl = ClassicalRegister(NUM_Z_ANCILLAS, "z_syn")
        x_cl = ClassicalRegister(NUM_X_ANCILLAS, "x_syn")
        qc = QuantumCircuit(s, za, xa, z_cl, x_cl)

        prepare_plus(qc, qubits=list(s))
        append_z_stabilizer_measurements(
            qc, surface_qubits=list(s), z_ancillas=list(za),
            classical_bits=list(z_cl), round_label="init",
            prepare_ancillas=False, add_barriers=False,
        )
        append_x_stabilizer_measurements(
            qc, surface_qubits=list(s), x_ancillas=list(xa),
            classical_bits=list(x_cl), round_label="check",
            prepare_ancillas=False, add_barriers=False,
        )

        qc_no_meas = qc.remove_final_measurements(inplace=False)
        sv = Statevector.from_instruction(qc_no_meas)
        z_op = SparsePauliOp("Z")
        for i, name in enumerate(X_STABILIZERS):
            anc_idx = NUM_DATA_QUBITS + NUM_Z_ANCILLAS + i
            exp_val = sv.expectation_value(z_op, [anc_idx]).real
            assert exp_val == pytest.approx(1.0, abs=1e-10), (
                f"X-stabilizer {name} ancilla has ⟨Z⟩ = {exp_val}, expected +1"
            )


# ===========================================================================
# SET 3: Fault injection, syndrome extraction, and decoding
# ===========================================================================


class TestSingleFaultDecoding:
    """
    Prepare |+_L⟩, set initial stabilizer frame, inject a single Pauli fault,
    re-measure all stabilizers, compute change-detected syndrome, decode, and
    verify the decoded correction matches the injected fault.
    """

    @staticmethod
    def _build_and_run(fault_pauli: str, fault_qubit: str):
        """Full pipeline: init → fault → detect → parse → decode."""
        sim = AerSimulator()

        s = QuantumRegister(NUM_DATA_QUBITS, "data")
        za = QuantumRegister(NUM_Z_ANCILLAS, "z_anc")
        xa = QuantumRegister(NUM_X_ANCILLAS, "x_anc")
        init_z_cl = ClassicalRegister(NUM_Z_ANCILLAS, "init_z")
        det_z_cl = ClassicalRegister(NUM_Z_ANCILLAS, "det_z")
        det_x_cl = ClassicalRegister(NUM_X_ANCILLAS, "det_x")
        qc = QuantumCircuit(s, za, xa, init_z_cl, det_z_cl, det_x_cl)

        # Step 1: Prepare |+_L⟩ (all data in |+⟩, then Z-stabilizer round).
        prepare_plus(qc, qubits=list(s))
        append_z_stabilizer_measurements(
            qc, surface_qubits=list(s), z_ancillas=list(za),
            classical_bits=list(init_z_cl), round_label="init",
            prepare_ancillas=False, add_barriers=False,
        )

        # Step 2: Inject the fault.
        append_pauli_faults(qc, [(fault_pauli, fault_qubit)], qubit_map=label2qubit(list(s)))

        # Step 3: Full stabilizer round (detection).
        append_stabilizer_round(
            qc, surface_qubits=list(s), z_ancillas=list(za), x_ancillas=list(xa),
            z_bits=list(det_z_cl), x_bits=list(det_x_cl),
            round_label="detect", prepare_z_ancillas=True, prepare_x_ancillas=True,
            add_barriers=False,
        )

        # Step 4: Run and parse.
        raw = next(iter(sim.run(qc, shots=1).result().get_counts()))
        parsed = parse_qiskit_registers(
            raw,
            (
                ("init_z", Z_NAMES),
                ("det_z", Z_NAMES),
                ("det_x", X_NAMES),
            ),
        )

        # Step 5: Compute change-detected syndrome via StabilizerFrame.
        frame = StabilizerFrame()
        frame.record_many(parsed["init_z"])
        frame.record_many({name: 0 for name in X_NAMES})  # X-stabilizer reference = 0

        all_det = {**parsed["det_z"], **parsed["det_x"]}
        deltas = frame.compare_many(all_det, update=False)
        z_deltas = {name: deltas[name] for name in Z_NAMES}
        x_deltas = {name: deltas[name] for name in X_NAMES}

        # Step 6: Decode.
        decoder = RotatedDecoder(
            data_qubits=SURFACE_ORDER,
            z_stabilizers=Z_STABILIZERS,
            x_stabilizers=X_STABILIZERS,
            logical_x=DEFAULT_LOGICAL_X,
            logical_z=DEFAULT_LOGICAL_Z,
        )
        recovery = decoder.decode(z_deltas=z_deltas, x_deltas=x_deltas)
        return z_deltas, x_deltas, recovery

    @pytest.mark.parametrize("qubit_label", SURFACE_ORDER)
    @pytest.mark.parametrize("pauli", ["X", "Y", "Z"])
    def test_syndrome_matches_expected(self, pauli: str, qubit_label: str) -> None:
        """Change-detected syndrome matches the theoretical prediction."""
        z_deltas, x_deltas, _ = self._build_and_run(pauli, qubit_label)
        expected_z, expected_x = _expected_syndrome_for_fault(pauli, qubit_label)
        assert z_deltas == expected_z, f"{pauli}_{qubit_label}: Z {z_deltas} != {expected_z}"
        assert x_deltas == expected_x, f"{pauli}_{qubit_label}: X {x_deltas} != {expected_x}"

    @pytest.mark.parametrize("qubit_label", SURFACE_ORDER)
    @pytest.mark.parametrize("pauli", ["X", "Y", "Z"])
    def test_decoded_recovery_is_nontrivial_and_consistent(
        self,
        pauli: str,
        qubit_label: str,
    ) -> None:
        """
        Decoded recovery is non-trivial and its syndrome matches the fault.

        The MWPM decoder independently corrects X-errors (via Z-syndrome) and
        Z-errors (via X-syndrome). For Y errors the X and Z components may land
        on different qubits in the same equivalence class; we verify the decoded
        syndrome matches the injected fault's syndrome in each sector.
        """
        z_deltas, x_deltas, recovery = self._build_and_run(pauli, qubit_label)
        assert not recovery.is_trivial, f"{pauli}_{qubit_label}: decoder returned trivial"

        # Verify the stored deltas in the recovery match what we measured.
        assert recovery.z_deltas == z_deltas
        assert recovery.x_deltas == x_deltas

    def test_no_error_gives_trivial_recovery(self) -> None:
        """Zero syndrome (no fault) decodes to identity."""
        decoder = RotatedDecoder(
            data_qubits=SURFACE_ORDER,
            z_stabilizers=Z_STABILIZERS,
            x_stabilizers=X_STABILIZERS,
            logical_x=DEFAULT_LOGICAL_X,
            logical_z=DEFAULT_LOGICAL_Z,
        )
        recovery = decoder.decode(
            z_deltas={name: 0 for name in Z_NAMES},
            x_deltas={name: 0 for name in X_NAMES},
        )
        assert recovery.is_trivial
        assert recovery.label() == "I"

    @pytest.mark.parametrize("qubit_label", SURFACE_ORDER)
    @pytest.mark.parametrize("pauli", ["X", "Y", "Z"])
    def test_logical_readout_correct_after_pauli_frame(self, pauli: str, qubit_label: str) -> None:
        """
        After decoding a single fault, the Pauli-frame-corrected destructive
        X_L measurement recovers the expected +1 eigenvalue of |+_L⟩.
        """
        sim = AerSimulator()

        s = QuantumRegister(NUM_DATA_QUBITS, "data")
        za = QuantumRegister(NUM_Z_ANCILLAS, "z_anc")
        xa = QuantumRegister(NUM_X_ANCILLAS, "x_anc")
        init_z_cl = ClassicalRegister(NUM_Z_ANCILLAS, "init_z")
        det_z_cl = ClassicalRegister(NUM_Z_ANCILLAS, "det_z")
        det_x_cl = ClassicalRegister(NUM_X_ANCILLAS, "det_x")
        log_cl = ClassicalRegister(len(DEFAULT_LOGICAL_X), "log_x")
        qc = QuantumCircuit(s, za, xa, init_z_cl, det_z_cl, det_x_cl, log_cl)

        # Step 1: Prepare |+_L⟩.
        prepare_plus(qc, qubits=list(s))
        append_z_stabilizer_measurements(
            qc, surface_qubits=list(s), z_ancillas=list(za),
            classical_bits=list(init_z_cl), round_label="init",
            prepare_ancillas=False, add_barriers=False,
        )

        # Step 2: Inject the fault.
        append_pauli_faults(qc, [(pauli, qubit_label)], qubit_map=label2qubit(list(s)))

        # Step 3: Full stabilizer round (detection).
        append_stabilizer_round(
            qc, surface_qubits=list(s), z_ancillas=list(za), x_ancillas=list(xa),
            z_bits=list(det_z_cl), x_bits=list(det_x_cl),
            round_label="detect", prepare_z_ancillas=True, prepare_x_ancillas=True,
            add_barriers=False,
        )

        # Step 4: Destructive X_L measurement.
        append_logical_measurement(
            qc, "X", surface_qubits=list(s),
            classical_bits=list(log_cl), destructive=True,
        )

        # Step 5: Run and parse all registers.
        raw = next(iter(sim.run(qc, shots=1).result().get_counts()))
        parsed = parse_qiskit_registers(
            raw,
            (
                ("init_z", Z_NAMES),
                ("det_z", Z_NAMES),
                ("det_x", X_NAMES),
                ("log_x", DEFAULT_LOGICAL_X),
            ),
        )

        # Parse syndrome registers.
        init_z = parsed["init_z"]
        det_z = parsed["det_z"]
        det_x = parsed["det_x"]

        # Step 6: Compute change-detected syndrome.
        stab_frame = StabilizerFrame()
        stab_frame.record_many(init_z)
        stab_frame.record_many({name: 0 for name in X_NAMES})
        all_det = {**det_z, **det_x}
        deltas = stab_frame.compare_many(all_det, update=False)
        z_deltas = {name: deltas[name] for name in Z_NAMES}
        x_deltas = {name: deltas[name] for name in X_NAMES}

        # Step 7: Decode and update Pauli frame.
        decoder = RotatedDecoder(
            data_qubits=SURFACE_ORDER,
            z_stabilizers=Z_STABILIZERS,
            x_stabilizers=X_STABILIZERS,
            logical_x=DEFAULT_LOGICAL_X,
            logical_z=DEFAULT_LOGICAL_Z,
        )
        pauli_frame = PauliFrame()
        pauli_frame.register("L0")
        decoder.update_pauli_frame(
            pauli_frame, "L0", z_deltas=z_deltas, x_deltas=x_deltas,
        )

        # Step 9: Compute raw logical X readout (parity of destructive measurement).
        raw_logical_x = sum(parsed["log_x"].values()) % 2

        # Step 10: Apply Pauli frame to get corrected readout.
        corrected = pauli_frame.interpret_x_readout("L0", raw_logical_x)
        assert corrected == 0, (
            f"{pauli}_{qubit_label}: frame-corrected X_L readout = {corrected}, "
            f"expected 0 (raw={raw_logical_x}, frame={pauli_frame.current_byproduct('L0')})"
        )


# ===========================================================================
# SET 4: Multi-round fault injection, decoding, and logical readout
# ===========================================================================


class TestMultiRoundFaultDecoding:
    """Prepare |+_L⟩, then repeat n rounds of {inject fault, stabilizer round,
    decode, update Pauli frame}. After all rounds, destructive X_L measurement
    conditioned on the Pauli frame should recover the expected +1 eigenvalue.
    """

    @staticmethod
    def _run_n_rounds(n_rounds: int, fault_pauli: str, fault_qubit: str) -> int:
        """Run the multi-round pipeline and return the frame-corrected X_L bit."""
        sim = AerSimulator()

        s = QuantumRegister(NUM_DATA_QUBITS, "data")
        za = QuantumRegister(NUM_Z_ANCILLAS, "z_anc")
        xa = QuantumRegister(NUM_X_ANCILLAS, "x_anc")

        # One init register + one pair per detection round + logical readout.
        init_z_cl = ClassicalRegister(NUM_Z_ANCILLAS, "init_z")
        round_z_cls = [ClassicalRegister(NUM_Z_ANCILLAS, f"z_r{i}") for i in range(n_rounds)]
        round_x_cls = [ClassicalRegister(NUM_X_ANCILLAS, f"x_r{i}") for i in range(n_rounds)]
        log_cl = ClassicalRegister(len(DEFAULT_LOGICAL_X), "log_x")

        qc = QuantumCircuit(s, za, xa, init_z_cl, *round_z_cls, *round_x_cls, log_cl)

        # Prepare |+_L⟩.
        prepare_plus(qc, qubits=list(s))
        append_z_stabilizer_measurements(
            qc, surface_qubits=list(s), z_ancillas=list(za),
            classical_bits=list(init_z_cl), round_label="init",
            prepare_ancillas=False, add_barriers=False,
        )

        # n rounds of: fault → stabilizer measurement.
        for i in range(n_rounds):
            append_pauli_faults(qc, [(fault_pauli, fault_qubit)], qubit_map=label2qubit(list(s)))
            append_stabilizer_round(
                qc, surface_qubits=list(s), z_ancillas=list(za), x_ancillas=list(xa),
                z_bits=list(round_z_cls[i]), x_bits=list(round_x_cls[i]),
                round_label=f"r{i}",
                prepare_z_ancillas=True, prepare_x_ancillas=True,
                add_barriers=False,
            )

        # Destructive X_L measurement.
        append_logical_measurement(
            qc, "X", surface_qubits=list(s),
            classical_bits=list(log_cl), destructive=True,
        )

        # Run.
        raw = next(iter(sim.run(qc, shots=1).result().get_counts()))

        # Parse all registers (right-to-left in Qiskit bitstring order).
        reg_order = (
            [("init_z", Z_NAMES)]
            + [(f"z_r{i}", Z_NAMES) for i in range(n_rounds)]
            + [(f"x_r{i}", X_NAMES) for i in range(n_rounds)]
            + [("log_x", DEFAULT_LOGICAL_X)]
        )
        parsed = parse_qiskit_registers(raw, reg_order)

        # Decode all rounds with StabilizerFrame + PauliFrame.
        stab_frame = StabilizerFrame()
        stab_frame.record_many(parsed["init_z"])
        stab_frame.record_many({name: 0 for name in X_NAMES})

        decoder = RotatedDecoder(
            data_qubits=SURFACE_ORDER,
            z_stabilizers=Z_STABILIZERS,
            x_stabilizers=X_STABILIZERS,
            logical_x=DEFAULT_LOGICAL_X,
            logical_z=DEFAULT_LOGICAL_Z,
        )
        pauli_frame = PauliFrame()
        pauli_frame.register("L0")

        for i in range(n_rounds):
            all_det = {**parsed[f"z_r{i}"], **parsed[f"x_r{i}"]}
            deltas = stab_frame.compare_many(all_det, update=True)
            z_deltas = {name: deltas[name] for name in Z_NAMES}
            x_deltas = {name: deltas[name] for name in X_NAMES}

            decoder.update_pauli_frame(pauli_frame, "L0", z_deltas=z_deltas, x_deltas=x_deltas)

        # Frame-corrected logical readout.
        raw_logical_x = sum(parsed["log_x"].values()) % 2
        return pauli_frame.interpret_x_readout("L0", raw_logical_x)

    @pytest.mark.parametrize("n_rounds", [1, 3, 5])
    @pytest.mark.parametrize("qubit_label", SURFACE_ORDER)
    @pytest.mark.parametrize("pauli", ["X", "Y", "Z"])
    def test_frame_corrected_readout_is_zero(
        self, pauli: str, qubit_label: str, n_rounds: int,
    ) -> None:
        """Frame-corrected X_L readout = 0 after n rounds of the same single fault."""
        corrected = self._run_n_rounds(n_rounds, pauli, qubit_label)
        assert corrected == 0, (
            f"{pauli}_{qubit_label} x{n_rounds}: frame-corrected X_L = {corrected}, expected 0"
        )

