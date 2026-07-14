"""Tests for the rotated surface-code decoder and Pauli frame.

Fault injection, syndrome extraction, decoding, and frame-corrected logical
readout on rotated d=3 and d=5 patches.
"""

from __future__ import annotations

import pytest
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit_aer import AerSimulator

import research_qec_haoran.surface_code.rotated_d3_qiskit as d3
import research_qec_haoran.surface_code.rotated_d5_qiskit as d5
from research_qec_haoran.noise import append_pauli_faults
from research_qec_haoran.surface_code.decoding import PauliFrame, RotatedDecoder, StabilizerFrame
from research_qec_haoran.utils.normalize import parse_qiskit_registers

# ===========================================================================
# Shared constants and helpers (d=3)
# ===========================================================================

NUM_DATA_QUBITS = 9
NUM_Z_ANCILLAS = 4
NUM_X_ANCILLAS = 4
Z_NAMES = tuple(d3.Z_STABILIZERS)
X_NAMES = tuple(d3.X_STABILIZERS)



def _expected_syndrome_for_fault(pauli: str, qubit_label: str):
    """Compute the expected (z_deltas, x_deltas) for a single-qubit Pauli fault."""
    z_deltas = {name: 0 for name in Z_NAMES}
    x_deltas = {name: 0 for name in X_NAMES}
    if pauli in {"X", "Y"}:
        for name, support in d3.Z_STABILIZERS.items():
            z_deltas[name] = int(qubit_label in support)
    if pauli in {"Z", "Y"}:
        for name, support in d3.X_STABILIZERS.items():
            x_deltas[name] = int(qubit_label in support)
    return z_deltas, x_deltas


def _d3_decoder() -> RotatedDecoder:
    return RotatedDecoder(
        data_qubits=d3.SURFACE_ORDER,
        z_stabilizers=d3.Z_STABILIZERS,
        x_stabilizers=d3.X_STABILIZERS,
        logical_x=d3.DEFAULT_LOGICAL_X,
        logical_z=d3.DEFAULT_LOGICAL_Z,
    )


# ===========================================================================
# SET 1: Single-fault decoding (d=3)
# ===========================================================================


class TestD3SingleFaultDecoding:
    """
    Prepare |+_L⟩, set initial stabilizer frame, inject a single Pauli fault,
    re-measure all stabilizers, compute change-detected syndrome, decode, and
    verify the decoded correction matches the injected fault.
    """

    @staticmethod
    def _build_and_run(fault_pauli: str, fault_qubit: str):
        """Full pipeline: init → fault → detect → parse → decode."""
        sim = AerSimulator(method="stabilizer")

        data = QuantumRegister(NUM_DATA_QUBITS, "data")
        z_anc = QuantumRegister(NUM_Z_ANCILLAS, "z_anc")
        x_anc = QuantumRegister(NUM_X_ANCILLAS, "x_anc")
        init_z_cl = ClassicalRegister(NUM_Z_ANCILLAS, "init_z")
        det_z_cl = ClassicalRegister(NUM_Z_ANCILLAS, "det_z")
        det_x_cl = ClassicalRegister(NUM_X_ANCILLAS, "det_x")
        circuit = QuantumCircuit(data, z_anc, x_anc, init_z_cl, det_z_cl, det_x_cl)

        # Step 1: Prepare |+_L⟩ (all data in |+⟩, then Z-stabilizer round).
        d3.prepare_plus(circuit, qubits=list(data))
        d3.append_z_stabilizer_measurements(
            circuit,
            surface_qubits=list(data),
            z_ancillas=list(z_anc),
            classical_bits=list(init_z_cl),
            round_label="init",
            prepare_ancillas=False,
            add_barriers=False,
        )

        # Step 2: Inject the fault.
        append_pauli_faults(
            circuit,
            [(fault_pauli, fault_qubit)],
            qubit_map=d3.label2qubit(list(data)),
        )

        # Step 3: Full stabilizer round (detection).
        d3.append_stabilizer_round(
            circuit,
            surface_qubits=list(data),
            z_ancillas=list(z_anc),
            x_ancillas=list(x_anc),
            z_bits=list(det_z_cl),
            x_bits=list(det_x_cl),
            round_label="detect",
            prepare_z_ancillas=True,
            prepare_x_ancillas=True,
            add_barriers=False,
        )

        # Step 4: Run and parse.
        raw = next(iter(sim.run(circuit, shots=1).result().get_counts()))
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
        frame.record_many({name: 0 for name in X_NAMES})
        all_det = {**parsed["det_z"], **parsed["det_x"]}
        deltas = frame.compare_many(all_det, update=False)
        z_deltas = {name: deltas[name] for name in Z_NAMES}
        x_deltas = {name: deltas[name] for name in X_NAMES}

        # Step 6: Decode.
        return z_deltas, x_deltas, _d3_decoder().decode(z_deltas=z_deltas, x_deltas=x_deltas)

    @pytest.mark.parametrize("qubit_label", d3.SURFACE_ORDER)
    @pytest.mark.parametrize("pauli", ["X", "Y", "Z"])
    def test_syndrome_matches_expected(self, pauli: str, qubit_label: str) -> None:
        """Change-detected syndrome matches the theoretical prediction."""
        z_deltas, x_deltas, _ = self._build_and_run(pauli, qubit_label)
        expected_z, expected_x = _expected_syndrome_for_fault(pauli, qubit_label)
        assert z_deltas == expected_z
        assert x_deltas == expected_x

    @pytest.mark.parametrize("qubit_label", d3.SURFACE_ORDER)
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
        assert not recovery.is_trivial
        assert recovery.z_deltas == z_deltas
        assert recovery.x_deltas == x_deltas

    def test_no_error_gives_trivial_recovery(self) -> None:
        """Zero syndrome (no fault) decodes to identity."""
        recovery = _d3_decoder().decode(
            z_deltas={name: 0 for name in Z_NAMES},
            x_deltas={name: 0 for name in X_NAMES},
        )
        assert recovery.is_trivial

    @pytest.mark.parametrize("qubit_label", d3.SURFACE_ORDER)
    @pytest.mark.parametrize("pauli", ["X", "Y", "Z"])
    def test_logical_readout_correct_after_pauli_frame(self, pauli: str, qubit_label: str) -> None:
        """
        After decoding a single fault, the Pauli-frame-corrected destructive
        X_L measurement recovers the expected +1 eigenvalue of |+_L⟩.
        """
        sim = AerSimulator(method="stabilizer")

        data = QuantumRegister(NUM_DATA_QUBITS, "data")
        z_anc = QuantumRegister(NUM_Z_ANCILLAS, "z_anc")
        x_anc = QuantumRegister(NUM_X_ANCILLAS, "x_anc")
        init_z_cl = ClassicalRegister(NUM_Z_ANCILLAS, "init_z")
        det_z_cl = ClassicalRegister(NUM_Z_ANCILLAS, "det_z")
        det_x_cl = ClassicalRegister(NUM_X_ANCILLAS, "det_x")
        log_cl = ClassicalRegister(len(d3.DEFAULT_LOGICAL_X), "log_x")
        circuit = QuantumCircuit(data, z_anc, x_anc, init_z_cl, det_z_cl, det_x_cl, log_cl)

        # Step 1: Prepare |+_L⟩.
        d3.prepare_plus(circuit, qubits=list(data))
        d3.append_z_stabilizer_measurements(
            circuit,
            surface_qubits=list(data),
            z_ancillas=list(z_anc),
            classical_bits=list(init_z_cl),
            round_label="init",
            prepare_ancillas=False,
            add_barriers=False,
        )

        # Step 2: Inject the fault.
        append_pauli_faults(
            circuit,
            [(pauli, qubit_label)],
            qubit_map=d3.label2qubit(list(data)),
        )

        # Step 3: Full stabilizer round (detection).
        d3.append_stabilizer_round(
            circuit,
            surface_qubits=list(data),
            z_ancillas=list(z_anc),
            x_ancillas=list(x_anc),
            z_bits=list(det_z_cl),
            x_bits=list(det_x_cl),
            round_label="detect",
            prepare_z_ancillas=True,
            prepare_x_ancillas=True,
            add_barriers=False,
        )

        # Step 4: Destructive X_L measurement.
        d3.append_logical_measurement(
            circuit,
            "X",
            surface_qubits=list(data),
            classical_bits=list(log_cl),
            destructive=True,
        )

        # Step 5: Run and parse all registers.
        raw = next(iter(sim.run(circuit, shots=1).result().get_counts()))
        parsed = parse_qiskit_registers(
            raw,
            (
                ("init_z", Z_NAMES),
                ("det_z", Z_NAMES),
                ("det_x", X_NAMES),
                ("log_x", d3.DEFAULT_LOGICAL_X),
            ),
        )

        # Step 6: Compute change-detected syndrome.
        stab_frame = StabilizerFrame()
        stab_frame.record_many(parsed["init_z"])
        stab_frame.record_many({name: 0 for name in X_NAMES})
        deltas = stab_frame.compare_many(
            {**parsed["det_z"], **parsed["det_x"]},
            update=False,
        )
        z_deltas = {name: deltas[name] for name in Z_NAMES}
        x_deltas = {name: deltas[name] for name in X_NAMES}

        # Step 7: Decode and update Pauli frame.
        pauli_frame = PauliFrame()
        pauli_frame.register("L0")
        _d3_decoder().update_pauli_frame(
            pauli_frame,
            "L0",
            z_deltas=z_deltas,
            x_deltas=x_deltas,
        )

        # Step 8: Frame-corrected logical readout.
        raw_logical_x = sum(parsed["log_x"].values()) % 2
        assert pauli_frame.interpret_x_readout("L0", raw_logical_x) == 0


# ===========================================================================
# SET 2: Multi-round fault decoding (d=3)
# ===========================================================================


class TestD3MultiRoundFaultDecoding:
    """Prepare |+_L⟩, then repeat n rounds of {inject fault, stabilizer round,
    decode, update Pauli frame}. After all rounds, destructive X_L measurement
    conditioned on the Pauli frame should recover the expected +1 eigenvalue.
    """

    @staticmethod
    def _run_n_rounds(n_rounds: int, fault_pauli: str, fault_qubit: str) -> int:
        """Run the multi-round pipeline and return the frame-corrected X_L bit."""
        sim = AerSimulator(method="stabilizer")

        data = QuantumRegister(NUM_DATA_QUBITS, "data")
        z_anc = QuantumRegister(NUM_Z_ANCILLAS, "z_anc")
        x_anc = QuantumRegister(NUM_X_ANCILLAS, "x_anc")
        init_z_cl = ClassicalRegister(NUM_Z_ANCILLAS, "init_z")
        round_z_cls = [ClassicalRegister(NUM_Z_ANCILLAS, f"z_r{i}") for i in range(n_rounds)]
        round_x_cls = [ClassicalRegister(NUM_X_ANCILLAS, f"x_r{i}") for i in range(n_rounds)]
        log_cl = ClassicalRegister(len(d3.DEFAULT_LOGICAL_X), "log_x")
        circuit = QuantumCircuit(data, z_anc, x_anc, init_z_cl, *round_z_cls, *round_x_cls, log_cl)

        # Prepare |+_L⟩.
        d3.prepare_plus(circuit, qubits=list(data))
        d3.append_z_stabilizer_measurements(
            circuit,
            surface_qubits=list(data),
            z_ancillas=list(z_anc),
            classical_bits=list(init_z_cl),
            round_label="init",
            prepare_ancillas=False,
            add_barriers=False,
        )

        # n rounds of: fault → stabilizer measurement.
        for index in range(n_rounds):
            append_pauli_faults(
                circuit,
                [(fault_pauli, fault_qubit)],
                qubit_map=d3.label2qubit(list(data)),
            )
            d3.append_stabilizer_round(
                circuit,
                surface_qubits=list(data),
                z_ancillas=list(z_anc),
                x_ancillas=list(x_anc),
                z_bits=list(round_z_cls[index]),
                x_bits=list(round_x_cls[index]),
                round_label=f"r{index}",
                prepare_z_ancillas=True,
                prepare_x_ancillas=True,
                add_barriers=False,
            )

        # Destructive X_L measurement.
        d3.append_logical_measurement(
            circuit,
            "X",
            surface_qubits=list(data),
            classical_bits=list(log_cl),
            destructive=True,
        )

        # Run and parse all registers.
        raw = next(iter(sim.run(circuit, shots=1).result().get_counts()))
        reg_order = (
            [("init_z", Z_NAMES)]
            + [(f"z_r{i}", Z_NAMES) for i in range(n_rounds)]
            + [(f"x_r{i}", X_NAMES) for i in range(n_rounds)]
            + [("log_x", d3.DEFAULT_LOGICAL_X)]
        )
        parsed = parse_qiskit_registers(raw, reg_order)

        # Decode all rounds with StabilizerFrame + PauliFrame.
        stab_frame = StabilizerFrame()
        stab_frame.record_many(parsed["init_z"])
        stab_frame.record_many({name: 0 for name in X_NAMES})
        pauli_frame = PauliFrame()
        pauli_frame.register("L0")
        decoder = _d3_decoder()

        for index in range(n_rounds):
            deltas = stab_frame.compare_many(
                {**parsed[f"z_r{index}"], **parsed[f"x_r{index}"]},
                update=True,
            )
            decoder.update_pauli_frame(
                pauli_frame,
                "L0",
                z_deltas={name: deltas[name] for name in Z_NAMES},
                x_deltas={name: deltas[name] for name in X_NAMES},
            )

        # Frame-corrected logical readout.
        raw_logical_x = sum(parsed["log_x"].values()) % 2
        return pauli_frame.interpret_x_readout("L0", raw_logical_x)

    @pytest.mark.parametrize("n_rounds", [1, 3, 5])
    @pytest.mark.parametrize("qubit_label", d3.SURFACE_ORDER)
    @pytest.mark.parametrize("pauli", ["X", "Y", "Z"])
    def test_frame_corrected_readout_is_zero(
        self,
        pauli: str,
        qubit_label: str,
        n_rounds: int,
    ) -> None:
        assert self._run_n_rounds(n_rounds, pauli, qubit_label) == 0


# ===========================================================================
# Shared constants and helpers (d=5)
# ===========================================================================


D5_NUM_DATA_QUBITS = 25
D5_NUM_Z_ANCILLAS = 12
D5_NUM_X_ANCILLAS = 12
D5_Z_NAMES = tuple(d5.Z_STABILIZERS)
D5_X_NAMES = tuple(d5.X_STABILIZERS)


def _expected_d5_syndrome_for_fault(pauli: str, qubit_label: str):
    z_deltas = {name: 0 for name in D5_Z_NAMES}
    x_deltas = {name: 0 for name in D5_X_NAMES}
    if pauli in {"X", "Y"}:
        for name, support in d5.Z_STABILIZERS.items():
            z_deltas[name] = int(qubit_label in support)
    if pauli in {"Z", "Y"}:
        for name, support in d5.X_STABILIZERS.items():
            x_deltas[name] = int(qubit_label in support)
    return z_deltas, x_deltas


def _d5_decoder() -> RotatedDecoder:
    return RotatedDecoder(
        data_qubits=d5.SURFACE_ORDER,
        z_stabilizers=d5.Z_STABILIZERS,
        x_stabilizers=d5.X_STABILIZERS,
        logical_x=d5.DEFAULT_LOGICAL_X,
        logical_z=d5.DEFAULT_LOGICAL_Z,
    )


# ===========================================================================
# SET 3: Single-fault decoding (d=5)
# ===========================================================================

class TestD5SingleFaultDecoding:
    """Repeat the d=3 single-fault decoding checks on a rotated d=5 patch."""

    @staticmethod
    def _build_and_run(fault_pauli: str, fault_qubit: str):
        sim = AerSimulator(method="stabilizer")

        data = QuantumRegister(D5_NUM_DATA_QUBITS, "data")
        z_anc = QuantumRegister(D5_NUM_Z_ANCILLAS, "z_anc")
        x_anc = QuantumRegister(D5_NUM_X_ANCILLAS, "x_anc")
        init_z_cl = ClassicalRegister(D5_NUM_Z_ANCILLAS, "init_z")
        det_z_cl = ClassicalRegister(D5_NUM_Z_ANCILLAS, "det_z")
        det_x_cl = ClassicalRegister(D5_NUM_X_ANCILLAS, "det_x")
        circuit = QuantumCircuit(data, z_anc, x_anc, init_z_cl, det_z_cl, det_x_cl)

        d5.prepare_plus(circuit, qubits=list(data))
        d5.append_z_stabilizer_measurements(
            circuit,
            surface_qubits=list(data),
            z_ancillas=list(z_anc),
            classical_bits=list(init_z_cl),
            round_label="init",
            prepare_ancillas=False,
            add_barriers=False,
        )
        append_pauli_faults(
            circuit,
            [(fault_pauli, fault_qubit)],
            qubit_map=d5.label2qubit(list(data)),
        )
        d5.append_stabilizer_round(
            circuit,
            surface_qubits=list(data),
            z_ancillas=list(z_anc),
            x_ancillas=list(x_anc),
            z_bits=list(det_z_cl),
            x_bits=list(det_x_cl),
            round_label="detect",
            prepare_z_ancillas=True,
            prepare_x_ancillas=True,
            add_barriers=False,
        )

        raw = next(iter(sim.run(circuit, shots=1).result().get_counts()))
        parsed = parse_qiskit_registers(
            raw,
            (
                ("init_z", D5_Z_NAMES),
                ("det_z", D5_Z_NAMES),
                ("det_x", D5_X_NAMES),
            ),
        )

        frame = StabilizerFrame()
        frame.record_many(parsed["init_z"])
        frame.record_many({name: 0 for name in D5_X_NAMES})
        all_det = {**parsed["det_z"], **parsed["det_x"]}
        deltas = frame.compare_many(all_det, update=False)
        z_deltas = {name: deltas[name] for name in D5_Z_NAMES}
        x_deltas = {name: deltas[name] for name in D5_X_NAMES}
        return z_deltas, x_deltas, _d5_decoder().decode(z_deltas=z_deltas, x_deltas=x_deltas)

    @pytest.mark.parametrize("qubit_label", d5.SURFACE_ORDER)
    @pytest.mark.parametrize("pauli", ["X", "Y", "Z"])
    def test_syndrome_matches_expected(self, pauli: str, qubit_label: str) -> None:
        z_deltas, x_deltas, _ = self._build_and_run(pauli, qubit_label)
        expected_z, expected_x = _expected_d5_syndrome_for_fault(pauli, qubit_label)
        assert z_deltas == expected_z
        assert x_deltas == expected_x

    def test_no_error_gives_trivial_recovery(self) -> None:
        recovery = _d5_decoder().decode(
            z_deltas={name: 0 for name in D5_Z_NAMES},
            x_deltas={name: 0 for name in D5_X_NAMES},
        )
        assert recovery.is_trivial

    @pytest.mark.parametrize("qubit_label", d5.SURFACE_ORDER)
    @pytest.mark.parametrize("pauli", ["X", "Y", "Z"])
    def test_logical_readout_correct_after_pauli_frame(self, pauli: str, qubit_label: str) -> None:
        sim = AerSimulator(method="stabilizer")

        data = QuantumRegister(D5_NUM_DATA_QUBITS, "data")
        z_anc = QuantumRegister(D5_NUM_Z_ANCILLAS, "z_anc")
        x_anc = QuantumRegister(D5_NUM_X_ANCILLAS, "x_anc")
        init_z_cl = ClassicalRegister(D5_NUM_Z_ANCILLAS, "init_z")
        det_z_cl = ClassicalRegister(D5_NUM_Z_ANCILLAS, "det_z")
        det_x_cl = ClassicalRegister(D5_NUM_X_ANCILLAS, "det_x")
        log_cl = ClassicalRegister(len(d5.DEFAULT_LOGICAL_X), "log_x")
        circuit = QuantumCircuit(data, z_anc, x_anc, init_z_cl, det_z_cl, det_x_cl, log_cl)

        d5.prepare_plus(circuit, qubits=list(data))
        d5.append_z_stabilizer_measurements(
            circuit,
            surface_qubits=list(data),
            z_ancillas=list(z_anc),
            classical_bits=list(init_z_cl),
            round_label="init",
            prepare_ancillas=False,
            add_barriers=False,
        )
        append_pauli_faults(
            circuit,
            [(pauli, qubit_label)],
            qubit_map=d5.label2qubit(list(data)),
        )
        d5.append_stabilizer_round(
            circuit,
            surface_qubits=list(data),
            z_ancillas=list(z_anc),
            x_ancillas=list(x_anc),
            z_bits=list(det_z_cl),
            x_bits=list(det_x_cl),
            round_label="detect",
            prepare_z_ancillas=True,
            prepare_x_ancillas=True,
            add_barriers=False,
        )
        d5.append_logical_measurement(
            circuit,
            "X",
            surface_qubits=list(data),
            classical_bits=list(log_cl),
            destructive=True,
        )

        raw = next(iter(sim.run(circuit, shots=1).result().get_counts()))
        parsed = parse_qiskit_registers(
            raw,
            (
                ("init_z", D5_Z_NAMES),
                ("det_z", D5_Z_NAMES),
                ("det_x", D5_X_NAMES),
                ("log_x", d5.DEFAULT_LOGICAL_X),
            ),
        )

        stab_frame = StabilizerFrame()
        stab_frame.record_many(parsed["init_z"])
        stab_frame.record_many({name: 0 for name in D5_X_NAMES})
        deltas = stab_frame.compare_many(
            {**parsed["det_z"], **parsed["det_x"]},
            update=False,
        )
        z_deltas = {name: deltas[name] for name in D5_Z_NAMES}
        x_deltas = {name: deltas[name] for name in D5_X_NAMES}

        pauli_frame = PauliFrame()
        pauli_frame.register("L0")
        _d5_decoder().update_pauli_frame(
            pauli_frame,
            "L0",
            z_deltas=z_deltas,
            x_deltas=x_deltas,
        )

        raw_logical_x = sum(parsed["log_x"].values()) % 2
        assert pauli_frame.interpret_x_readout("L0", raw_logical_x) == 0


# ===========================================================================
# SET 4: Multi-round fault decoding (d=5)
# ===========================================================================


class TestD5MultiRoundFaultDecoding:
    """Run repeated d5 syndrome rounds and fold recoveries into a Pauli frame."""

    @staticmethod
    def _run_n_rounds(n_rounds: int, fault_pauli: str, fault_qubit: str) -> int:
        sim = AerSimulator(method="stabilizer")

        data = QuantumRegister(D5_NUM_DATA_QUBITS, "data")
        z_anc = QuantumRegister(D5_NUM_Z_ANCILLAS, "z_anc")
        x_anc = QuantumRegister(D5_NUM_X_ANCILLAS, "x_anc")
        init_z_cl = ClassicalRegister(D5_NUM_Z_ANCILLAS, "init_z")
        round_z_cls = [ClassicalRegister(D5_NUM_Z_ANCILLAS, f"z_r{i}") for i in range(n_rounds)]
        round_x_cls = [ClassicalRegister(D5_NUM_X_ANCILLAS, f"x_r{i}") for i in range(n_rounds)]
        log_cl = ClassicalRegister(len(d5.DEFAULT_LOGICAL_X), "log_x")
        circuit = QuantumCircuit(data, z_anc, x_anc, init_z_cl, *round_z_cls, *round_x_cls, log_cl)

        d5.prepare_plus(circuit, qubits=list(data))
        d5.append_z_stabilizer_measurements(
            circuit,
            surface_qubits=list(data),
            z_ancillas=list(z_anc),
            classical_bits=list(init_z_cl),
            round_label="init",
            prepare_ancillas=False,
            add_barriers=False,
        )

        for index in range(n_rounds):
            append_pauli_faults(
                circuit,
                [(fault_pauli, fault_qubit)],
                qubit_map=d5.label2qubit(list(data)),
            )
            d5.append_stabilizer_round(
                circuit,
                surface_qubits=list(data),
                z_ancillas=list(z_anc),
                x_ancillas=list(x_anc),
                z_bits=list(round_z_cls[index]),
                x_bits=list(round_x_cls[index]),
                round_label=f"r{index}",
                prepare_z_ancillas=True,
                prepare_x_ancillas=True,
                add_barriers=False,
            )

        d5.append_logical_measurement(
            circuit,
            "X",
            surface_qubits=list(data),
            classical_bits=list(log_cl),
            destructive=True,
        )

        raw = next(iter(sim.run(circuit, shots=1).result().get_counts()))
        reg_order = (
            [("init_z", D5_Z_NAMES)]
            + [(f"z_r{i}", D5_Z_NAMES) for i in range(n_rounds)]
            + [(f"x_r{i}", D5_X_NAMES) for i in range(n_rounds)]
            + [("log_x", d5.DEFAULT_LOGICAL_X)]
        )
        parsed = parse_qiskit_registers(raw, reg_order)

        stab_frame = StabilizerFrame()
        stab_frame.record_many(parsed["init_z"])
        stab_frame.record_many({name: 0 for name in D5_X_NAMES})
        pauli_frame = PauliFrame()
        pauli_frame.register("L0")
        decoder = _d5_decoder()

        for index in range(n_rounds):
            deltas = stab_frame.compare_many(
                {**parsed[f"z_r{index}"], **parsed[f"x_r{index}"]},
                update=True,
            )
            decoder.update_pauli_frame(
                pauli_frame,
                "L0",
                z_deltas={name: deltas[name] for name in D5_Z_NAMES},
                x_deltas={name: deltas[name] for name in D5_X_NAMES},
            )

        raw_logical_x = sum(parsed["log_x"].values()) % 2
        return pauli_frame.interpret_x_readout("L0", raw_logical_x)

    @pytest.mark.parametrize("n_rounds", [1, 3])
    @pytest.mark.parametrize("qubit_label", d5.SURFACE_ORDER)
    @pytest.mark.parametrize("pauli", ["X", "Y", "Z"])
    def test_frame_corrected_readout_is_zero(
        self,
        pauli: str,
        qubit_label: str,
        n_rounds: int,
    ) -> None:
        assert self._run_n_rounds(n_rounds, pauli, qubit_label) == 0
