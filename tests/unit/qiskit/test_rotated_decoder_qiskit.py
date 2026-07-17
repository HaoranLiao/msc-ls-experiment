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
from research_qec_haoran.surface_code.decoding import (
    PauliFrame,
    RotatedDecoder,
    StabilizerFrame,
    TemporalRotatedDecoder,
)
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

        # Step 6: Initialize the stabilizer reference.
        stab_frame = StabilizerFrame()
        stab_frame.record_many(parsed["init_z"])
        stab_frame.record_many({name: 0 for name in X_NAMES})

        # Step 7: Change-detect, decode, and update the Pauli frame.
        pauli_frame = PauliFrame()
        pauli_frame.register("L0")
        _d3_decoder().decode_from_raw_syndromes(
            pauli_frame=pauli_frame,
            logical_qubit_label="L0",
            z_raw=parsed["det_z"],
            x_raw=parsed["det_x"],
            stab_frame=stab_frame,
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
            decoder.decode_from_raw_syndromes(
                pauli_frame=pauli_frame,
                logical_qubit_label="L0",
                z_raw=parsed[f"z_r{index}"],
                x_raw=parsed[f"x_r{index}"],
                stab_frame=stab_frame,
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

        pauli_frame = PauliFrame()
        pauli_frame.register("L0")
        _d5_decoder().decode_from_raw_syndromes(
            pauli_frame=pauli_frame,
            logical_qubit_label="L0",
            z_raw=parsed["det_z"],
            x_raw=parsed["det_x"],
            stab_frame=stab_frame,
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
            decoder.decode_from_raw_syndromes(
                pauli_frame=pauli_frame,
                logical_qubit_label="L0",
                z_raw=parsed[f"z_r{index}"],
                x_raw=parsed[f"x_r{index}"],
                stab_frame=stab_frame,
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


# ===========================================================================
# SET 5: Joint temporal decoding across repeated syndrome rounds
# ===========================================================================


@pytest.mark.parametrize(
    ("code", "withheld_x"),
    [(d3, {"X_ab"}), (d5, {"X_ab", "X_cd"})],
    ids=("d3", "d5"),
)
class TestJointTemporalDecoding:
    """Exercise MWPM on detector events from all transfer rounds at once.

    A detector event is a *change* in a stabilizer outcome, rather than the raw
    outcome itself. Repeating the checks adds a time direction to the ordinary
    surface-code matching graph:

    * a spatial edge represents a Pauli error on a data qubit;
    * a timelike edge represents an incorrect ancilla measurement whose effect
      appears in two adjacent change-detection layers.

    During lattice surgery, ``X_ab`` (and ``X_cd`` for d=5) is temporarily
    replaced by merged-boundary checks, so only the remaining compatible X
    stabilizers are supplied to this decoder.

    The two CSS sectors intentionally have different history lengths. The
    surface patch starts in |+_L>, giving compatible X checks a known +1
    reference before round 1 and therefore one detector layer per each of the
    three rounds. Initial Z-check outcomes are not predetermined, so round 1
    establishes their reference and only the round-2 and round-3 changes are
    detectors. Thus these tests pass three X-check layers but two Z-check
    layers.
    """

    @staticmethod
    def _decoder(code, withheld_x: set[str]):
        """Build the decoder from checks that remain active during the merge."""
        measured_x = {
            name: support for name, support in code.X_STABILIZERS.items() if name not in withheld_x
        }
        return (
            TemporalRotatedDecoder(
                data_qubits=code.SURFACE_ORDER,
                z_stabilizers=code.Z_STABILIZERS,
                x_stabilizers=measured_x,
            ),
            measured_x,
        )

    def test_persistent_data_faults_decode_to_net_correction(self, code, withheld_x) -> None:
        """A lasting data fault creates one event and requires a data recovery.

        Although the raw stabilizer outcome remains flipped in later rounds,
        its *change* history contains an event only when the fault first
        appears. MWPM therefore connects that event through a spatial data
        edge, returning the corresponding physical-qubit correction.
        """
        decoder, measured_x = self._decoder(code, withheld_x)
        zero_z = {name: 0 for name in code.Z_STABILIZERS}
        zero_x = {name: 0 for name in measured_x}

        # Detector patterns produced by X_a and Z_d, respectively. Z checks
        # detect X errors, while X checks detect Z errors because the operators
        # anticommute on every shared data qubit.
        x_a = {name: int("a" in support) for name, support in code.Z_STABILIZERS.items()}
        z_d = {name: int("d" in support) for name, support in measured_x.items()}

        # Z checks have no known preparation value, so m1 is the reference and
        # the available detectors are d2 = m2 ^ m1 and d3 = m3 ^ m2. For every
        # Z check touched by X_a, the simulated fault begins after m1 and then
        # persists:
        #
        #   location of X_a             raw m1 m2 m3    d2 d3
        #   absent throughout                0  0  0     0  0
        #   begins between m1 and m2         0  1  1     1  0
        #
        # Therefore x_a appears in the first Z-detector layer and the second
        # layer is zero. The spatial pattern across all affected Z checks
        # identifies data qubit a, so the decoder returns an X_a correction.
        x_recovery = decoder.decode(
            z_delta_history=[x_a, zero_z],
            x_delta_history=[zero_x, zero_x, zero_x],
        )

        # X checks do have a known preparation value m0 = 0. Their detectors
        # are d1 = m1 ^ m0, d2 = m2 ^ m1, and d3 = m3 ^ m2. For every X check
        # touched by Z_d, the simulated fault begins after preparation but
        # before m1 and then persists:
        #
        #   location of Z_d             raw m0 m1 m2 m3    d1 d2 d3
        #   absent throughout                0  0  0  0     0  0  0
        #   begins between m0 and m1         0  1  1  1     1  0  0
        #
        # Thus z_d appears only in the preparation-boundary detector layer.
        # Its spatial pattern identifies data qubit d, producing a Z_d
        # correction rather than a timelike measurement-fault explanation.
        z_recovery = decoder.decode(
            z_delta_history=[zero_z, zero_z],
            x_delta_history=[z_d, zero_x, zero_x],
        )

        assert x_recovery.x_error_sites == {"a"}
        assert z_recovery.z_error_sites == {"d"}

    def test_transient_measurement_fault_uses_timelike_edge(self, code, withheld_x) -> None:
        """A one-round readout fault is matched in time, not to a data qubit.

        For one Z check, let d2 = m2 ^ m1 and d3 = m3 ^ m2. A measurement
        error only in m2 changes the raw history from 000 to 010 and therefore
        gives detector history [d2, d3] = [1, 1]. MWPM explains those two
        adjacent events with one timelike measurement-fault edge.

        A persistent data fault is different: if already present at m1 it
        changes 000 to 111 and gives [0, 0]; if it first appears between m1
        and m2 it changes 000 to 011 and gives [1, 0]. To reproduce [1, 1]
        with data faults, an error must both appear before m2 and disappear
        before m3 (010), requiring two data-fault transitions. The one-edge
        measurement-error explanation is therefore preferred, and the decoder
        returns no data correction.
        """
        decoder, measured_x = self._decoder(code, withheld_x)
        zero_z = {name: 0 for name in code.Z_STABILIZERS}
        zero_x = {name: 0 for name in measured_x}

        # Select one stabilizer in each sector and mark its adjacent detector
        # events. The particular stabilizer is irrelevant to this property.
        z_event = {**zero_z, next(iter(zero_z)): 1}
        x_event = {**zero_x, next(iter(zero_x)): 1}

        recovery = decoder.decode(
            # For raw Z-check outcomes m1, m2, m3, these layers are
            # d2 = m2 ^ m1 and d3 = m3 ^ m2:
            #
            #   cause                         raw m1 m2 m3    d2 d3
            #   one measurement error at m2       0  1  0     1  1
            #   data fault present before m1      1  1  1     0  0
            #   data fault begins before m2       0  1  1     1  0
            #   data fault begins and ends at m2  0  1  0     1  1
            #
            # Thus [1, 1] can be one measurement fault or two data-fault
            # transitions. Each individual edge has weight 1: the timelike
            # explanation costs 1, whereas appearing and disappearing costs
            # 1 + 1 = 2. MWPM therefore chooses the measurement-fault edge.
            z_delta_history=[z_event, z_event],
            # X checks additionally compare m1 with the known preparation
            # value m0 = 0, so their history includes d1 = m1 ^ m0:
            #
            #   cause                    raw m0 m1 m2 m3    d1 d2 d3
            #   measurement error at m2       0  0  1  0     0  1  1
            #   data fault before m1           0  1  1  1     1  0  0
            #   data fault before m2           0  0  1  1     0  1  0
            #   data fault only during m2      0  0  1  0     0  1  1
            #
            # The test models the first row: d1 is zero because round 1 agrees
            # with preparation, while d2 and d3 contain the adjacent events.
            # As in the Z sector, one timelike edge costs less than the two
            # data-fault transitions in the last row.
            x_delta_history=[zero_x, x_event, x_event],
        )

        # The decoder explains both sectors using measurement errors alone.
        assert recovery.is_trivial


def test_temporal_decoder_updates_frame_from_physical_fault_history() -> None:
    """Decode a physical Z_d fault from three measured rounds into the frame."""
    sim = AerSimulator(method="stabilizer")
    data = QuantumRegister(NUM_DATA_QUBITS, "data")
    z_anc = QuantumRegister(NUM_Z_ANCILLAS, "z_anc")
    x_anc = QuantumRegister(NUM_X_ANCILLAS, "x_anc")
    init_z_cl = ClassicalRegister(NUM_Z_ANCILLAS, "init_z")
    round_z_cls = [ClassicalRegister(NUM_Z_ANCILLAS, f"z_r{index}") for index in range(3)]
    round_x_cls = [ClassicalRegister(NUM_X_ANCILLAS, f"x_r{index}") for index in range(3)]
    log_cl = ClassicalRegister(len(d3.DEFAULT_LOGICAL_X), "log_x")
    circuit = QuantumCircuit(
        data,
        z_anc,
        x_anc,
        init_z_cl,
        *round_z_cls,
        *round_x_cls,
        log_cl,
    )

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
    append_pauli_faults(
        circuit,
        [("Z", "d")],
        qubit_map=d3.label2qubit(list(data)),
    )
    for index in range(3):
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
    d3.append_logical_measurement(
        circuit,
        "X",
        surface_qubits=list(data),
        classical_bits=list(log_cl),
        destructive=True,
    )

    raw = next(iter(sim.run(circuit, shots=1).result().get_counts()))
    parsed = parse_qiskit_registers(
        raw,
        [("init_z", Z_NAMES)]
        + [(f"z_r{index}", Z_NAMES) for index in range(3)]
        + [(f"x_r{index}", X_NAMES) for index in range(3)]
        + [("log_x", d3.DEFAULT_LOGICAL_X)],
    )

    stabilizer_frame = StabilizerFrame()
    stabilizer_frame.record_many(parsed["init_z"])
    stabilizer_frame.record_many({name: 0 for name in X_NAMES})
    z_delta_history = []
    x_delta_history = []
    for index in range(3):
        deltas = stabilizer_frame.compare_many(
            {**parsed[f"z_r{index}"], **parsed[f"x_r{index}"]},
            update=True,
        )
        z_delta_history.append({name: deltas[name] for name in Z_NAMES})
        x_delta_history.append({name: deltas[name] for name in X_NAMES})

    decoder = TemporalRotatedDecoder(
        data_qubits=d3.SURFACE_ORDER,
        z_stabilizers=d3.Z_STABILIZERS,
        x_stabilizers=d3.X_STABILIZERS,
        logical_x=d3.DEFAULT_LOGICAL_X,
        logical_z=d3.DEFAULT_LOGICAL_Z,
    )
    pauli_frame = PauliFrame()
    pauli_frame.register("L0")
    recovery = decoder.update_pauli_frame(
        pauli_frame=pauli_frame,
        logical_qubit_label="L0",
        z_delta_history=z_delta_history,
        x_delta_history=x_delta_history,
    )

    assert recovery.x_error_sites == set()
    assert recovery.z_error_sites == {"d"}
    assert pauli_frame.current_byproduct("L0") == "Z"
    raw_logical_x = sum(parsed["log_x"].values()) % 2
    assert pauli_frame.interpret_x_readout("L0", raw_logical_x) == 0
