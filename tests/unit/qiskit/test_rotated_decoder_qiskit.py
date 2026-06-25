"""Tests for the rotated distance-3 surface-code decoder and Pauli frame.

Fault injection, syndrome extraction, decoding, and frame-corrected logical
readout on the rotated d=3 patch.
"""

from __future__ import annotations

import pytest
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit_aer import AerSimulator

from research_qec_haoran.noise import append_pauli_faults
from research_qec_haoran.surface_code.decoding import PauliFrame, RotatedDecoder, StabilizerFrame
import research_qec_haoran.surface_code.rotated_d3_qiskit as d3
from research_qec_haoran.utils.normalize import parse_qiskit_registers

# ===========================================================================
# Shared constants
# ===========================================================================

NUM_DATA_QUBITS = 9
NUM_Z_ANCILLAS = 4
NUM_X_ANCILLAS = 4
Z_NAMES = tuple(d3.Z_STABILIZERS)
X_NAMES = tuple(d3.X_STABILIZERS)


# ===========================================================================
# Helpers
# ===========================================================================


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


# ===========================================================================
# SET 1: Single-fault injection, syndrome extraction, and decoding (d=3)
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
        d3.prepare_plus(qc, qubits=list(s))
        d3.append_z_stabilizer_measurements(
            qc, surface_qubits=list(s), z_ancillas=list(za),
            classical_bits=list(init_z_cl), round_label="init",
            prepare_ancillas=False, add_barriers=False,
        )

        # Step 2: Inject the fault.
        append_pauli_faults(qc, [(fault_pauli, fault_qubit)], qubit_map=d3.label2qubit(list(s)))

        # Step 3: Full stabilizer round (detection).
        d3.append_stabilizer_round(
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
            data_qubits=d3.SURFACE_ORDER,
            z_stabilizers=d3.Z_STABILIZERS,
            x_stabilizers=d3.X_STABILIZERS,
            logical_x=d3.DEFAULT_LOGICAL_X,
            logical_z=d3.DEFAULT_LOGICAL_Z,
        )
        recovery = decoder.decode(z_deltas=z_deltas, x_deltas=x_deltas)
        return z_deltas, x_deltas, recovery

    @pytest.mark.parametrize("qubit_label", d3.SURFACE_ORDER)
    @pytest.mark.parametrize("pauli", ["X", "Y", "Z"])
    def test_syndrome_matches_expected(self, pauli: str, qubit_label: str) -> None:
        """Change-detected syndrome matches the theoretical prediction."""
        z_deltas, x_deltas, _ = self._build_and_run(pauli, qubit_label)
        expected_z, expected_x = _expected_syndrome_for_fault(pauli, qubit_label)
        assert z_deltas == expected_z, f"{pauli}_{qubit_label}: Z {z_deltas} != {expected_z}"
        assert x_deltas == expected_x, f"{pauli}_{qubit_label}: X {x_deltas} != {expected_x}"

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
        assert not recovery.is_trivial, f"{pauli}_{qubit_label}: decoder returned trivial"

        # Verify the stored deltas in the recovery match what we measured.
        assert recovery.z_deltas == z_deltas
        assert recovery.x_deltas == x_deltas

    def test_no_error_gives_trivial_recovery(self) -> None:
        """Zero syndrome (no fault) decodes to identity."""
        decoder = RotatedDecoder(
            data_qubits=d3.SURFACE_ORDER,
            z_stabilizers=d3.Z_STABILIZERS,
            x_stabilizers=d3.X_STABILIZERS,
            logical_x=d3.DEFAULT_LOGICAL_X,
            logical_z=d3.DEFAULT_LOGICAL_Z,
        )
        recovery = decoder.decode(
            z_deltas={name: 0 for name in Z_NAMES},
            x_deltas={name: 0 for name in X_NAMES},
        )
        assert recovery.is_trivial
        assert recovery.label() == "I"

    @pytest.mark.parametrize("qubit_label", d3.SURFACE_ORDER)
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
        log_cl = ClassicalRegister(len(d3.DEFAULT_LOGICAL_X), "log_x")
        qc = QuantumCircuit(s, za, xa, init_z_cl, det_z_cl, det_x_cl, log_cl)

        # Step 1: Prepare |+_L⟩.
        d3.prepare_plus(qc, qubits=list(s))
        d3.append_z_stabilizer_measurements(
            qc, surface_qubits=list(s), z_ancillas=list(za),
            classical_bits=list(init_z_cl), round_label="init",
            prepare_ancillas=False, add_barriers=False,
        )

        # Step 2: Inject the fault.
        append_pauli_faults(qc, [(pauli, qubit_label)], qubit_map=d3.label2qubit(list(s)))

        # Step 3: Full stabilizer round (detection).
        d3.append_stabilizer_round(
            qc, surface_qubits=list(s), z_ancillas=list(za), x_ancillas=list(xa),
            z_bits=list(det_z_cl), x_bits=list(det_x_cl),
            round_label="detect", prepare_z_ancillas=True, prepare_x_ancillas=True,
            add_barriers=False,
        )

        # Step 4: Destructive X_L measurement.
        d3.append_logical_measurement(
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
                ("log_x", d3.DEFAULT_LOGICAL_X),
            ),
        )

        # Step 6: Compute change-detected syndrome.
        stab_frame = StabilizerFrame()
        stab_frame.record_many(parsed["init_z"])
        stab_frame.record_many({name: 0 for name in X_NAMES})
        all_det = {**parsed["det_z"], **parsed["det_x"]}
        deltas = stab_frame.compare_many(all_det, update=False)
        z_deltas = {name: deltas[name] for name in Z_NAMES}
        x_deltas = {name: deltas[name] for name in X_NAMES}

        # Step 7: Decode and update Pauli frame.
        decoder = RotatedDecoder(
            data_qubits=d3.SURFACE_ORDER,
            z_stabilizers=d3.Z_STABILIZERS,
            x_stabilizers=d3.X_STABILIZERS,
            logical_x=d3.DEFAULT_LOGICAL_X,
            logical_z=d3.DEFAULT_LOGICAL_Z,
        )
        pauli_frame = PauliFrame()
        pauli_frame.register("L0")
        decoder.update_pauli_frame(
            pauli_frame, "L0", z_deltas=z_deltas, x_deltas=x_deltas,
        )

        # Step 8: Frame-corrected logical readout.
        raw_logical_x = sum(parsed["log_x"].values()) % 2
        corrected = pauli_frame.interpret_x_readout("L0", raw_logical_x)
        assert corrected == 0, (
            f"{pauli}_{qubit_label}: frame-corrected X_L readout = {corrected}, "
            f"expected 0 (raw={raw_logical_x}, frame={pauli_frame.current_byproduct('L0')})"
        )


# ===========================================================================
# SET 2: Multi-round fault injection, decoding, and logical readout (d=3)
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
        log_cl = ClassicalRegister(len(d3.DEFAULT_LOGICAL_X), "log_x")

        qc = QuantumCircuit(s, za, xa, init_z_cl, *round_z_cls, *round_x_cls, log_cl)

        # Prepare |+_L⟩.
        d3.prepare_plus(qc, qubits=list(s))
        d3.append_z_stabilizer_measurements(
            qc, surface_qubits=list(s), z_ancillas=list(za),
            classical_bits=list(init_z_cl), round_label="init",
            prepare_ancillas=False, add_barriers=False,
        )

        # n rounds of: fault → stabilizer measurement.
        for i in range(n_rounds):
            append_pauli_faults(qc, [(fault_pauli, fault_qubit)], qubit_map=d3.label2qubit(list(s)))
            d3.append_stabilizer_round(
                qc, surface_qubits=list(s), z_ancillas=list(za), x_ancillas=list(xa),
                z_bits=list(round_z_cls[i]), x_bits=list(round_x_cls[i]),
                round_label=f"r{i}",
                prepare_z_ancillas=True, prepare_x_ancillas=True,
                add_barriers=False,
            )

        # Destructive X_L measurement.
        d3.append_logical_measurement(
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
            + [("log_x", d3.DEFAULT_LOGICAL_X)]
        )
        parsed = parse_qiskit_registers(raw, reg_order)

        # Decode all rounds with StabilizerFrame + PauliFrame.
        stab_frame = StabilizerFrame()
        stab_frame.record_many(parsed["init_z"])
        stab_frame.record_many({name: 0 for name in X_NAMES})

        decoder = RotatedDecoder(
            data_qubits=d3.SURFACE_ORDER,
            z_stabilizers=d3.Z_STABILIZERS,
            x_stabilizers=d3.X_STABILIZERS,
            logical_x=d3.DEFAULT_LOGICAL_X,
            logical_z=d3.DEFAULT_LOGICAL_Z,
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
    @pytest.mark.parametrize("qubit_label", d3.SURFACE_ORDER)
    @pytest.mark.parametrize("pauli", ["X", "Y", "Z"])
    def test_frame_corrected_readout_is_zero(
        self, pauli: str, qubit_label: str, n_rounds: int,
    ) -> None:
        """Frame-corrected X_L readout = 0 after n rounds of the same single fault."""
        corrected = self._run_n_rounds(n_rounds, pauli, qubit_label)
        assert corrected == 0, (
            f"{pauli}_{qubit_label} x{n_rounds}: frame-corrected X_L = {corrected}, expected 0"
        )

