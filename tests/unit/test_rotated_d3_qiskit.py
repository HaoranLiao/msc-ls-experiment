import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit_aer import AerSimulator

from research_qec_haoran.surface_code.decoding import StabilizerFrame
from research_qec_haoran.surface_code.decoding import (
    apply_rotated_d3_data_pauli,
    decode_rotated_d3_single_qubit_syndrome,
    rotated_d3_single_qubit_relative_syndrome,
)
from research_qec_haoran.surface_code.rotated_d3_qiskit import (
    append_prepare_plus_state,
    append_rotated_d3_z_stabilizer_measurements,
    ROTATED_D3_INIT_Z_LABELS,
    append_surface_d3_stabilizer_round,
    build_rotated_d3_logical_plus_statevector,
    rotated_d3_logical_plus_expectations,
    rotated_d3_z_syndrome_bits_from_measurements,
)


class TestStabilizerFrame:
    def test_record_and_current(self) -> None:
        frame = StabilizerFrame()
        frame.record("Z_abde", 1)
        assert frame.current("Z_abde") == 1

    def test_compare_without_update(self) -> None:
        frame = StabilizerFrame()
        frame.record_many({"Z_abde": 1, "Z_cf": 0})
        assert frame.compare("Z_abde", 1, update=False) == 0
        assert frame.compare("Z_cf", 1, update=False) == 1
        assert frame.current("Z_cf") == 0

    def test_advance_updates_reference(self) -> None:
        frame = StabilizerFrame()
        frame.record_many({"Z_abde": 1, "Z_cf": 0})
        deltas = frame.advance({"Z_abde": 0, "Z_cf": 0})
        assert deltas == {"Z_abde": 1, "Z_cf": 0}
        assert frame.as_dict() == {"Z_abde": 0, "Z_cf": 0}


def test_rotated_d3_direct_initialization_circuit_shape() -> None:
    surface = QuantumRegister(9, "surface")
    z_anc = QuantumRegister(4, "z_anc")
    classical = ClassicalRegister(4, "m")
    circuit = QuantumCircuit(surface, z_anc, classical)
    append_prepare_plus_state(circuit, qubits=list(surface))
    append_rotated_d3_z_stabilizer_measurements(
        circuit,
        surface_qubits=list(surface),
        z_ancillas=list(z_anc),
        classical_bits=list(classical),
        round_label="init",
    )
    assert tuple(f"init_{name.lower()}" for name in ("Z_abde", "Z_cf", "Z_dg", "Z_efhi")) == ROTATED_D3_INIT_Z_LABELS
    assert circuit.num_qubits == 13


def test_measurement_labels_convert_to_z_syndrome() -> None:
    syndrome = rotated_d3_z_syndrome_bits_from_measurements(
        {
            "init_z_abde": 1,
            "init_z_cf": 0,
            "init_z_dg": 1,
            "init_z_efhi": 0,
        }
    )
    assert syndrome == {
        "Z_abde": 1,
        "Z_cf": 0,
        "Z_dg": 1,
        "Z_efhi": 0,
    }


def test_nontrivial_z_syndrome_branch_preserves_logical_plus() -> None:
    expectations = rotated_d3_logical_plus_expectations(
        {
            "Z_abde": 1,
            "Z_cf": 0,
            "Z_dg": 1,
            "Z_efhi": 0,
        }
    )

    assert expectations["Z_abde"] < -0.999999
    assert expectations["Z_cf"] > 0.999999
    assert expectations["Z_dg"] < -0.999999
    assert expectations["Z_efhi"] > 0.999999
    assert expectations["logical_X"] > 0.999999
    assert abs(expectations["logical_Z"]) < 1e-12


def test_rotated_d3_single_qubit_decoder_handles_no_error() -> None:
    recovery = decode_rotated_d3_single_qubit_syndrome(
        z_deltas={name: 0 for name in ("Z_abde", "Z_cf", "Z_dg", "Z_efhi")},
        x_deltas={name: 0 for name in ("X_ab", "X_bcef", "X_degh", "X_hi")},
    )
    assert recovery.is_trivial
    assert recovery.label() == "I"


def test_rotated_d3_single_qubit_decoder_restores_all_single_qubit_paulis() -> None:
    clean_state = build_rotated_d3_logical_plus_statevector(
        {
            "Z_abde": 1,
            "Z_cf": 0,
            "Z_dg": 1,
            "Z_efhi": 0,
        }
    )

    for qubit_label in ("a", "b", "c", "d", "e", "f", "g", "h", "i"):
        for pauli in ("X", "Y", "Z"):
            z_deltas, x_deltas = rotated_d3_single_qubit_relative_syndrome(
                pauli=pauli,
                qubit_label=qubit_label,
            )
            recovery = decode_rotated_d3_single_qubit_syndrome(
                z_deltas=z_deltas,
                x_deltas=x_deltas,
            )
            assert not recovery.is_trivial

            errored_state = apply_rotated_d3_data_pauli(
                clean_state,
                pauli=pauli,
                qubit_label=qubit_label,
            )
            corrected_state = apply_rotated_d3_data_pauli(
                errored_state,
                pauli=recovery.pauli,
                qubit_label=recovery.qubit_label,
            )

            overlap = abs(np.vdot(clean_state.data, corrected_state.data))
            assert overlap > 1 - 1e-12


def test_rotated_d3_physical_recovery_returns_to_recorded_branch() -> None:
    z_syndrome = {
        "Z_abde": 1,
        "Z_cf": 0,
        "Z_dg": 1,
        "Z_efhi": 0,
    }
    reference_syndrome = {**z_syndrome, **{name: 0 for name in ("X_ab", "X_bcef", "X_degh", "X_hi")}}
    clean_state = build_rotated_d3_logical_plus_statevector(z_syndrome)
    faulted_state = apply_rotated_d3_data_pauli(clean_state, pauli="Y", qubit_label="e")

    surface = QuantumRegister(9, "surface")
    z_anc = QuantumRegister(4, "z_anc")
    x_anc = QuantumRegister(4, "x_anc")
    classical = ClassicalRegister(8, "m")
    circuit = QuantumCircuit(surface, z_anc, x_anc, classical)
    circuit.initialize(faulted_state.data, list(surface))
    bits = list(classical)
    append_surface_d3_stabilizer_round(
        circuit,
        surface_qubits=list(surface),
        z_ancillas=list(z_anc),
        x_ancillas=list(x_anc),
        z_bits=bits[:4],
        x_bits=bits[4:],
        round_label="fault round",
    )

    counts = AerSimulator().run(circuit, shots=1).result().get_counts()
    bitstring = next(iter(counts))
    measured_bits = [int(bit) for bit in bitstring.replace(" ", "")]
    labels = ("Z_abde", "Z_cf", "Z_dg", "Z_efhi", "X_ab", "X_bcef", "X_degh", "X_hi")
    measurements = {label: bit for label, bit in zip(reversed(labels), measured_bits, strict=False)}

    frame = StabilizerFrame()
    frame.record_many(reference_syndrome)
    relative = frame.compare_many(measurements, update=False)
    z_deltas = {name: relative[name] for name in ("Z_abde", "Z_cf", "Z_dg", "Z_efhi")}
    x_deltas = {name: relative[name] for name in ("X_ab", "X_bcef", "X_degh", "X_hi")}

    recovery = decode_rotated_d3_single_qubit_syndrome(z_deltas=z_deltas, x_deltas=x_deltas)
    corrected_state = apply_rotated_d3_data_pauli(
        faulted_state,
        pauli=recovery.pauli,
        qubit_label=recovery.qubit_label,
    )

    recovery_circuit = QuantumCircuit(surface, z_anc, x_anc, classical)
    recovery_circuit.initialize(corrected_state.data, list(surface))
    append_surface_d3_stabilizer_round(
        recovery_circuit,
        surface_qubits=list(surface),
        z_ancillas=list(z_anc),
        x_ancillas=list(x_anc),
        z_bits=bits[:4],
        x_bits=bits[4:],
        round_label="corrected round",
    )
    corrected_counts = AerSimulator().run(recovery_circuit, shots=1).result().get_counts()
    corrected_bitstring = next(iter(corrected_counts))
    corrected_bits = [int(bit) for bit in corrected_bitstring.replace(" ", "")]
    corrected_measurements = {
        label: bit
        for label, bit in zip(reversed(labels), corrected_bits, strict=False)
    }
    corrected_relative = frame.advance(corrected_measurements)

    assert all(bit == 0 for bit in corrected_relative.values())
    overlap = abs(np.vdot(clean_state.data, corrected_state.data))
    assert overlap > 1 - 1e-12
