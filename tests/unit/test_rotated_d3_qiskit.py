import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit_aer import AerSimulator

from research_qec_haoran.surface_code.decoding import StabilizerFrame
from research_qec_haoran.surface_code.decoding import (
    decode_rotated_d3_single_qubit_syndrome,
    rotated_d3_single_qubit_relative_syndrome,
)
from research_qec_haoran.surface_code.rotated_d3_layout_qiskit import (
    ROTATED_D3_SURFACE_ORDER,
    ROTATED_D3_X_STABILIZERS,
    ROTATED_D3_Z_STABILIZERS,
)
from research_qec_haoran.surface_code.rotated_d3_qiskit import (
    append_data_pauli,
    append_prepare_plus_state,
    append_rotated_d3_z_stabilizer_measurements,
    ROTATED_D3_INIT_Z_LABELS,
    append_surface_d3_stabilizer_round,
    rotated_d3_z_syndrome_bits_from_measurements,
)

_Z_NAMES = tuple(ROTATED_D3_Z_STABILIZERS)
_X_NAMES = tuple(ROTATED_D3_X_STABILIZERS)


def _build_and_run_detect_circuit(fault_pauli: str, fault_qubit: str):
    """Build init→fault→detect circuit, run, parse, decode from measurements."""
    sim = AerSimulator()
    s = QuantumRegister(9, "surface")
    za = QuantumRegister(4, "z_anc")
    xa = QuantumRegister(4, "x_anc")
    i_cl = ClassicalRegister(4, "init_z")
    dz_cl = ClassicalRegister(4, "det_z")
    dx_cl = ClassicalRegister(4, "det_x")
    qc = QuantumCircuit(s, za, xa, i_cl, dz_cl, dx_cl)

    append_prepare_plus_state(qc, qubits=list(s))
    append_rotated_d3_z_stabilizer_measurements(
        qc, surface_qubits=list(s), z_ancillas=list(za),
        classical_bits=list(i_cl), round_label="init",
        prepare_ancillas=False, add_barriers=False,
    )
    append_data_pauli(qc, list(s), fault_pauli, fault_qubit)
    append_surface_d3_stabilizer_round(
        qc, surface_qubits=list(s), z_ancillas=list(za), x_ancillas=list(xa),
        z_bits=list(dz_cl), x_bits=list(dx_cl),
        round_label="detect", prepare_z_ancillas=True, prepare_x_ancillas=True,
        add_barriers=False,
    )

    raw = next(iter(sim.run(qc, shots=1).result().get_counts())).replace(" ", "")
    offset = len(raw)
    regs = {}
    for rname, labels in [("init_z", _Z_NAMES), ("det_z", _Z_NAMES), ("det_x", _X_NAMES)]:
        offset -= len(labels)
        chunk = raw[offset:offset + len(labels)]
        regs[rname] = {l: int(b) for l, b in zip(labels, reversed(chunk), strict=False)}

    fr = StabilizerFrame()
    fr.record_many(regs["init_z"])
    fr.record_many({n: 0 for n in _X_NAMES})
    det_rel = fr.compare_many({**regs["det_z"], **regs["det_x"]}, update=False)
    dz = {n: det_rel[n] for n in _Z_NAMES}
    dx = {n: det_rel[n] for n in _X_NAMES}

    dec = decode_rotated_d3_single_qubit_syndrome(z_deltas=dz, x_deltas=dx)
    return dz, dx, dec


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


def test_rotated_d3_single_qubit_decoder_handles_no_error() -> None:
    recovery = decode_rotated_d3_single_qubit_syndrome(
        z_deltas={name: 0 for name in _Z_NAMES},
        x_deltas={name: 0 for name in _X_NAMES},
    )
    assert recovery.is_trivial
    assert recovery.label() == "I"


def test_rotated_d3_single_qubit_decoder_detects_all_faults_from_circuit() -> None:
    """Inject each single-qubit Pauli via a real gate, decode from circuit measurements."""
    for qubit_label in ROTATED_D3_SURFACE_ORDER:
        for pauli in ("X", "Y", "Z"):
            dz, dx, recovery = _build_and_run_detect_circuit(pauli, qubit_label)

            expected_z, expected_x = rotated_d3_single_qubit_relative_syndrome(
                pauli=pauli, qubit_label=qubit_label,
            )
            assert dz == expected_z, f"{pauli}_{qubit_label}: Z {dz} != {expected_z}"
            assert dx == expected_x, f"{pauli}_{qubit_label}: X {dx} != {expected_x}"
            assert not recovery.is_trivial, f"{pauli}_{qubit_label}: trivial recovery"


def test_rotated_d3_single_fault_decoded_from_circuit_measurements() -> None:
    """Detailed single-fault test: Y on qubit e, decoded from actual measurements."""
    dz, dx, recovery = _build_and_run_detect_circuit("Y", "e")

    expected_z, expected_x = rotated_d3_single_qubit_relative_syndrome(pauli="Y", qubit_label="e")
    assert dz == expected_z
    assert dx == expected_x
    assert recovery.label() == "Y_e"
