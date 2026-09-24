"""Qiskit integration tests for the distance-generic rotated-code decoders."""

from __future__ import annotations

from types import ModuleType

import pytest
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit_aer import AerSimulator

import msc.rotated_surface_code.d3_qiskit as d3
import msc.rotated_surface_code.d5_qiskit as d5
from msc.noise import append_pauli_faults
from msc.rotated_surface_code.decoding import (
    PauliFrame,
    RotatedDecoder,
    RotatedDetectedErrors,
    SpaceTimeRotatedDecoder,
    StabilizerFrame,
)
from msc.utils.normalize import parse_qiskit_registers

# =============================================================================
# Single-round decoding (d=3 and d=5)
# =============================================================================


def _run_single_fault(
    code: ModuleType,
    pauli: str,
    qubit_label: str,
) -> tuple[RotatedDetectedErrors, int]:
    """Return the recovery and corrected logical-X readout for one data fault."""
    z_names = tuple(code.Z_STABILIZERS)
    x_names = tuple(code.X_STABILIZERS)
    data = QuantumRegister(len(code.SURFACE_ORDER), "data")
    z_ancillas = QuantumRegister(len(z_names), "z_anc")
    x_ancillas = QuantumRegister(len(x_names), "x_anc")
    init_z = ClassicalRegister(len(z_names), "init_z")
    detect_z = ClassicalRegister(len(z_names), "detect_z")
    detect_x = ClassicalRegister(len(x_names), "detect_x")
    logical_x = ClassicalRegister(len(code.DEFAULT_LOGICAL_X), "logical_x")
    circuit = QuantumCircuit(
        data,
        z_ancillas,
        x_ancillas,
        init_z,
        detect_z,
        detect_x,
        logical_x,
    )
    surface_qubits = list(data)

    # Prepare |+_L> and establish the initial Z-check reference.
    code.prepare_product_plus(circuit, qubits=surface_qubits)
    code.append_scheduled_stabilizer_round(
        circuit,
        surface_qubits=surface_qubits,
        selected_stabilizer_names=z_names,
        ancillas=list(z_ancillas),
        bits=list(init_z),
        round_label="initialize",
        reset_ancillas_initially=False,
        add_barriers=False,
    )

    # Inject one data fault, then measure all stabilizers and logical X.
    append_pauli_faults(
        circuit,
        [(pauli, qubit_label)],
        qubit_map=code.label2qubit(surface_qubits),
    )
    code.append_scheduled_stabilizer_round(
        circuit,
        surface_qubits=surface_qubits,
        selected_stabilizer_names=z_names + x_names,
        ancillas=[*list(z_ancillas), *list(x_ancillas)],
        bits=[*list(detect_z), *list(detect_x)],
        round_label="detect",
        reset_ancillas_initially=True,
        add_barriers=False,
    )
    code.append_logical_measurement(
        circuit,
        "X",
        surface_qubits=surface_qubits,
        bits=list(logical_x),
        destructive=True,
    )

    # Parse the circuit record and decode relative to the preparation frame.
    raw_counts_key = next(
        iter(AerSimulator(method="stabilizer").run(circuit, shots=1).result().get_counts())
    )
    measurements = parse_qiskit_registers(
        raw_counts_key,
        (
            ("init_z", z_names),
            ("detect_z", z_names),
            ("detect_x", x_names),
            ("logical_x", code.DEFAULT_LOGICAL_X),
        ),
    )
    stabilizer_frame = StabilizerFrame()
    stabilizer_frame.record_many(measurements["init_z"])
    stabilizer_frame.record_many(dict.fromkeys(x_names, 0))
    pauli_frame = PauliFrame()
    pauli_frame.register("L0")
    recovery = RotatedDecoder(
        data_qubits=code.SURFACE_ORDER,
        z_stabilizers=code.Z_STABILIZERS,
        x_stabilizers=code.X_STABILIZERS,
        logical_x=code.DEFAULT_LOGICAL_X,
        logical_z=code.DEFAULT_LOGICAL_Z,
    ).decode_raw_syndromes(
        pauli_frame=pauli_frame,
        logical_qubit_label="L0",
        z_raw=measurements["detect_z"],
        x_raw=measurements["detect_x"],
        stab_frame=stabilizer_frame,
    )
    raw_logical_x = sum(measurements["logical_x"].values()) % 2
    return recovery, pauli_frame.interpret_x_readout("L0", raw_logical_x)


@pytest.mark.parametrize("code", (d3, d5), ids=("d3", "d5"))
def test_zero_syndrome_decodes_to_identity(code: ModuleType) -> None:
    recovery = RotatedDecoder(
        data_qubits=code.SURFACE_ORDER,
        z_stabilizers=code.Z_STABILIZERS,
        x_stabilizers=code.X_STABILIZERS,
        logical_x=code.DEFAULT_LOGICAL_X,
        logical_z=code.DEFAULT_LOGICAL_Z,
    ).decode(
        z_deltas=dict.fromkeys(code.Z_STABILIZERS, 0),
        x_deltas=dict.fromkeys(code.X_STABILIZERS, 0),
    )
    assert recovery.is_trivial


@pytest.mark.parametrize("code", (d3, d5), ids=("d3", "d5"))
@pytest.mark.parametrize("pauli", ("X", "Y", "Z"))
def test_single_fault_syndrome_and_corrected_readout(
    code: ModuleType,
    pauli: str,
) -> None:
    for qubit_label in code.SURFACE_ORDER:
        recovery, corrected_logical_x = _run_single_fault(code, pauli, qubit_label)
        expected_z_deltas = {
            name: int(pauli in {"X", "Y"} and qubit_label in support)
            for name, support in code.Z_STABILIZERS.items()
        }
        expected_x_deltas = {
            name: int(pauli in {"Z", "Y"} and qubit_label in support)
            for name, support in code.X_STABILIZERS.items()
        }
        assert recovery.z_deltas == expected_z_deltas
        assert recovery.x_deltas == expected_x_deltas
        assert not recovery.is_trivial
        assert corrected_logical_x == 0


# =============================================================================
# Sequential multi-round decoding (d=3 and d=5)
# =============================================================================


def _run_repeated_faults(
    code: ModuleType,
    pauli: str,
    qubit_label: str,
    rounds: int,
) -> int:
    """Return the corrected logical-X readout after repeated fault/QEC rounds."""
    z_names = tuple(code.Z_STABILIZERS)
    x_names = tuple(code.X_STABILIZERS)
    data = QuantumRegister(len(code.SURFACE_ORDER), "data")
    z_ancillas = QuantumRegister(len(z_names), "z_anc")
    x_ancillas = QuantumRegister(len(x_names), "x_anc")
    init_z = ClassicalRegister(len(z_names), "init_z")
    round_z_registers = [
        ClassicalRegister(len(z_names), f"z_round_{round_index}") for round_index in range(rounds)
    ]
    round_x_registers = [
        ClassicalRegister(len(x_names), f"x_round_{round_index}") for round_index in range(rounds)
    ]
    logical_x = ClassicalRegister(len(code.DEFAULT_LOGICAL_X), "logical_x")
    circuit = QuantumCircuit(
        data,
        z_ancillas,
        x_ancillas,
        init_z,
        *round_z_registers,
        *round_x_registers,
        logical_x,
    )
    surface_qubits = list(data)

    # Prepare |+_L> and establish the initial Z-check reference.
    code.prepare_product_plus(circuit, qubits=surface_qubits)
    code.append_scheduled_stabilizer_round(
        circuit,
        surface_qubits=surface_qubits,
        selected_stabilizer_names=z_names,
        ancillas=list(z_ancillas),
        bits=list(init_z),
        round_label="initialize",
        reset_ancillas_initially=False,
        add_barriers=False,
    )

    # Repeat fault injection and syndrome measurement before final readout.
    for round_index in range(rounds):
        append_pauli_faults(
            circuit,
            [(pauli, qubit_label)],
            qubit_map=code.label2qubit(surface_qubits),
        )
        code.append_scheduled_stabilizer_round(
            circuit,
            surface_qubits=surface_qubits,
            selected_stabilizer_names=z_names + x_names,
            ancillas=[*list(z_ancillas), *list(x_ancillas)],
            bits=[
                *list(round_z_registers[round_index]),
                *list(round_x_registers[round_index]),
            ],
            round_label=f"round_{round_index}",
            reset_ancillas_initially=True,
            add_barriers=False,
        )
    code.append_logical_measurement(
        circuit,
        "X",
        surface_qubits=surface_qubits,
        bits=list(logical_x),
        destructive=True,
    )

    # Parse every round and update the two frames sequentially.
    raw_counts_key = next(
        iter(AerSimulator(method="stabilizer").run(circuit, shots=1).result().get_counts())
    )
    measurements = parse_qiskit_registers(
        raw_counts_key,
        [("init_z", z_names)]
        + [(f"z_round_{index}", z_names) for index in range(rounds)]
        + [(f"x_round_{index}", x_names) for index in range(rounds)]
        + [("logical_x", code.DEFAULT_LOGICAL_X)],
    )
    stabilizer_frame = StabilizerFrame()
    stabilizer_frame.record_many(measurements["init_z"])
    stabilizer_frame.record_many(dict.fromkeys(x_names, 0))
    pauli_frame = PauliFrame()
    pauli_frame.register("L0")
    decoder = RotatedDecoder(
        data_qubits=code.SURFACE_ORDER,
        z_stabilizers=code.Z_STABILIZERS,
        x_stabilizers=code.X_STABILIZERS,
        logical_x=code.DEFAULT_LOGICAL_X,
        logical_z=code.DEFAULT_LOGICAL_Z,
    )
    for round_index in range(rounds):
        decoder.decode_raw_syndromes(
            pauli_frame=pauli_frame,
            logical_qubit_label="L0",
            z_raw=measurements[f"z_round_{round_index}"],
            x_raw=measurements[f"x_round_{round_index}"],
            stab_frame=stabilizer_frame,
        )
    raw_logical_x = sum(measurements["logical_x"].values()) % 2
    return pauli_frame.interpret_x_readout("L0", raw_logical_x)


@pytest.mark.parametrize("code", (d3, d5), ids=("d3", "d5"))
@pytest.mark.parametrize("rounds", (1, 3))
@pytest.mark.parametrize("pauli", ("X", "Y", "Z"))
def test_repeated_faults_have_corrected_logical_readout(
    code: ModuleType,
    rounds: int,
    pauli: str,
) -> None:
    for qubit_label in code.SURFACE_ORDER:
        assert _run_repeated_faults(code, pauli, qubit_label, rounds) == 0


# =============================================================================
# Spacetime decoding (d=3 and d=5)
# =============================================================================


@pytest.mark.parametrize(
    ("code", "withheld_x_labels"),
    ((d3, {"X_ab"}), (d5, {"X_ab", "X_cd"})),
    ids=("d3", "d5"),
)
def test_spacetime_data_faults_use_spatial_edges(
    code: ModuleType,
    withheld_x_labels: set[str],
) -> None:
    """Show persistent data faults use spatial edges in spacetime MWPM.

    A detector event is a *change* in a stabilizer outcome, rather than the raw
    outcome itself. Repeating the checks adds a time direction to the ordinary
    surface-code matching graph:

    - a spatial edge represents a Pauli error on a data qubit;
    - a timelike edge represents an incorrect ancilla measurement whose effect
      appears in two adjacent change-detection layers.

    During lattice surgery, ``X_ab`` (and ``X_cd`` for d=5) is temporarily
    replaced by merged-boundary checks, so only the remaining compatible X
    stabilizers are supplied to this decoder.

    The two CSS sectors intentionally have different history lengths. The
    surface patch starts in |+_L>, giving compatible X checks a known +1
    reference before round 1 and therefore one detector layer for each of the
    three rounds. Initial Z-check outcomes are not predetermined, so round 1
    establishes their reference and only the round-2 and round-3 changes are
    detectors. Thus these tests pass three X-check layers but two Z-check
    layers.

    A persistent data fault flips its checks in every later raw round, but its
    detector history contains an event only when the fault first appears. MWPM
    therefore returns the corresponding spatial data-qubit correction.
    """
    measured_x_stabilizers = {
        name: support
        for name, support in code.X_STABILIZERS.items()
        if name not in withheld_x_labels
    }
    decoder = SpaceTimeRotatedDecoder(
        data_qubits=code.SURFACE_ORDER,
        z_stabilizers=code.Z_STABILIZERS,
        x_stabilizers=measured_x_stabilizers,
    )
    zero_z_deltas = dict.fromkeys(code.Z_STABILIZERS, 0)
    zero_x_deltas = dict.fromkeys(measured_x_stabilizers, 0)
    x_fault_z_deltas = {name: int("a" in support) for name, support in code.Z_STABILIZERS.items()}
    z_fault_x_deltas = {
        name: int("d" in support) for name, support in measured_x_stabilizers.items()
    }

    # Z checks have no known preparation value, so m1 is the reference and
    # the available detectors are d2 = m2 ^ m1 and d3 = m3 ^ m2. For every
    # Z check touched by X_a, the simulated fault begins after m1 and then
    # persists:
    #
    #   location of X_a             raw m1 m2 m3    d2 d3
    #   absent throughout                0  0  0     0  0
    #   begins between m1 and m2         0  1  1     1  0
    #
    # Thus x_a appears only in the first Z-detector layer. Its spatial pattern
    # identifies data qubit a, producing an X_a correction.
    x_recovery = decoder.decode(
        z_delta_history=(x_fault_z_deltas, zero_z_deltas),
        x_delta_history=(zero_x_deltas, zero_x_deltas, zero_x_deltas),
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
        z_delta_history=(zero_z_deltas, zero_z_deltas),
        x_delta_history=(z_fault_x_deltas, zero_x_deltas, zero_x_deltas),
    )

    assert x_recovery.x_error_sites == {"a"}
    assert z_recovery.z_error_sites == {"d"}


@pytest.mark.parametrize(
    ("code", "withheld_x_labels"),
    ((d3, {"X_ab"}), (d5, {"X_ab", "X_cd"})),
    ids=("d3", "d5"),
)
def test_spacetime_measurement_faults_use_timelike_edges(
    code: ModuleType,
    withheld_x_labels: set[str],
) -> None:
    """Show a one-round measurement fault is matched in time, not to data.

    For one Z check, let d2 = m2 ^ m1 and d3 = m3 ^ m2. A measurement error
    only in m2 changes the raw history from 000 to 010 and therefore gives
    detector history [d2, d3] = [1, 1]. MWPM explains those adjacent events
    with one timelike measurement-fault edge.

    A persistent data fault instead produces [0, 0] if present at m1 or [1, 0]
    if it begins between m1 and m2. Reproducing [1, 1] with data faults requires
    the error to appear before m2 and disappear before m3, costing two spatial
    transitions. The one-edge timelike explanation is therefore preferred,
    and the decoder returns no data correction.
    """
    measured_x_stabilizers = {
        name: support
        for name, support in code.X_STABILIZERS.items()
        if name not in withheld_x_labels
    }
    decoder = SpaceTimeRotatedDecoder(
        data_qubits=code.SURFACE_ORDER,
        z_stabilizers=code.Z_STABILIZERS,
        x_stabilizers=measured_x_stabilizers,
    )
    zero_z_deltas = dict.fromkeys(code.Z_STABILIZERS, 0)
    zero_x_deltas = dict.fromkeys(measured_x_stabilizers, 0)
    z_measurement_fault_deltas = {
        **zero_z_deltas,
        next(iter(zero_z_deltas)): 1,
    }
    x_measurement_fault_deltas = {
        **zero_x_deltas,
        next(iter(zero_x_deltas)): 1,
    }

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
        # One timelike edge costs 1, while the two data-fault transitions
        # needed for [1, 1] cost 2, so MWPM chooses the measurement fault.
        z_delta_history=(z_measurement_fault_deltas, z_measurement_fault_deltas),
        # X checks also compare m1 with the known preparation value m0 = 0:
        #
        #   cause                    raw m0 m1 m2 m3    d1 d2 d3
        #   measurement error at m2       0  0  1  0     0  1  1
        #   data fault before m1           0  1  1  1     1  0  0
        #   data fault before m2           0  0  1  1     0  1  0
        #   data fault only during m2      0  0  1  0     0  1  1
        #
        # Here d1 is zero because round 1 agrees with preparation; d2 and d3
        # are again matched by one timelike edge.
        x_delta_history=(
            zero_x_deltas,
            x_measurement_fault_deltas,
            x_measurement_fault_deltas,
        ),
    )

    # Both sectors are explained by measurement faults, so no data recovery.
    assert recovery.is_trivial


@pytest.mark.parametrize(
    ("code", "withheld_x_labels", "measurement_stabilizer_name"),
    (
        (d3, {"X_ab"}, "Z_abde"),
        (d3, {"X_ab"}, "Z_cf"),
        (d5, {"X_ab", "X_cd"}, "Z_abfg"),
        (d5, {"X_ab", "X_cd"}, "Z_ej"),
    ),
    ids=("d3-overlapping", "d3-disjoint", "d5-overlapping", "d5-disjoint"),
)
def test_spacetime_decodes_selected_data_and_measurement_fault_pairs(
    code: ModuleType,
    withheld_x_labels: set[str],
    measurement_stabilizer_name: str,
) -> None:
    """Decode selected correctable pairs of one spatial and one timelike fault.

    An X_a data fault begins before m2 and persists, while one Z-check
    measurement is flipped only at m2. The first detector layer is the XOR of
    both faults; the second contains only the measurement fault. Cases where
    their first-layer detector events overlap and cancel are included alongside
    disjoint cases.

    This verifies representative two-fault patterns, not arbitrary pairs:
    distance 3 does not guarantee correction of every two-fault configuration.
    """
    measured_x_stabilizers = {
        name: support
        for name, support in code.X_STABILIZERS.items()
        if name not in withheld_x_labels
    }
    zero_z_deltas = dict.fromkeys(code.Z_STABILIZERS, 0)
    zero_x_deltas = dict.fromkeys(measured_x_stabilizers, 0)
    x_fault_z_deltas = {name: int("a" in support) for name, support in code.Z_STABILIZERS.items()}
    measurement_fault_z_deltas = {
        **zero_z_deltas,
        measurement_stabilizer_name: 1,
    }
    combined_first_z_deltas = {
        name: x_fault_z_deltas[name] ^ measurement_fault_z_deltas[name]
        for name in code.Z_STABILIZERS
    }

    recovery = SpaceTimeRotatedDecoder(
        data_qubits=code.SURFACE_ORDER,
        z_stabilizers=code.Z_STABILIZERS,
        x_stabilizers=measured_x_stabilizers,
    ).decode(
        z_delta_history=(combined_first_z_deltas, measurement_fault_z_deltas),
        x_delta_history=(zero_x_deltas, zero_x_deltas, zero_x_deltas),
    )

    assert recovery.x_error_sites == {"a"}
    assert recovery.z_error_sites == set()
