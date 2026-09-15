from __future__ import annotations

import math
from collections import Counter

import pytest
from qiskit.circuit import CircuitInstruction
from qiskit.circuit.library import XGate
from qiskit_aer import AerSimulator

from msc.state_transfer import common
from msc.state_transfer import helper_stim as transfer_stim_helper
from msc.state_transfer import to_d3_rotated_qiskit as d3
from msc.state_transfer import to_d5_rotated_qiskit as d5
from msc.state_transfer import to_d5_rotated_stim as d5_stim

TARGETS = (
    pytest.param(d3, d3.build_steane_to_rotated_d3_lattice_surgery_circuit, id="d3"),
    pytest.param(d5, d5.build_steane_to_rotated_d5_lattice_surgery_circuit, id="d5"),
)
EXPECTED_INTERFACE_CHECKS = {
    3: (
        ("Z1a", ("1",), ("a",)),
        ("Z35bc", ("3", "5"), ("b", "c")),
    ),
    5: (
        ("Z1a", ("1",), ("a",)),
        ("Z35bc", ("3", "5"), ("b", "c")),
        ("Zde", (), ("d", "e")),
    ),
}
EXPECTED_WITHHELD_X_CHECKS = {3: ("X_ab",), 5: ("X_ab", "X_cd")}
EXPECTED_STEANE_VERIFICATION_READOUT_SITES = {
    11: {
        "Z0145": 6,
        "Z0235": 7,
        "X0235": 7,
        "Z0246": 2,
        "X0246": 2,
    },
    14: {
        "Z0145": 8,
        "Z0235": 12,
        "X0235": 11,
        "Z0246": 10,
        "X0246": 9,
    },
}
EXPECTED_STEANE_BELL_PARTNER_FLAG_SITES = {
    "Z0145": 7,
    "Z0235": 11,
    "X0235": 12,
    "Z0246": 9,
    "X0246": 10,
}

LOGICAL_EXPECTATIONS = {
    "0": {"X": 0.0, "Y": 0.0, "Z": 1.0},
    "1": {"X": 0.0, "Y": 0.0, "Z": -1.0},
    "+": {"X": 1.0, "Y": 0.0, "Z": 0.0},
    "-": {"X": -1.0, "Y": 0.0, "Z": 0.0},
    "i": {"X": 0.0, "Y": 1.0, "Z": 0.0},
    "-i": {"X": 0.0, "Y": -1.0, "Z": 0.0},
}


# =============================================================================
# Hirano merge schedule and measurement rules
# =============================================================================


def test_14_site_verification_groups_reuse_each_bell_pair_once() -> None:
    """Require simultaneous same-face extraction and skip bare X0145."""
    assert common.group_14_site_superdense_verification_checks(
        common.STEANE_MERGE_VERIFICATION_ROUNDS[0]
    ) == (
        common.SteaneSuperdenseVerificationGroup("Z", (0,), ("Z0145",)),
        common.SteaneSuperdenseVerificationGroup(
            "BOTH",
            (1, 2),
            ("X0235", "Z0235", "X0246", "Z0246"),
        ),
    )
    assert common.group_14_site_superdense_verification_checks(
        common.STEANE_MERGE_VERIFICATION_ROUNDS[1]
    ) == (
        common.SteaneSuperdenseVerificationGroup(
            "Z",
            (0, 1, 2),
            ("Z0145", "Z0235", "Z0246"),
        ),
    )
    assert not common.group_14_site_superdense_verification_checks(
        common.STEANE_MERGE_VERIFICATION_ROUNDS[2]
    )
    assert all(
        check_name != "X0145"
        for round_checks in common.STEANE_MERGE_VERIFICATION_ROUNDS
        for check_name in round_checks
    )


@pytest.mark.parametrize(("module", "build"), TARGETS)
@pytest.mark.parametrize("patch_sites", [11, 14])
def test_measurements_follow_hirano_merge_table(module, build, patch_sites: int) -> None:
    """Verify the built circuit against every row of the merge table.

    Each of the three merge rounds must measure interface checks first, all
    surface Z checks plus only the compatible surface X checks second, and the
    listed color-code checks last. The third color-code list is deliberately
    empty. Demolition then measures every Steane data qubit in X before the
    one ``X_ab`` completion check is measured.

    The d=3 target is the compact projection of the same rules: it omits
    ``Zde``. At either distance, completion measures only ``X_ab``. The mixed
    ``Z35bc`` check must retain
    the hook-safe ``3 -> b -> 5 -> c`` CNOT order at either distance.
    """
    spec = module.TRANSFER_SPEC
    patch_spec = common.resolve_steane_transfer_patch_spec(patch_sites)
    expected_interface_checks = EXPECTED_INTERFACE_CHECKS[spec.distance]
    expected_withheld_x_checks = EXPECTED_WITHHELD_X_CHECKS[spec.distance]
    expected_color_checks = (
        ("Z0145", "Z0235", "X0235", "Z0246", "X0246"),
        ("Z0145", "Z0235", "Z0246"),
        (),
    )
    expected_final_x_checks = {
        "X0235": ("0", "2", "3", "5"),
        "X0246": ("0", "2", "4", "6"),
        "X0145ab": ("0", "1", "4", "5", "X_ab"),
    }

    # Check the target geometry and the exact table-defined measurement sets.
    assert spec.merged_boundary_checks == expected_interface_checks
    assert spec.withheld_surface_stabilizer_labels == expected_withheld_x_checks
    assert spec.merged_x_labels == tuple(
        label for label in spec.x_labels if label not in expected_withheld_x_checks
    )
    assert common.STEANE_MERGE_VERIFICATION_ROUNDS == expected_color_checks
    assert common.FINAL_X_VERIFICATION_CHECKS == expected_final_x_checks

    # Index every physical measurement by semantic family and label.
    circuit = build(
        patch_sites=patch_sites,
        injected_state="0",
        include_cultivation=False,
        add_barriers=False,
    )
    labels = common.transfer_measurement_labels(
        spec,
        patch_sites=patch_sites,
        include_cultivation=False,
        surface_readout_basis=None,
    )
    measurement_by_bit = {
        instruction.clbits[0]: (index, instruction)
        for index, instruction in enumerate(circuit.data)
        if instruction.operation.name == "measure"
    }
    positions = {
        family: {
            label: measurement_by_bit[bit][0]
            for label, bit in zip(
                family_labels,
                next(register for register in circuit.cregs if register.name == family),
                strict=True,
            )
        }
        for family, family_labels in labels.items()
    }
    qubit_names = {register[0]: register.name for register in circuit.qregs}

    # Verify interface -> surface -> color-code ordering in all three rounds.
    previous_round_end = max(
        *positions["surface_init_z"].values(),
        *positions["surface_init_x"].values(),
    )
    for round_index, color_checks in enumerate(expected_color_checks, start=1):
        prefix = f"round{round_index}_"
        interface_positions = tuple(
            positions["interface_z"][prefix + check_name]
            for check_name, _, _ in expected_interface_checks
        )
        surface_positions = tuple(
            positions[family][prefix + check_name]
            for family, check_names in (
                ("merged_surface_z", spec.z_labels),
                ("merged_surface_x", spec.merged_x_labels),
            )
            for check_name in check_names
        )
        color_positions = tuple(
            positions["steane_verify"][prefix + check_name] for check_name in color_checks
        )
        bell_partner_flag_positions = (
            tuple(
                position
                for label, position in positions["steane_bell_partner_flags"].items()
                if label.startswith(prefix)
            )
            if patch_sites == 14
            else ()
        )
        all_color_positions = (*color_positions, *bell_partner_flag_positions)

        assert previous_round_end < min(interface_positions)
        assert max(interface_positions) < min(surface_positions)
        if all_color_positions:
            assert max(surface_positions) < min(all_color_positions)
        assert tuple(
            qubit_names[circuit.data[position].qubits[0]] for position in interface_positions
        ) == tuple(f"zz_{check_name[1:].lower()}" for check_name, _, _ in expected_interface_checks)
        assert tuple(
            qubit_names[circuit.data[position].qubits[0]] for position in surface_positions
        ) == tuple(label.lower() for label in spec.z_labels + spec.merged_x_labels)
        assert (
            tuple(
                label.removeprefix(prefix)
                for label in labels["steane_verify"]
                if label.startswith(prefix)
            )
            == color_checks
        )
        assert tuple(
            qubit_names[circuit.data[position].qubits[0]] for position in color_positions
        ) == tuple(
            patch_spec.site_labels[
                EXPECTED_STEANE_VERIFICATION_READOUT_SITES[patch_sites][check_name]
            ]
            for check_name in color_checks
        )
        if patch_sites == 14:
            expected_flag_sites = (
                (
                    EXPECTED_STEANE_BELL_PARTNER_FLAG_SITES["Z0145"],
                    13,
                )
                if round_index == 1
                else (
                    *(
                        EXPECTED_STEANE_BELL_PARTNER_FLAG_SITES[check_name]
                        for check_name in color_checks
                    ),
                    *((13,) if round_index == 2 else ()),
                )
            )
            assert tuple(
                qubit_names[circuit.data[position].qubits[0]]
                for position in bell_partner_flag_positions
            ) == tuple(patch_spec.site_labels[site] for site in expected_flag_sites)
            if bell_partner_flag_positions:
                assert bell_partner_flag_positions[0] < color_positions[0]
            if round_index == 1:
                assert max(color_positions) < bell_partner_flag_positions[-1]
        elif patch_sites == 11:
            assert "steane_bell_partner_flags" not in positions
        else:  # pragma: no cover - parametrization uses only supported patches.
            raise ValueError(f"Unsupported patch size: {patch_sites}.")
        previous_round_end = max((*surface_positions, *all_color_positions))

    # Verify the alternating mixed-code path used to suppress Z35bc hooks.
    z35bc_ancilla = next(register[0] for register in circuit.qregs if register.name == "zz_35bc")
    assert (
        tuple(
            qubit_names[instruction.qubits[0]]
            for instruction in circuit.data
            if instruction.operation.name == "cx" and instruction.qubits[1] == z35bc_ancilla
        )
        == ("3", "b", "5", "c") * common.LATTICE_SURGERY_ROUNDS
    )

    # Verify X-basis Steane demolition precedes top-edge X-check recovery.
    demolition_positions = tuple(positions["steane_x"].values())
    recovery_positions = tuple(positions["surface_x_recovery"].values())
    assert previous_round_end < min(demolition_positions)
    assert max(demolition_positions) < min(recovery_positions)
    assert (
        tuple(qubit_names[circuit.data[position].qubits[0]] for position in demolition_positions)
        == common.HIRANO_STEANE_DATA_ORDER
    )
    assert tuple(
        qubit_names[circuit.data[position].qubits[0]] for position in recovery_positions
    ) == tuple(label.lower() for label in common.SURFACE_RECOVERY_X_LABELS)
    recovery_ancilla = circuit.data[recovery_positions[0]].qubits[0]
    recovery_reset_position = max(
        index
        for index, instruction in enumerate(circuit.data[: recovery_positions[0]])
        if instruction.operation.name == "reset" and recovery_ancilla in instruction.qubits
    )
    assert max(demolition_positions) < recovery_reset_position
    for position in demolition_positions:
        measured_qubit = circuit.data[position].qubits[0]
        previous_operation = next(
            instruction.operation.name
            for instruction in reversed(circuit.data[:position])
            if measured_qubit in instruction.qubits
        )
        assert previous_operation == "h"


# =============================================================================
# Logical correctness through the public readout APIs
# =============================================================================


@pytest.mark.slow
@pytest.mark.parametrize(("module", "build"), TARGETS)
@pytest.mark.parametrize("patch_sites", [11, 14])
def test_clifford_state_pauli_expectation_values_without_cultivation(
    module,
    build,
    patch_sites: int,
) -> None:
    """Check supported Pauli expectations of all six Pauli eigenstates.

    Raw surface readout is interpreted only after the teleportation and
    spacetime-decoder byproducts have been folded into the logical Pauli frame.
    Both targets use full-data X/Z readout to close the compatible syndrome
    history. A single destructive product-basis readout cannot close both
    histories for logical Y.
    """
    experiments = tuple(
        (state, basis, expected)
        for state, expectations in LOGICAL_EXPECTATIONS.items()
        for basis, expected in expectations.items()
        if basis in ("X", "Z")
    )
    circuits = []
    for state, basis, _ in experiments:
        if module is d5:
            circuit = d5.build_steane_to_rotated_d5_lattice_surgery_circuit(
                patch_sites=patch_sites,
                injected_state=state,
                include_cultivation=False,
                surface_readout_basis=basis,
                full_post_selection=False,
                add_barriers=False,
            )
        else:
            circuit = build(
                patch_sites=patch_sites,
                injected_state=state,
                include_cultivation=False,
                surface_readout_basis=basis,
                add_barriers=False,
            )
        circuits.append(circuit)
    result = (
        AerSimulator(method="stabilizer")
        .run(
            circuits,
            shots=256,
            seed_simulator=42,
        )
        .result()
    )

    for (state, basis, expected), counts in zip(experiments, result.get_counts(), strict=True):
        logical_counts: Counter[int] = Counter()
        for bitstring, count in counts.items():
            measurements = module.parse_transfer_measurements(
                bitstring,
                patch_sites=patch_sites,
                include_cultivation=False,
                surface_readout_basis=basis,
            )
            expected_readout_stabilizers = (
                module.TRANSFER_SPEC.x_stabilizers
                if basis == "X"
                else module.TRANSFER_SPEC.z_stabilizers
            )
            surface_data_readout = measurements["surface_data_readout"]
            surface_readout_stabilizers = measurements["surface_readout_stabilizers"]
            assert isinstance(surface_data_readout, dict)
            assert isinstance(surface_readout_stabilizers, dict)
            assert set(surface_data_readout) == set(module.TRANSFER_SPEC.surface_order)
            assert set(surface_readout_stabilizers) == set(expected_readout_stabilizers)
            if common.transfer_acceptance_succeeds(measurements):
                logical_counts[
                    common.corrected_surface_logical_readout(measurements, basis=basis)
                ] += count
        accepted = sum(logical_counts.values())
        assert accepted >= 200, f"{state}/{basis} accepted only {accepted} shots"
        expectation = (logical_counts[0] - logical_counts[1]) / accepted
        assert expectation == pytest.approx(expected, abs=0.2)


@pytest.mark.slow
@pytest.mark.parametrize("patch_sites", [11, 14])
@pytest.mark.parametrize("cultivated_state", ("T", "Ty"))
def test_magic_state_pauli_expectation_values_with_cultivation(
    patch_sites: int,
    cultivated_state: str,
) -> None:
    """Check accessible magic-state Bloch components after cultivation and transfer.

    Exact Clifford-state tests above exercise d=5's full-data boundary. This
    statistical non-Clifford check stays on d=3's full-data X/Z API; d=5
    full-data MPS sampling is prohibitively expensive. X/Z readout covers all
    Ty components and the accessible X/Z components of T; fault-tolerant
    logical-Y tomography is not implemented.
    """
    expected_by_basis = {
        "X": 1 / math.sqrt(2),
        "Z": 0.0 if cultivated_state == "T" else 1 / math.sqrt(2),
    }
    circuits = [
        d3.build_steane_to_rotated_d3_lattice_surgery_circuit(
            patch_sites=patch_sites,
            injected_state=cultivated_state,
            cultivated_state=cultivated_state,
            include_cultivation=True,
            surface_readout_basis=basis,
            add_barriers=False,
        )
        for basis in expected_by_basis
    ]
    result = (
        AerSimulator(method="matrix_product_state")
        .run(
            circuits,
            shots=64,
            seed_simulator=42,
        )
        .result()
    )

    for basis, counts in zip(expected_by_basis, result.get_counts(), strict=True):
        logical_counts: Counter[int] = Counter()
        for bitstring, count in counts.items():
            measurements = d3.parse_transfer_measurements(
                bitstring,
                patch_sites=patch_sites,
                include_cultivation=True,
                surface_readout_basis=basis,
            )
            if common.transfer_acceptance_succeeds(measurements):
                logical_counts[
                    common.corrected_surface_logical_readout(measurements, basis=basis)
                ] += count
        accepted = sum(logical_counts.values())
        assert accepted
        expected = expected_by_basis[basis]
        expectation = (logical_counts[0] - logical_counts[1]) / accepted
        tolerance = 4 * math.sqrt((1 - expected**2) / accepted) + 0.02
        assert expectation == pytest.approx(expected, abs=tolerance)


# =============================================================================
# Post-transfer continuation and full-data readout
# =============================================================================


@pytest.mark.parametrize(("module", "build"), TARGETS)
@pytest.mark.parametrize("full_post_selection", (False, True))
@pytest.mark.parametrize("readout_basis", ("X", "Z"))
def test_post_transfer_branch_and_full_data_readout(
    module,
    build,
    full_post_selection: bool,
    readout_basis: str,
) -> None:
    """Expose both continuation lengths and measure every target data qubit."""
    circuit = build(
        injected_state="0",
        include_cultivation=False,
        surface_readout_basis=readout_basis,
        full_post_selection=full_post_selection,
        add_barriers=False,
    )
    expected_rounds = 1 if full_post_selection else module.TRANSFER_SPEC.distance
    registers = {register.name: register for register in circuit.cregs}
    assert len(registers["post_transfer_surface_z"]) == (
        expected_rounds * len(module.TRANSFER_SPEC.z_labels)
    )
    assert len(registers["post_transfer_surface_x"]) == (
        expected_rounds * len(module.TRANSFER_SPEC.x_labels)
    )
    data_register = registers[f"surface_data_{readout_basis.lower()}"]
    assert len(data_register) == len(module.TRANSFER_SPEC.surface_order)

    qubit_names = {register[0]: register.name for register in circuit.qregs}
    data_measurements = {
        instruction.clbits[0]: qubit_names[instruction.qubits[0]]
        for instruction in circuit.data
        if instruction.operation.name == "measure" and instruction.clbits[0] in data_register
    }
    assert tuple(data_measurements[bit] for bit in data_register) == (
        module.TRANSFER_SPEC.surface_order
    )


def test_d5_post_transfer_temporal_decoding() -> None:
    """Decode a middle-round measurement flip as a timelike fault.

    Flipping one middle-round Z-ancilla result creates two adjacent temporal
    detection events and therefore no data recovery.
    """
    temporal_fault_circuit = d5.build_steane_to_rotated_d5_lattice_surgery_circuit(
        injected_state="0",
        include_cultivation=False,
        surface_readout_basis="Z",
        add_barriers=False,
    )
    z_register = next(
        register
        for register in temporal_fault_circuit.cregs
        if register.name == "post_transfer_surface_z"
    )
    middle_z_bit = z_register[2 * len(d5.TRANSFER_SPEC.z_labels)]
    middle_z_measurement_index = next(
        index
        for index, instruction in enumerate(temporal_fault_circuit.data)
        if instruction.operation.name == "measure" and instruction.clbits == (middle_z_bit,)
    )
    middle_z_ancilla = temporal_fault_circuit.data[middle_z_measurement_index].qubits[0]
    temporal_fault_circuit.data.insert(
        middle_z_measurement_index,
        CircuitInstruction(XGate(), (middle_z_ancilla,), ()),
    )

    counts = (
        AerSimulator(method="stabilizer")
        .run(temporal_fault_circuit, shots=1, seed_simulator=42)
        .result()
        .get_counts()
    )
    temporal_measurements = d5.parse_transfer_measurements(
        next(iter(counts)),
        include_cultivation=False,
        surface_readout_basis="Z",
    )
    assert common.transfer_acceptance_succeeds(temporal_measurements)
    temporal_recovery = temporal_measurements["post_transfer_surface_qec_recovery"]
    assert tuple(delta_round["Z_abfg"] for delta_round in temporal_recovery.z_delta_history) == (
        0,
        0,
        1,
        1,
        0,
        0,
    )
    assert temporal_recovery.is_trivial


@pytest.mark.parametrize(
    ("readout_basis", "injected_state", "recovery_sites_name"),
    (
        pytest.param("Z", "0", "x_error_sites", id="Z-readout-X-fault"),
        pytest.param("X", "+", "z_error_sites", id="X-readout-Z-fault"),
    ),
)
@pytest.mark.parametrize(("module", "build"), TARGETS)
def test_full_data_readout_boundary_is_reconstructed_and_decoded(
    module,
    build,
    readout_basis: str,
    injected_state: str,
    recovery_sites_name: str,
) -> None:
    """Recover an anticommuting final fault through the readout syndrome.

    The inserted X is immediately before computational measurement. For
    X-basis readout this is after the basis-change H and is equivalent to a Z
    fault immediately before that H.
    """
    circuit = build(
        injected_state=injected_state,
        include_cultivation=False,
        surface_readout_basis=readout_basis,
        add_barriers=False,
    )
    data_register = next(
        register
        for register in circuit.cregs
        if register.name == f"surface_data_{readout_basis.lower()}"
    )
    first_data_measurement_index = next(
        index
        for index, instruction in enumerate(circuit.data)
        if instruction.operation.name == "measure" and instruction.clbits == (data_register[0],)
    )
    first_data_qubit = circuit.data[first_data_measurement_index].qubits[0]
    circuit.data.insert(
        first_data_measurement_index,
        CircuitInstruction(XGate(), (first_data_qubit,), ()),
    )

    counts = (
        AerSimulator(method="stabilizer")
        .run(circuit, shots=1, seed_simulator=42)
        .result()
        .get_counts()
    )
    measurements = module.parse_transfer_measurements(
        next(iter(counts)),
        include_cultivation=False,
        surface_readout_basis=readout_basis,
    )
    recovery = measurements["post_transfer_surface_qec_recovery"]

    assert common.transfer_acceptance_succeeds(measurements)
    assert getattr(recovery, recovery_sites_name) == {"a"}
    assert (
        common.corrected_surface_logical_readout(
            measurements,
            basis=readout_basis,
        )
        == 0
    )


@pytest.mark.slow
def test_d5_qiskit_measurements_match_stim_acceptance_detector_accounting() -> None:
    """Document every Qiskit-bit to Stim-detector count used for acceptance.

    Feeder-dependent known-zero measurements and flags::

        +-------+-----+-----+------+-------------+----------------+----------------+
        | Patch | Enc | QED | Cult | Cult flags  | Transfer flags | Strict flags   |
        +=======+=====+=====+======+=============+================+================+
        | 11    | 4   | 6   | 5    | 4           | 0              | 4 off / 8 on   |
        +-------+-----+-----+------+-------------+----------------+----------------+
        | 14    | 5   | 7   | 6    | 4           | 6              | 12 off / 16 on |
        +-------+-----+-----+------+-------------+----------------+----------------+

    ``Enc``, ``QED``, and ``Cult`` each map one Qiskit bit to one Stim
    detector. Strict flags count encoder flags, the 14-site QED D2-copy flag,
    cultivation fixed-zero helper flags, and transfer Bell-partner/D2-copy
    flags; they do not count QED stabilizers, the cultivation root projection,
    or its relational repeated-root check.

    Shared d=5 phase accounting::

        +--------------------------+-------------+----------------+----------------------+
        | Phase                    | Qiskit bits | Stim detectors | Correction treatment |
        +==========================+=============+================+======================+
        | Pre-merge surface Z      | 12          | 0              | reference only       |
        +--------------------------+-------------+----------------+----------------------+
        | Pre-merge surface X      | 12          | 12             | postselect 2         |
        +--------------------------+-------------+----------------+----------------------+
        | Interface, three rounds  | 9           | 6              | postselect all 6     |
        +--------------------------+-------------+----------------+----------------------+
        | Merged surface Z         | 36          | 36             | decode all           |
        +--------------------------+-------------+----------------+----------------------+
        | Merged surface X         | 30          | 30             | postselect 6         |
        +--------------------------+-------------+----------------+----------------------+
        | Steane merge verify      | 8           | 8              | postselect all 8     |
        +--------------------------+-------------+----------------+----------------------+
        | Steane demolition + m_ab | 8           | 3              | postselect 3 parities|
        +--------------------------+-------------+----------------+----------------------+
        | Five-round continuation  | 120         | 119            | postselect 10        |
        +--------------------------+-------------+----------------+----------------------+
        | One-round continuation   | 24          | 23             | full branch only     |
        +--------------------------+-------------+----------------+----------------------+
        | Full-data readout        | 25          | 12             | decode/full-select   |
        +--------------------------+-------------+----------------+----------------------+

    Aggregate acceptance counts. ``Off`` and ``on`` mean cultivation disabled
    and enabled, respectively::

        +-----------------------------------------+--------+-------+--------+-------+
        | Quantity                                | 11/off | 11/on | 14/off | 14/on |
        +=========================================+========+=======+========+=======+
        | Physical known-zero flag bits           | 4      | 8     | 12     | 16    |
        +-----------------------------------------+--------+-------+--------+-------+
        | Always-postselected protocol detectors | 27     | 32    | 35     | 41    |
        +-----------------------------------------+--------+-------+--------+-------+
        | Correction postselected detectors      | 45     | 50    | 53     | 59    |
        +-----------------------------------------+--------+-------+--------+-------+
        | Correction decoded surface detectors   | 191    | 191   | 191    | 191   |
        +-----------------------------------------+--------+-------+--------+-------+
        | Full-postselection rejected detectors  | 140    | 145   | 148    | 154   |
        +-----------------------------------------+--------+-------+--------+-------+

    Aggregate circuit sizes::

        +---------------------------------------+--------+-------+--------+-------+
        | Quantity                              | 11/off | 11/on | 14/off | 14/on |
        +=======================================+========+=======+========+=======+
        | Correction raw measurement bits       | 270    | 275   | 278    | 284   |
        +---------------------------------------+--------+-------+--------+-------+
        | Correction total Stim detectors       | 236    | 241   | 244    | 250   |
        +---------------------------------------+--------+-------+--------+-------+
        | Full-postselection raw measurement bits| 174    | 179   | 182    | 188   |
        +---------------------------------------+--------+-------+--------+-------+
        | Full-postselection total Stim detectors| 140    | 145   | 148    | 154   |
        +---------------------------------------+--------+-------+--------+-------+

    The correction branch postselects the protocol detectors plus the 18
    persistent ``X_bcgh``/``X_fgkl`` events. It decodes the other 191 surface
    events. The full-postselection branch rejects any nonzero detector. These
    tables apply equally to X and Z readout.
    """
    shared_counts = {
        "surface_init_z_bits": 12,
        "surface_init_z_detectors": 0,
        "surface_init_x_bits": 12,
        "surface_init_x_detectors": 12,
        "interface_bits": 9,
        "interface_change_detectors": 6,
        "merged_z_bits_and_detectors": 36,
        "merged_x_bits_and_detectors": 30,
        "steane_verification_bits_and_detectors": 8,
        "steane_demolition_and_m_ab_bits": 8,
        "final_x_verification_detectors": 3,
        "surface_data_readout_bits": 25,
        "readout_boundary_detectors": 12,
    }
    patch_counts: dict[common.SteanePatchSites, dict[str, int]] = {
        11: {
            "encoder": 4,
            "qed": 6,
            "cultivation": 5,
            "cultivation_helper_flags": 4,
            "transfer_flags": 0,
            "strict_flags_without_cultivation": 4,
            "transfer_bits_without_cultivation": 125,
            "protocol_detectors_without_cultivation": 27,
            "correction_acceptance_events_without_cultivation": 45,
            "correction_measurements_without_cultivation": 270,
            "full_postselection_measurements_without_cultivation": 174,
            "correction_detectors_without_cultivation": 236,
            "full_postselection_detectors_without_cultivation": 140,
        },
        14: {
            "encoder": 5,
            "qed": 7,
            "cultivation": 6,
            "cultivation_helper_flags": 4,
            "transfer_flags": 6,
            "strict_flags_without_cultivation": 12,
            "transfer_bits_without_cultivation": 133,
            "protocol_detectors_without_cultivation": 35,
            "correction_acceptance_events_without_cultivation": 53,
            "correction_measurements_without_cultivation": 278,
            "full_postselection_measurements_without_cultivation": 182,
            "correction_detectors_without_cultivation": 244,
            "full_postselection_detectors_without_cultivation": 148,
        },
    }
    branch_counts = {
        False: {
            "rounds": 5,
            "continuation_bits": 120,
            "continuation_detectors": 119,
        },
        True: {
            "rounds": 1,
            "continuation_bits": 24,
            "continuation_detectors": 23,
        },
    }

    assert len(d5.TRANSFER_SPEC.z_labels) == shared_counts["surface_init_z_bits"]
    assert len(d5.TRANSFER_SPEC.x_labels) == shared_counts["surface_init_x_bits"]
    assert len(d5.TRANSFER_SPEC.merged_boundary_checks) * common.LATTICE_SURGERY_ROUNDS == (
        shared_counts["interface_bits"]
    )
    assert len(d5.TRANSFER_SPEC.z_labels) * common.LATTICE_SURGERY_ROUNDS == (
        shared_counts["merged_z_bits_and_detectors"]
    )
    assert len(d5.TRANSFER_SPEC.merged_x_labels) * common.LATTICE_SURGERY_ROUNDS == (
        shared_counts["merged_x_bits_and_detectors"]
    )
    assert len(common.STEANE_MERGE_VERIFICATION_LABELS) == (
        shared_counts["steane_verification_bits_and_detectors"]
    )
    assert len(common.HIRANO_STEANE_DATA_ORDER) + len(common.SURFACE_RECOVERY_X_LABELS) == (
        shared_counts["steane_demolition_and_m_ab_bits"]
    )
    assert len(common.FINAL_X_VERIFICATION_CHECKS) == (
        shared_counts["final_x_verification_detectors"]
    )
    assert len(d5.TRANSFER_SPEC.surface_order) == shared_counts["surface_data_readout_bits"]
    assert (
        len(common.PERSISTENT_POSTSELECTED_SURFACE_X_LABELS)
        * (1 + common.LATTICE_SURGERY_ROUNDS + d5.TRANSFER_SPEC.distance)
        == 18
    )

    for patch_sites in (11, 14):
        expected_patch_counts = patch_counts[patch_sites]
        for include_cultivation in (False, True):
            cultivation_count = expected_patch_counts["cultivation"] if include_cultivation else 0
            cultivation_helper_flags = (
                expected_patch_counts["cultivation_helper_flags"] if include_cultivation else 0
            )
            expected_strict_flags = (
                expected_patch_counts["strict_flags_without_cultivation"] + cultivation_helper_flags
            )
            assert expected_strict_flags == (
                expected_patch_counts["encoder"]
                + expected_patch_counts["qed"]
                - 6
                + expected_patch_counts["transfer_flags"]
                + cultivation_helper_flags
            )
            expected_protocol_detectors = (
                expected_patch_counts["protocol_detectors_without_cultivation"] + cultivation_count
            )
            expected_known_zero_detectors = (
                expected_patch_counts["encoder"]
                + expected_patch_counts["qed"]
                + cultivation_count
                + shared_counts["steane_verification_bits_and_detectors"]
                + expected_patch_counts["transfer_flags"]
            )
            assert expected_protocol_detectors == (
                expected_known_zero_detectors
                + shared_counts["interface_change_detectors"]
                + shared_counts["final_x_verification_detectors"]
            )

            transfer_layout = transfer_stim_helper.transfer_measurement_layout(
                d5_stim.TRANSFER_SPEC,
                patch_sites=patch_sites,
                include_cultivation=include_cultivation,
            )
            assert (
                transfer_stim_helper.num_transfer_protocol_postselection_detectors(transfer_layout)
                == expected_protocol_detectors
            )

            for full_post_selection, expected_branch_counts in branch_counts.items():
                expected_surface_detectors = (
                    shared_counts["surface_init_z_detectors"]
                    + shared_counts["surface_init_x_detectors"]
                    + shared_counts["merged_z_bits_and_detectors"]
                    + shared_counts["merged_x_bits_and_detectors"]
                    + expected_branch_counts["continuation_detectors"]
                    + shared_counts["readout_boundary_detectors"]
                )
                expected_detector_total = (
                    expected_patch_counts[
                        (
                            "full_postselection_detectors_without_cultivation"
                            if full_post_selection
                            else "correction_detectors_without_cultivation"
                        )
                    ]
                    + cultivation_count
                )
                assert expected_detector_total == (
                    expected_protocol_detectors + expected_surface_detectors
                )
                if not full_post_selection:
                    assert expected_protocol_detectors + 18 == (
                        expected_patch_counts["correction_acceptance_events_without_cultivation"]
                        + cultivation_count
                    )
                    assert expected_detector_total - expected_protocol_detectors - 18 == 191

                for readout_basis in ("X", "Z"):
                    qiskit_circuit = d5.build_steane_to_rotated_d5_lattice_surgery_circuit(
                        patch_sites=patch_sites,
                        injected_state="0",
                        include_cultivation=include_cultivation,
                        surface_readout_basis=readout_basis,
                        full_post_selection=full_post_selection,
                        add_barriers=False,
                    )
                    stim_circuit = d5_stim.build_steane_to_rotated_d5_lattice_surgery_circuit(
                        patch_sites=patch_sites,
                        injected_state="0",
                        include_cultivation=include_cultivation,
                        surface_readout_basis=readout_basis,
                        full_post_selection=full_post_selection,
                        add_ticks=False,
                    )
                    register_widths = {
                        register.name: len(register) for register in qiskit_circuit.cregs
                    }
                    assert register_widths["enc_flags"] == expected_patch_counts["encoder"]
                    assert register_widths["steane_qed"] == expected_patch_counts["qed"]
                    assert register_widths.get("cult", 0) == cultivation_count
                    assert register_widths.get("steane_bell_partner_flags", 0) == (
                        expected_patch_counts["transfer_flags"]
                    )
                    assert register_widths["surface_init_z"] == (
                        shared_counts["surface_init_z_bits"]
                    )
                    assert register_widths["surface_init_x"] == (
                        shared_counts["surface_init_x_bits"]
                    )
                    assert register_widths["interface_z"] == shared_counts["interface_bits"]
                    assert register_widths["merged_surface_z"] == (
                        shared_counts["merged_z_bits_and_detectors"]
                    )
                    assert register_widths["merged_surface_x"] == (
                        shared_counts["merged_x_bits_and_detectors"]
                    )
                    assert register_widths["steane_verify"] == (
                        shared_counts["steane_verification_bits_and_detectors"]
                    )
                    assert register_widths["steane_x"] + register_widths["surface_x_recovery"] == (
                        shared_counts["steane_demolition_and_m_ab_bits"]
                    )
                    assert (
                        register_widths["post_transfer_surface_z"]
                        + register_widths["post_transfer_surface_x"]
                        == expected_branch_counts["continuation_bits"]
                    )
                    assert register_widths[f"surface_data_{readout_basis.lower()}"] == (
                        shared_counts["surface_data_readout_bits"]
                    )

                    expected_measurement_total = (
                        expected_patch_counts["transfer_bits_without_cultivation"]
                        + cultivation_count
                        + expected_branch_counts["continuation_bits"]
                        + shared_counts["surface_data_readout_bits"]
                    )
                    assert expected_measurement_total == (
                        expected_patch_counts[
                            (
                                "full_postselection_measurements_without_cultivation"
                                if full_post_selection
                                else "correction_measurements_without_cultivation"
                            )
                        ]
                        + cultivation_count
                    )
                    assert qiskit_circuit.num_clbits == expected_measurement_total
                    assert stim_circuit.num_measurements == expected_measurement_total
                    assert stim_circuit.num_detectors == expected_detector_total
