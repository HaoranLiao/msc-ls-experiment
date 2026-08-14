from __future__ import annotations

import math
from collections import Counter

import pytest
from qiskit_aer import AerSimulator

from msc.state_transfer import common
from msc.state_transfer import to_d3_rotated_qiskit as d3
from msc.state_transfer import to_d5_rotated_qiskit as d5
from msc.steane_code.state_injection_qiskit import PATCH_STABILIZER_ANCILLA_SITES

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
EXPECTED_TOP_X_CHECKS = {3: ("X_ab",), 5: ("X_ab", "X_cd")}

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


@pytest.mark.parametrize(("module", "build"), TARGETS)
def test_measurements_follow_hirano_merge_table(module, build) -> None:
    """Verify the built circuit against every row of the merge table.

    Each of the three merge rounds must measure interface checks first, all
    surface Z checks plus only the compatible surface X checks second, and the
    listed color-code checks last. The third color-code list is deliberately
    empty. Demolition then measures every Steane data qubit in X before the
    withheld top-edge surface X checks are restored.

    The d=3 target is the compact projection of the same rules: it omits
    ``Zde`` and restores only ``X_ab``. The mixed ``Z35bc`` check must retain
    the hook-safe ``3 -> b -> 5 -> c`` CNOT order at either distance.
    """
    spec = module.TRANSFER_SPEC
    expected_interface_checks = EXPECTED_INTERFACE_CHECKS[spec.distance]
    expected_top_x_checks = EXPECTED_TOP_X_CHECKS[spec.distance]
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
    assert spec.top_x_recovery_labels == expected_top_x_checks
    assert spec.merged_x_labels == tuple(
        label for label in spec.x_labels if label not in expected_top_x_checks
    )
    assert common.STEANE_MERGE_VERIFICATION_ROUNDS == expected_color_checks
    assert common.FINAL_X_VERIFICATION_CHECKS == expected_final_x_checks

    # Index every physical measurement by semantic family and label.
    circuit = build(injected_state="0", include_cultivation=False, add_barriers=False)
    labels = common.transfer_measurement_labels(
        spec,
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
    previous_round_end = max(positions["surface_init_z"].values())
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

        assert previous_round_end < min(interface_positions)
        assert max(interface_positions) < min(surface_positions)
        if color_positions:
            assert max(surface_positions) < min(color_positions)
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
            common.HIRANO_PATCH_LABEL_BY_GOOGLE_SITE[
                PATCH_STABILIZER_ANCILLA_SITES[
                    common.STEANE_MERGE_VERIFICATION_STABILIZER_INDICES[check_name]
                ]
            ]
            for check_name in color_checks
        )
        previous_round_end = max((*surface_positions, *color_positions))

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
    ) == tuple(label.lower() for label in expected_top_x_checks)
    for position in demolition_positions:
        measured_qubit = circuit.data[position].qubits[0]
        previous_operation = next(
            instruction.operation.name
            for instruction in reversed(circuit.data[:position])
            if measured_qubit in instruction.qubits
        )
        assert previous_operation == "h"


# =============================================================================
# Logical correctness of the transferred state
# =============================================================================


@pytest.mark.parametrize(("module", "build"), TARGETS)
def test_transferred_pauli_expectation_values_without_cultivation(module, build) -> None:
    """Check all X/Y/Z expectations of all six Pauli eigenstates.

    Raw surface readout is interpreted only after the teleportation and
    spacetime-decoder byproducts have been folded into the logical Pauli frame.
    Thus this checks the transferred logical state, not merely its uncorrected
    destructive measurement parity.
    """
    experiments = tuple(
        (state, basis, expected)
        for state, expectations in LOGICAL_EXPECTATIONS.items()
        for basis, expected in expectations.items()
    )
    circuits = [
        build(
            injected_state=state,
            include_cultivation=False,
            surface_readout_basis=basis,
            add_barriers=False,
        )
        for state, basis, _ in experiments
    ]
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
                include_cultivation=False,
                surface_readout_basis=basis,
            )
            if common.transfer_acceptance_succeeds(measurements):
                logical_counts[
                    common.corrected_surface_logical_readout(measurements, basis=basis)
                ] += count
        accepted = sum(logical_counts.values())
        assert accepted >= 200, f"{state}/{basis} accepted only {accepted} shots"
        expectation = (logical_counts[0] - logical_counts[1]) / accepted
        assert expectation == pytest.approx(expected, abs=0.2)


def test_cultivated_t_state_pauli_expectation_values() -> None:
    """Check the non-Clifford T-state Bloch vector after cultivation and transfer.

    This keeps the default Qiskit path covered in addition to the six exact
    Pauli eigenstates above: accepted H_XY cultivation must preserve the
    ``(+X +Y) / sqrt(2)`` logical state through d=3 lattice surgery.
    """
    expected_by_basis = {"X": 1 / math.sqrt(2), "Y": 1 / math.sqrt(2), "Z": 0.0}
    circuits = [
        d3.build_steane_to_rotated_d3_lattice_surgery_circuit(
            injected_state="T",
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
            shots=128,
            seed_simulator=42,
        )
        .result()
    )

    for basis, counts in zip(expected_by_basis, result.get_counts(), strict=True):
        logical_counts: Counter[int] = Counter()
        for bitstring, count in counts.items():
            measurements = d3.parse_transfer_measurements(
                bitstring,
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
