from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from examples.state_transfer.stim_logical_error_rate_ls import (
    LATTICE_SURGERY_EXPERIMENTS,
    LatticeSurgeryOnlyShotAnalyzer,
    build_lattice_surgery_only_circuit,
)
from msc.state_transfer import common
from msc.state_transfer import helper_stim as helper
from msc.state_transfer import to_d5_rotated_stim as transfer
from msc.steane_code.state_injection_stim import PATCH_11_SITE_STABILIZER_ANCILLA_SITES

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


def test_measurements_follow_hirano_merge_table() -> None:
    """Verify the physical Stim measurement record against the merge table.

    Each round measures ``Z1a``, ``Z35bc``, and ``Zde`` first; every surface Z
    check and only the compatible surface X checks second; and the listed
    color-code checks last. The third color-code list is deliberately empty.
    Demolition then measures all seven Steane data qubits in X before restoring
    ``X_ab`` and ``X_cd``.

    The interface path itself is also part of the rule: ``Z35bc`` alternates
    between codes as ``3 -> b -> 5 -> c`` to avoid a damaging hook error.
    """
    spec = transfer.TRANSFER_SPEC
    expected_interface_checks = (
        ("Z1a", ("1",), ("a",)),
        ("Z35bc", ("3", "5"), ("b", "c")),
        ("Zde", (), ("d", "e")),
    )
    expected_color_checks = (
        ("Z0145", "Z0235", "X0235", "Z0246", "X0246"),
        ("Z0145", "Z0235", "Z0246"),
        (),
    )

    # Check the d=5 geometry and the exact table-defined measurement sets.
    assert spec.merged_boundary_checks == expected_interface_checks
    assert spec.top_x_recovery_labels == ("X_ab", "X_cd")
    assert spec.merged_x_labels == tuple(
        label for label in spec.x_labels if label not in spec.top_x_recovery_labels
    )
    assert common.STEANE_MERGE_VERIFICATION_ROUNDS == expected_color_checks
    assert common.FINAL_X_VERIFICATION_CHECKS == {
        "X0235": ("0", "2", "3", "5"),
        "X0246": ("0", "2", "4", "6"),
        "X0145ab": ("0", "1", "4", "5", "X_ab"),
    }

    # Index every physical Stim measurement by global record position.
    circuit = transfer.build_steane_to_rotated_d5_lattice_surgery_circuit(
        injected_state="0",
        include_cultivation=False,
        add_ticks=False,
    )
    layout = helper.transfer_measurement_layout(
        spec,
        include_cultivation=False,
        surface_readout_basis=None,
    )
    qubits = helper.transfer_qubits(spec)
    measurements = tuple(
        (instruction.name, target.qubit_value)
        for instruction in circuit
        if instruction.name in {"M", "MX", "MY"}
        for target in instruction.targets_copy()
    )
    assert len(measurements) == circuit.num_measurements == layout.num_measurements

    # Verify interface -> surface -> color-code ordering in all three rounds.
    assert tuple(
        measurements[position][1] for position in layout.positions["surface_init_x"]
    ) == tuple(qubits[label.lower()] for label in spec.x_labels)
    previous_round_end = max(layout.positions["surface_init_x"])
    steane_patch = tuple(
        qubits[common.HIRANO_PATCH_LABEL_BY_GOOGLE_SITE[site]] for site in range(11)
    )
    for round_index, color_checks in enumerate(expected_color_checks):
        interface_positions = layout.rounds("interface_z")[round_index]
        surface_positions = (
            layout.rounds("merged_surface_z")[round_index]
            + layout.rounds("merged_surface_x")[round_index]
        )
        prefix = f"round{round_index + 1}_"
        color_positions = tuple(
            layout.by_label("steane_verify")[prefix + check_name] for check_name in color_checks
        )

        assert previous_round_end < min(interface_positions)
        assert max(interface_positions) < min(surface_positions)
        if color_positions:
            assert max(surface_positions) < min(color_positions)
        assert tuple(measurements[position][1] for position in interface_positions) == tuple(
            qubits[f"zz_{check_name[1:].lower()}"] for check_name, _, _ in expected_interface_checks
        )
        assert tuple(measurements[position][1] for position in surface_positions) == tuple(
            qubits[label.lower()] for label in spec.z_labels + spec.merged_x_labels
        )
        assert (
            tuple(
                label.removeprefix(prefix)
                for label in layout.labels["steane_verify"]
                if label.startswith(prefix)
            )
            == color_checks
        )
        assert tuple(measurements[position][1] for position in color_positions) == tuple(
            steane_patch[
                PATCH_11_SITE_STABILIZER_ANCILLA_SITES[
                    common.STEANE_MERGE_VERIFICATION_STABILIZER_INDICES[check_name]
                ]
            ]
            for check_name in color_checks
        )
        previous_round_end = max((*surface_positions, *color_positions))

    # Verify the alternating mixed-code path used to suppress Z35bc hooks.
    z35bc_ancilla = qubits["zz_35bc"]
    assert (
        tuple(
            targets[index].qubit_value
            for instruction in circuit
            if instruction.name == "CX"
            for targets in (instruction.targets_copy(),)
            for index in range(0, len(targets), 2)
            if targets[index + 1].qubit_value == z35bc_ancilla
        )
        == tuple(qubits[label] for label in ("3", "b", "5", "c")) * common.LATTICE_SURGERY_ROUNDS
    )

    # Verify X-basis Steane demolition precedes top-edge X-check recovery.
    demolition_positions = layout.positions["steane_x"]
    recovery_positions = layout.positions["surface_x_recovery"]
    assert previous_round_end < min(demolition_positions)
    assert max(demolition_positions) < min(recovery_positions)
    assert tuple(measurements[position] for position in demolition_positions) == tuple(
        ("MX", qubits[label]) for label in common.HIRANO_STEANE_DATA_ORDER
    )
    assert tuple(measurements[position][1] for position in recovery_positions) == tuple(
        qubits[label.lower()] for label in spec.top_x_recovery_labels
    )


# =============================================================================
# Logical correctness of the transferred state
# =============================================================================


def test_transferred_pauli_expectation_values() -> None:
    """Check all X/Y/Z expectations after transfer-frame correction.

    The six Pauli eigenstates exercise both signs on every Bloch axis. The
    cultivated ``S|+>`` case additionally checks Stim's logical-Y Clifford
    proxy for H_XY cultivation. Every expectation is computed only from shots
    accepted by the color/interface postselection rules and after applying the
    spacetime-decoder and teleportation Pauli frame.
    """
    experiments = (
        *(
            (state, False, basis, expected)
            for state, expectations in LOGICAL_EXPECTATIONS.items()
            for basis, expected in expectations.items()
        ),
        *(("S", True, basis, expected) for basis, expected in LOGICAL_EXPECTATIONS["i"].items()),
    )
    for state, include_cultivation, basis, expected in experiments:
        circuit = transfer.build_steane_to_rotated_d5_lattice_surgery_circuit(
            injected_state=state,
            include_cultivation=include_cultivation,
            surface_readout_basis=basis,
            add_ticks=False,
        )
        logical_counts: Counter[int] = Counter()
        for shot in circuit.compile_sampler(seed=42).sample(256):
            measurements = transfer.parse_transfer_measurements(
                shot,
                include_cultivation=include_cultivation,
                surface_readout_basis=basis,
            )
            if common.transfer_acceptance_succeeds(measurements):
                logical_counts[
                    common.corrected_surface_logical_readout(measurements, basis=basis)
                ] += 1
        accepted = sum(logical_counts.values())
        assert accepted == 256
        expectation = (logical_counts[0] - logical_counts[1]) / accepted
        assert expectation == pytest.approx(expected, abs=0.2)


# =============================================================================
# Detector policy and graphlike distance
# =============================================================================


@pytest.mark.parametrize("experiment", LATTICE_SURGERY_EXPERIMENTS)
def test_postselected_color_interface_and_decoded_surface_distance_three(experiment) -> None:
    """Verify closing postselection, noisy-surface decoding, and distance three.

    Encoder, color-code, repeated-interface, and demolition checks form one
    leading postselection block. The pre-merge surface-code X-detector layer
    follows: ordinary surface-X events are decoded, while Hirano's two selected
    surface-X streams reject. The ideal closing-round changes are postselected,
    and the ideal logical-support readout adds no redundant boundary detectors.
    Both probes must require three graphlike faults.
    """
    circuit, layout = build_lattice_surgery_only_circuit(1e-3, experiment=experiment)
    protocol_postselection_count = helper.num_transfer_protocol_postselection_detectors(layout)
    assert layout.labels[f"surface_logical_{experiment.readout_basis.lower()}"] == (
        transfer.TRANSFER_SPEC.logical_x
        if experiment.readout_basis == "X"
        else transfer.TRANSFER_SPEC.logical_z
    )

    # steane_verify contains the five round-1 and three round-2 color-code
    # checks; round 3 has no additional color-code verification measurements.
    expected_protocol_postselection_count = (
        sum(
            len(layout.positions[family]) for family in ("enc_flags", "steane_qed", "steane_verify")
        )
        + (common.LATTICE_SURGERY_ROUNDS - 1) * len(transfer.TRANSFER_SPEC.merged_boundary_checks)
        + len(common.FINAL_X_VERIFICATION_CHECKS)
    )
    assert (
        protocol_postselection_count
        == expected_protocol_postselection_count
        < circuit.num_detectors
    )

    noisy_surface_history_detector_count = len(layout.positions["surface_init_x"]) + sum(
        len(round_positions)
        for family in ("merged_surface_z", "merged_surface_x")
        for round_positions in layout.rounds(family)
    )
    closing_detector_start = protocol_postselection_count + noisy_surface_history_detector_count
    closing_detector_count = sum(
        len(layout.positions[family])
        for family in (
            transfer.IDEAL_POST_TRANSFER_SURFACE_Z_FAMILY,
            transfer.IDEAL_POST_TRANSFER_SURFACE_X_FAMILY,
        )
    )
    assert closing_detector_start + closing_detector_count == circuit.num_detectors

    # Detector order is protocol prefix -> initial surface X -> all merged
    # surface Z -> all merged surface X -> ideal closing round.
    # Map the initial surface-X labels to absolute detector indices.
    initial_x_detector_by_label = {
        label: protocol_postselection_count + offset
        for offset, label in enumerate(layout.labels["surface_init_x"])
    }
    merged_x_detector_start = (
        protocol_postselection_count
        + len(layout.positions["surface_init_x"])
        + len(layout.positions["merged_surface_z"])
    )
    # Offsets within each merge round's surface-X detector block.
    merged_x_detector_offset_by_label = {
        label: offset for offset, label in enumerate(transfer.TRANSFER_SPEC.merged_x_labels)
    }
    merged_x_round_width = len(transfer.TRANSFER_SPEC.merged_x_labels)
    # All eight surface-X detectors postselected by Hirano's persistent flags:
    # two initialization detectors plus the same two streams in three rounds.
    postselected_surface_x_detector_indices = [
        initial_x_detector_by_label[label]
        for label in transfer.HIRANO_POSTSELECTED_SURFACE_X_LABELS
    ]
    postselected_surface_x_detector_indices.extend(
        merged_x_detector_start
        + round_index * merged_x_round_width
        + merged_x_detector_offset_by_label[label]
        for round_index in range(common.LATTICE_SURGERY_ROUNDS)
        for label in transfer.HIRANO_POSTSELECTED_SURFACE_X_LABELS
    )
    # One-hot test cases: protocol reject, decoded initialization event, all
    # persistent-X rejects, decoded merge event, then closing-round reject.
    one_hot_detector_indices = [
        0,
        initial_x_detector_by_label["X_ab"],
        *postselected_surface_x_detector_indices,
        merged_x_detector_start + merged_x_detector_offset_by_label["X_deij"],
        closing_detector_start,
    ]
    detection_events = np.zeros((len(one_hot_detector_indices), circuit.num_detectors), dtype=bool)
    detection_events[np.arange(len(one_hot_detector_indices)), one_hot_detector_indices] = True
    accepted, logical_errors = LatticeSurgeryOnlyShotAnalyzer(
        circuit,
        layout=layout,
        full_post_selection=False,
    ).analyze(
        detection_events,
        np.zeros((len(one_hot_detector_indices), 1), dtype=bool),
    )
    assert accepted.tolist() == [
        False,
        True,
        *([False] * len(postselected_surface_x_detector_indices)),
        True,
        False,
    ]
    assert not logical_errors.any()
    assert len(circuit.shortest_graphlike_error(canonicalize_circuit_errors=True)) == 3
