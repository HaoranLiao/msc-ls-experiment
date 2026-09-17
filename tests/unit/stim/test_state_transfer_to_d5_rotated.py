from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from examples.state_transfer.end_to_end_ler.stim_end_to_end_logical_error_rate import (
    build_end_to_end_distillation_circuit,
    estimate_end_to_end_complementary_gap_curve,
)
from examples.state_transfer.lattice_surgery_only_ler.stim_logical_error_rate import (
    LATTICE_SURGERY_EXPERIMENTS,
    LatticeSurgeryOnlyShotAnalyzer,
    build_lattice_surgery_only_circuit,
    estimate_lattice_surgery_logical_error_curves,
)
from msc.noise.fault_injection_stim import add_circuit_depolarizing_noise
from msc.noise.fault_tolerance_stim import assess_fully_postselected_circuit_fault_distance
from msc.state_transfer import common
from msc.state_transfer import helper_stim as helper
from msc.state_transfer import to_d5_rotated_stim as transfer

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
PATCH_SCHEDULE_CASES = (
    (11, "hirano"),
    (11, "minimal_z_only"),
    (14, "hirano"),
    (14, "minimal_z_only"),
)
EXPECTED_POSTSELECTION_DETECTOR_COUNTS = {
    (11, "hirano"): 35,
    (11, "minimal_z_only"): 32,
    (14, "hirano"): 41,
    (14, "minimal_z_only"): 38,
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


@pytest.mark.parametrize("patch_sites", [11, 14])
def test_measurements_follow_hirano_merge_table(patch_sites: int) -> None:
    """Verify the physical Stim measurement record against the merge table.

    Each round measures ``Z1a``, ``Z35bc``, and ``Zde`` first; every surface Z
    check and only the compatible surface X checks second; and the listed
    color-code checks last. The third color-code list is deliberately empty.
    Demolition then measures all seven Steane data qubits in X before measuring
    only ``X_ab`` as the completion result ``m_ab``. ``X_cd`` remains absent
    until the first complete post-transfer surface round.

    The interface path itself is also part of the rule: ``Z35bc`` alternates
    between codes as ``3 -> b -> 5 -> c`` to avoid a damaging hook error.
    """
    spec = transfer.TRANSFER_SPEC
    patch_spec = common.resolve_steane_transfer_patch_spec(patch_sites)
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
    assert spec.withheld_surface_stabilizer_labels == ("X_ab", "X_cd")
    assert spec.merged_x_labels == tuple(
        label for label in spec.x_labels if label not in spec.withheld_surface_stabilizer_labels
    )
    assert common.STEANE_MERGE_VERIFICATION_ROUNDS == expected_color_checks
    assert common.FINAL_X_VERIFICATION_CHECKS == {
        "X0235": ("0", "2", "3", "5"),
        "X0246": ("0", "2", "4", "6"),
        "X0145ab": ("0", "1", "4", "5", "X_ab"),
    }

    # Index every physical Stim measurement by global record position.
    circuit = transfer.build_steane_to_rotated_d5_lattice_surgery_circuit(
        patch_sites=patch_sites,
        injected_state="0",
        include_cultivation=False,
        add_ticks=False,
    )
    layout = helper.transfer_measurement_layout(
        spec,
        patch_sites=patch_sites,
        include_cultivation=False,
    )
    qubits = helper.transfer_qubits(spec, patch_sites=patch_sites)
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
    steane_patch = tuple(qubits[label] for label in patch_spec.site_labels)
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
        bell_partner_flag_positions = (
            tuple(
                position
                for label, position in layout.by_label("steane_bell_partner_flags").items()
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
        assert tuple(measurements[position][1] for position in interface_positions) == tuple(
            qubits[f"zz_{check_name[1:].lower()}"] for check_name, _, _ in expected_interface_checks
        )
        assert tuple(measurements[position][1] for position in surface_positions) == tuple(
            qubits[label.lower()] for label in spec.z_labels + spec.merged_x_labels
        )
        assert {
            label.removeprefix(prefix)
            for label in layout.labels["steane_verify"]
            if label.startswith(prefix)
        } == set(color_checks)
        assert tuple(measurements[position][1] for position in color_positions) == tuple(
            steane_patch[EXPECTED_STEANE_VERIFICATION_READOUT_SITES[patch_sites][check_name]]
            for check_name in color_checks
        )
        if patch_sites == 14:
            expected_flag_sites = (
                (
                    EXPECTED_STEANE_BELL_PARTNER_FLAG_SITES["Z0145"],
                    13,
                )
                if round_index == 0
                else (
                    *(
                        EXPECTED_STEANE_BELL_PARTNER_FLAG_SITES[check_name]
                        for check_name in color_checks
                    ),
                    *((13,) if round_index == 1 else ()),
                )
            )
            assert tuple(
                measurements[position][1] for position in bell_partner_flag_positions
            ) == tuple(steane_patch[site] for site in expected_flag_sites)
            if bell_partner_flag_positions:
                assert bell_partner_flag_positions[0] < color_positions[0]
            if round_index == 0:
                assert max(color_positions) < bell_partner_flag_positions[-1]
        elif patch_sites == 11:
            assert "steane_bell_partner_flags" not in layout.positions
        else:  # pragma: no cover - parametrization uses only supported patches.
            raise ValueError(f"Unsupported patch size: {patch_sites}.")
        previous_round_end = max((*surface_positions, *all_color_positions))

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
        qubits[label.lower()] for label in common.SURFACE_RECOVERY_X_LABELS
    )


# =============================================================================
# Minimal Z-only merge schedule
# =============================================================================


@pytest.mark.parametrize(
    ("probe", "patch_sites", "expected_distance"),
    (
        ("X", 11, 3), ("Z", 11, 3), ("S", 11, 2),
        ("X", 14, 3), ("Z", 14, 3), ("S", 14, 3),
    ),
)
def test_minimal_z_only_with_noisy_cultivation_fault_distance(
    probe: str, patch_sites: int, expected_distance: int,
) -> None:
    """Exhaust physical singles/pairs with the actual cultivated prefix included.

    The 11-site logical-Y path has no Bell-partner flags and exposes malignant
    pairs that the 14-site superdense readouts reject. Its X/Z diagnostic axes
    and all 14-site probes retain distance three.
    """
    if probe == "S":
        noisy, _, _ = build_end_to_end_distillation_circuit(
            .001, patch_sites=patch_sites, merge_schedule="minimal_z_only",
        )
    elif probe in ("X", "Z"):
        circuit = transfer.build_steane_to_rotated_d5_lattice_surgery_circuit(
            patch_sites=patch_sites, merge_schedule="minimal_z_only", include_cultivation=True,
            surface_readout_basis="Z" if probe == "X" else "X", full_post_selection=True,
        )
        receiver = helper.transfer_qubits(transfer.TRANSFER_SPEC, patch_sites=patch_sites)["a"]
        steane_qubits = helper.transfer_qubits(
            transfer.TRANSFER_SPEC, patch_sites=patch_sites,
        )
        steane_data = [steane_qubits[label] for label in common.HIRANO_STEANE_DATA_ORDER]
        split = next(
            i for i, op in enumerate(circuit)
            if op.name == "RX" and receiver in [target.value for target in op.targets_copy()]
        )
        noisy = add_circuit_depolarizing_noise(circuit[:split], p=.001)
        # Rotate only the diagnostic axis; injection/QED/cultivation remain noisy.
        noisy.append("S", steane_data)  # Logical S-dagger maps +Y to +X.
        if probe == "X":
            noisy.append("H", steane_data)
        noisy += add_circuit_depolarizing_noise(circuit[split:], p=.001)
    else:
        raise ValueError(f"Unknown probe: {probe}")
    detectors, logical = noisy.without_noise().compile_detector_sampler(seed=19).sample(
        64, separate_observables=True,
    )
    assert not detectors.any() and not logical.any()
    assessment = assess_fully_postselected_circuit_fault_distance(
        experiment_name=f"{patch_sites}-site cultivated minimal_z_only {probe}", circuit=noisy,
        max_exhaustive_weight=2, find_witness_weight=3,
    )
    assert not assessment.malignant_single_faults
    assert assessment.malignant_pair_count == (12 if expected_distance == 2 else 0)
    assert (assessment.first_malignant_pair is not None) == (expected_distance == 2)
    assert assessment.certified_distance == expected_distance


# =============================================================================
# Logical correctness of the transferred state
# =============================================================================


@pytest.mark.slow
@pytest.mark.parametrize(
    ("patch_sites", "merge_schedule"),
    PATCH_SCHEDULE_CASES,
)
def test_transferred_pauli_expectation_values(
    patch_sites: int, merge_schedule: common.MergeSchedule,
) -> None:
    """Check corrected X/Z observables after five full surface rounds.

    The six Pauli eigenstates exercise deterministic and unbiased X/Z cases. The
    cultivated ``S|+>`` case additionally checks Stim's logical-Y Clifford
    proxy for H_XY cultivation. The circuit observable includes the transfer
    byproducts and is measured only after the default five-round Hirano
    continuation and full-data readout.
    """
    experiments = (
        *(
            (state, False, basis, expected)
            for state, expectations in LOGICAL_EXPECTATIONS.items()
            for basis, expected in expectations.items()
            if basis in ("X", "Z")
        ),
        *(
            ("S", True, basis, expected)
            for basis, expected in LOGICAL_EXPECTATIONS["i"].items()
            if basis in ("X", "Z")
        ),
    )
    for state, include_cultivation, basis, expected in experiments:
        circuit = helper.build_lattice_surgery_circuit(
            transfer.TRANSFER_SPEC,
            patch_sites=patch_sites,
            merge_schedule=merge_schedule,
            injected_state=state,
            include_cultivation=include_cultivation,
            add_ticks=False,
        )
        transfer_layout = helper.transfer_measurement_layout(
            transfer.TRANSFER_SPEC,
            patch_sites=patch_sites,
            merge_schedule=merge_schedule,
            include_cultivation=include_cultivation,
        )
        layout = transfer.append_post_transfer_and_readout(
            circuit,
            transfer_layout=transfer_layout,
            readout_basis=basis,
            full_post_selection=False,
            add_ticks=False,
        )
        shots = circuit.compile_sampler(seed=42).sample(256)
        unique_shots, multiplicities = np.unique(shots, axis=0, return_counts=True)
        logical_counts: Counter[int] = Counter()
        for shot, multiplicity in zip(unique_shots, multiplicities, strict=True):
            measurements = common.interpret_transfer_measurements(
                layout.extract(shot),
                spec=transfer.TRANSFER_SPEC,
                include_cultivation=include_cultivation,
                post_transfer_stabilizer_rounds=transfer.TRANSFER_SPEC.distance,
                surface_readout_basis=basis,
            )
            if common.transfer_acceptance_succeeds(measurements):
                logical_counts[
                    common.corrected_surface_logical_readout(measurements, basis=basis)
                ] += int(multiplicity)
        accepted = sum(logical_counts.values())
        assert accepted == 256
        expectation = (logical_counts[0] - logical_counts[1]) / accepted
        assert expectation == pytest.approx(expected, abs=0.2)


# =============================================================================
# Detector policy and physical fault distance
# =============================================================================


@pytest.mark.parametrize("experiment", LATTICE_SURGERY_EXPERIMENTS)
@pytest.mark.parametrize(("patch_sites", "merge_schedule"), PATCH_SCHEDULE_CASES)
def test_postselected_color_interface_detector_policy(
    experiment,
    patch_sites: int,
    merge_schedule: common.MergeSchedule,
) -> None:
    """Verify the selective detector mask across the complete noisy circuit.

    Protocol checks and the persistent ``X_bcgh``/``X_fgkl`` streams reject;
    other surface detectors are decoded. The 14-site Bell-partner flags retain
    visibility of an otherwise-hidden X-readout path.
    """
    circuit, layout = build_lattice_surgery_only_circuit(
        1e-3,
        experiment=experiment,
        patch_sites=patch_sites,
        merge_schedule=merge_schedule,
    )
    analyzer = LatticeSurgeryOnlyShotAnalyzer(
        circuit,
        layout=layout,
        full_post_selection=False,
    )
    protocol_postselection_count = helper.num_transfer_protocol_postselection_detectors(layout)
    ideal_input_detector_count = sum(
        len(layout.positions[family]) for family in ("enc_flags", "steane_qed")
    )

    # Count the noisy protocol detectors preceding the surface history.
    steane_postselection_families = ["enc_flags", "steane_qed", "steane_verify"]
    if patch_sites == 14:
        steane_postselection_families.append("steane_bell_partner_flags")
    expected_protocol_postselection_count = (
        sum(len(layout.positions[family]) for family in steane_postselection_families)
        + (common.LATTICE_SURGERY_ROUNDS - 1) * len(transfer.TRANSFER_SPEC.merged_boundary_checks)
        + len(common.FINAL_X_VERIFICATION_CHECKS)
    )
    assert (
        protocol_postselection_count
        == expected_protocol_postselection_count
        < circuit.num_detectors
    )

    # Resolve detector ranges in circuit order.
    initial_x_detector_by_label = {
        label: protocol_postselection_count + offset
        for offset, label in enumerate(layout.labels["surface_init_x"])
    }
    merged_surface_detector_start = protocol_postselection_count + len(
        layout.positions["surface_init_x"]
    )
    merged_x_detector_start = merged_surface_detector_start + len(
        layout.positions["merged_surface_z"]
    )
    merged_x_detector_offset_by_label = {
        label: offset for offset, label in enumerate(transfer.TRANSFER_SPEC.merged_x_labels)
    }
    merged_x_round_width = len(transfer.TRANSFER_SPEC.merged_x_labels)
    post_transfer_z_detector_start = merged_x_detector_start + len(
        layout.positions["merged_surface_x"]
    )
    post_transfer_x_detector_start = post_transfer_z_detector_start + len(
        layout.positions[transfer.POST_TRANSFER_SURFACE_Z_FAMILY]
    )

    # Select both persistent X streams at initialization and throughout time.
    persistent_x_detector_indices = [
        initial_x_detector_by_label[label]
        for label in transfer.HIRANO_POSTSELECTED_SURFACE_X_LABELS
    ]
    persistent_x_detector_indices.extend(
        merged_x_detector_start
        + round_index * merged_x_round_width
        + merged_x_detector_offset_by_label[label]
        for round_index in range(common.LATTICE_SURGERY_ROUNDS)
        for label in transfer.HIRANO_POSTSELECTED_SURFACE_X_LABELS
    )
    post_transfer_rounds = layout.rounds(
        transfer.POST_TRANSFER_SURFACE_X_FAMILY,
        transfer.TRANSFER_SPEC.distance,
    )
    emitted_post_transfer_x_detector_labels = tuple(
        label
        for label in layout.labels[transfer.POST_TRANSFER_SURFACE_X_FAMILY]
        if label != "round1_X_cd"
    )
    post_transfer_x_detector_offset_by_label = {
        label: offset for offset, label in enumerate(emitted_post_transfer_x_detector_labels)
    }
    persistent_x_detector_indices.extend(
        post_transfer_x_detector_start
        + post_transfer_x_detector_offset_by_label[f"round{round_index + 1}_{label}"]
        for round_index in range(len(post_transfer_rounds))
        for label in transfer.HIRANO_POSTSELECTED_SURFACE_X_LABELS
    )
    readout_detector_start = post_transfer_x_detector_start + len(
        emitted_post_transfer_x_detector_labels
    )
    compatible_readout_detector_count = len(
        transfer.TRANSFER_SPEC.x_labels
        if experiment.readout_basis == "X"
        else transfer.TRANSFER_SPEC.z_labels
    )
    assert readout_detector_start + compatible_readout_detector_count == circuit.num_detectors
    assert layout.labels[f"surface_data_{experiment.readout_basis.lower()}"] == (
        transfer.TRANSFER_SPEC.surface_order
    )

    expected_postselection_detector_indices = {
        *range(ideal_input_detector_count, protocol_postselection_count),
        *persistent_x_detector_indices,
    }
    assert set(analyzer._postselection_detector_indices) == (
        expected_postselection_detector_indices
    )
    assert {
        initial_x_detector_by_label["X_ab"],
        merged_x_detector_start + merged_x_detector_offset_by_label["X_deij"],
        post_transfer_z_detector_start,
        readout_detector_start,
    }.isdisjoint(expected_postselection_detector_indices)
    expected_detector_count = EXPECTED_POSTSELECTION_DETECTOR_COUNTS[
        (patch_sites, merge_schedule)
    ]
    assert len(analyzer._postselection_detector_indices) == expected_detector_count
    assert circuit.num_detectors - ideal_input_detector_count == expected_detector_count + 191


@pytest.mark.parametrize("experiment", LATTICE_SURGERY_EXPERIMENTS)
@pytest.mark.parametrize(
    ("patch_sites", "merge_schedule"),
    PATCH_SCHEDULE_CASES,
)
def test_fully_postselected_lattice_surgery_only_has_fault_distance_three(
    experiment,
    patch_sites: int,
    merge_schedule: common.MergeSchedule,
) -> None:
    """Certify stage-only physical fault distance three under full postselection.

    ``build_lattice_surgery_only_circuit`` keeps injection, QED, and cultivation
    ideal, so this audit includes fault locations only from surface preparation
    through lattice surgery, post-transfer rounds, and readout. It therefore
    does not certify the fully noisy cultivated prefix exercised separately by
    ``test_minimal_z_only_with_noisy_cultivation_fault_distance``.

    The direct-parity audit applies only when every detector rejects. It does not
    certify the decoded ``full_post_selection=False`` branch, which requires a
    decoder-aware combination search; changing this flag alone would be insufficient.
    """
    circuit, _ = build_lattice_surgery_only_circuit(
        1e-3,
        experiment=experiment,
        patch_sites=patch_sites,
        merge_schedule=merge_schedule,
        full_post_selection=True,
    )
    assessment = assess_fully_postselected_circuit_fault_distance(
        experiment_name=f"{patch_sites}-site {experiment.name}",
        circuit=circuit,
        max_exhaustive_weight=2,
        find_witness_weight=3,
    )

    assert not assessment.malignant_single_faults
    assert assessment.malignant_pair_count == 0
    assert assessment.first_malignant_pair is None
    assert assessment.first_malignant_triple is not None
    assert assessment.certified_distance == 3

    witness = assessment.first_malignant_triple
    assert witness.weight == 3
    assert len({fault.location_key for fault in witness.faults}) == 3
    assert witness.combined_detector_mask == 0
    assert witness.combined_observable_mask == 1


# =============================================================================
# Post-transfer continuation
# =============================================================================


@pytest.mark.parametrize(
    ("full_post_selection", "expected_rounds"),
    ((False, transfer.TRANSFER_SPEC.distance), (True, 1)),
)
def test_post_transfer_round_count_matches_hirano_branch(
    full_post_selection: bool,
    expected_rounds: int,
) -> None:
    """Use five correction rounds for false and one rejection round for true."""
    experiment = LATTICE_SURGERY_EXPERIMENTS[1]
    circuit, layout = build_lattice_surgery_only_circuit(
        1e-3,
        experiment=experiment,
        patch_sites=14,
        full_post_selection=full_post_selection,
    )
    analyzer = LatticeSurgeryOnlyShotAnalyzer(
        circuit,
        layout=layout,
        full_post_selection=full_post_selection,
    )

    # Hirano performs one X_ab-only extraction at lattice-surgery completion:
    # m_ab completes X0145ab and seeds the restored X_ab stream. There is no
    # subsequent X_ab/X_cd recovery round; X_cd first occurs in full round 1.
    assert layout.labels["surface_x_recovery"] == ("X_ab",)
    assert "X_cd" not in layout.by_label("surface_x_recovery")
    assert max(layout.positions["surface_x_recovery"]) + 1 == min(
        layout.positions[transfer.POST_TRANSFER_SURFACE_Z_FAMILY]
    )
    assert "round1_X_cd" in layout.by_label(transfer.POST_TRANSFER_SURFACE_X_FAMILY)

    qubits = helper.transfer_qubits(transfer.TRANSFER_SPEC, patch_sites=14)
    instructions = tuple(circuit)
    steane_demolition_index = next(
        index
        for index, instruction in enumerate(instructions)
        if instruction.name == "MX"
        and tuple(target.qubit_value for target in instruction.targets_copy())
        == tuple(qubits[label] for label in common.HIRANO_STEANE_DATA_ORDER)
    )
    mab_reset_index = next(
        index
        for index, instruction in enumerate(
            instructions[steane_demolition_index + 1 :],
            start=steane_demolition_index + 1,
        )
        if instruction.name == "RX"
        and tuple(target.qubit_value for target in instruction.targets_copy()) == (qubits["x_ab"],)
    )
    mab_measurement_index = next(
        index
        for index, instruction in enumerate(
            instructions[mab_reset_index + 1 :],
            start=mab_reset_index + 1,
        )
        if instruction.name == "MX"
        and tuple(target.qubit_value for target in instruction.targets_copy()) == (qubits["x_ab"],)
    )
    mab_couplings = tuple(
        tuple(target.qubit_value for target in instruction.targets_copy())
        for instruction in instructions[mab_reset_index + 1 : mab_measurement_index]
        if instruction.name == "CX"
    )
    assert mab_couplings == (
        (qubits["x_ab"], qubits["a"]),
        (qubits["x_ab"], qubits["b"]),
    )
    assert len(layout.positions[transfer.POST_TRANSFER_SURFACE_Z_FAMILY]) == (
        expected_rounds * len(transfer.TRANSFER_SPEC.z_labels)
    )
    assert len(layout.positions[transfer.POST_TRANSFER_SURFACE_X_FAMILY]) == (
        expected_rounds * len(transfer.TRANSFER_SPEC.x_labels)
    )
    emitted_post_transfer_x_detector_labels = tuple(
        label
        for label in layout.labels[transfer.POST_TRANSFER_SURFACE_X_FAMILY]
        if label != "round1_X_cd"
    )
    assert "round1_X_cd" not in emitted_post_transfer_x_detector_labels
    assert len(emitted_post_transfer_x_detector_labels) == (
        expected_rounds * len(transfer.TRANSFER_SPEC.x_labels) - 1
    )
    if expected_rounds > 1:
        assert "round2_X_cd" in emitted_post_transfer_x_detector_labels
    assert len(layout.positions["surface_data_z"]) == len(transfer.TRANSFER_SPEC.surface_order)
    if full_post_selection:
        assert len(analyzer._postselection_detector_indices) == circuit.num_detectors
