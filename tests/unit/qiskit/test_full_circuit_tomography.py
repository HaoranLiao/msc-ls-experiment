"""Tests for logical Ty tomography after Qiskit state transfer."""

from __future__ import annotations

import math
from typing import cast

import pytest
from qiskit_aer import AerSimulator

from msc.rotated_surface_code.decoding.phenomenological_decoder import (
    SpaceTimeRotatedDetectedErrors,
)
from msc.state_transfer import common
from msc.state_transfer import to_d5_rotated_qiskit as d5
from msc.state_transfer.tomography_qiskit import (
    TOMOGRAPHY_BASES,
    analyze_ty_tomography_counts,
    build_ty_tomography_circuits,
)


@pytest.mark.parametrize("basis", TOMOGRAPHY_BASES)
def test_full_d5_ty_tomography_measurement_record_is_complete_and_semantic(
    basis: common.SurfaceReadoutBasis,
) -> None:
    """Route every d5 full-circuit bit through flags, detector histories, and decoding.

    This record follows Hirano's ``full_post_selection=False`` boundary
    semantics. Product ``|+>`` fixes every surface X check to +1, but Hirano
    still extracts one complete X/Z round before merging: its X results form a
    known-value detector boundary, while its nondeterministic Z results form the
    measured reference. Hirano sets persistent postselection on ``X_bcgh`` and
    ``X_fgkl`` before that extraction and never clears it, so their initial
    events, all three merge-round changes, and all five post-transfer changes
    reject. The final destructive-readout boundary is decoded rather than
    postselected. Five closing rounds likewise match Hirano: one initial closing
    round followed by ``distance - 1`` more. During deformation ``X_ab`` and
    ``X_cd`` are withheld; ``m_ab`` restores only the former, so the first
    post-transfer ``X_cd`` result starts a new reference without a detector.

    This is a semantic, not circuit-identical, comparison. This Qiskit workflow
    physically injects, verifies, and cultivates the 14-site Steane state and
    retains extra Bell-partner/D2 flags instead of using Hirano's ideal Steane
    prefix. Its modular merge serializes interface, surface, and Steane groups
    that Hirano interleaves. The corresponding Stim LER experiment also omits
    Hirano's per-tick idle noise. These deliberate differences do not change the
    false-branch measurement-boundary and postselection contract exercised here.
    """
    circuit = build_ty_tomography_circuits(
        target_distance=5,
        patch_sites=14,
        merge_schedule="hirano",
        full_post_selection=False,
        measure_encoder_flags=True,
        measure_cultivation_flags=True,
        post_injection_qed_checks="full",
        post_transfer_stabilizer_rounds=5,
        add_barriers=False,
    )[basis]
    z_labels = (
        "Z_abfg",
        "Z_cdhi",
        "Z_ghlm",
        "Z_ijno",
        "Z_klpq",
        "Z_mnrs",
        "Z_qrvw",
        "Z_stxy",
        "Z_ej",
        "Z_ot",
        "Z_fk",
        "Z_pu",
    )
    x_labels = (
        "X_ab",
        "X_cd",
        "X_bcgh",
        "X_deij",
        "X_fgkl",
        "X_himn",
        "X_lmqr",
        "X_nost",
        "X_pquv",
        "X_rswx",
        "X_vw",
        "X_xy",
    )
    merged_x_labels = x_labels[2:]
    expected_register_labels = {
        "enc_flags": ("enc_h8", "enc_h9", "enc_h10", "enc_h11", "enc_h12"),
        "steane_qed": (
            "X_0145",
            "Z_0145",
            "X_0235",
            "Z_0235",
            "X_0246",
            "Z_0246",
            "D2_copy_flag",
        ),
        "cult": (
            "root_x",
            "cult_f8",
            "cult_f9",
            "root_x_check",
            "cult_f11",
            "cult_f12",
        ),
        "surface_init_z": z_labels,
        "surface_init_x": x_labels,
        "interface_z": (
            "round1_Z1a",
            "round1_Z35bc",
            "round1_Zde",
            "round2_Z1a",
            "round2_Z35bc",
            "round2_Zde",
            "round3_Z1a",
            "round3_Z35bc",
            "round3_Zde",
        ),
        "merged_surface_z": tuple(
            f"round{round_index}_{label}" for round_index in range(1, 4) for label in z_labels
        ),
        "merged_surface_x": tuple(
            f"round{round_index}_{label}"
            for round_index in range(1, 4)
            for label in merged_x_labels
        ),
        "steane_verify": (
            "round1_Z0145",
            "round1_Z0235",
            "round1_X0235",
            "round1_Z0246",
            "round1_X0246",
            "round2_Z0145",
            "round2_Z0235",
            "round2_Z0246",
        ),
        "steane_x": ("0", "1", "2", "3", "4", "5", "6"),
        "surface_x_recovery": ("X_ab",),
        "steane_bell_partner_flags": (
            "round1_Z0145_bell_partner_flag",
            "round1_D2_copy_flag",
            "round2_Z0145_bell_partner_flag",
            "round2_Z0235_bell_partner_flag",
            "round2_Z0246_bell_partner_flag",
            "round2_D2_copy_flag",
        ),
        "post_transfer_surface_z": tuple(
            f"round{round_index}_{label}" for round_index in range(1, 6) for label in z_labels
        ),
        "post_transfer_surface_x": tuple(
            f"round{round_index}_{label}" for round_index in range(1, 6) for label in x_labels
        ),
        f"surface_data_{basis.lower()}": tuple("abcdefghijklmnopqrstuvwxy"),
    }
    assert tuple(register.name for register in circuit.cregs) == tuple(expected_register_labels)
    assert tuple(len(register) for register in circuit.cregs) == tuple(
        len(labels) for labels in expected_register_labels.values()
    )
    assert circuit.num_clbits == 284
    measured_bits = [
        instruction.clbits[0]
        for instruction in circuit.data
        if instruction.operation.name == "measure"
    ]
    assert len(measured_bits) == circuit.num_clbits
    assert set(measured_bits) == set(circuit.clbits)

    register_bits = {
        name: dict.fromkeys(labels, 0) for name, labels in expected_register_labels.items()
    }
    register_bits["enc_flags"].update(enc_h8=1, enc_h10=1, enc_h11=1)
    register_bits["steane_qed"].update(X_0145=1, Z_0235=1, X_0246=1, D2_copy_flag=1)
    register_bits["cult"].update(root_x=1, cult_f8=1, root_x_check=1, cult_f12=1)
    register_bits["surface_init_x"]["X_xy"] = 1
    register_bits["steane_verify"].update(
        round1_Z0145=1,
        round1_X0235=1,
        round1_X0246=1,
        round2_Z0235=1,
    )
    register_bits["steane_bell_partner_flags"].update(
        round1_D2_copy_flag=1,
        round2_Z0145_bell_partner_flag=1,
        round2_Z0246_bell_partner_flag=1,
    )
    register_bits["steane_x"].update({"0": 1, "2": 1, "4": 1, "6": 1})
    register_bits["surface_x_recovery"]["X_ab"] = 1
    for round_index in range(1, 4):
        register_bits["interface_z"][f"round{round_index}_Z1a"] = 1
        register_bits["merged_surface_z"][f"round{round_index}_Z_abfg"] = 1
        register_bits["merged_surface_x"][f"round{round_index}_X_xy"] = 1
    for round_index in range(1, 6):
        register_bits["post_transfer_surface_z"][f"round{round_index}_Z_abfg"] = 1
        register_bits["post_transfer_surface_x"][f"round{round_index}_X_ab"] = 1
        register_bits["post_transfer_surface_x"][f"round{round_index}_X_xy"] = 1
    register_bits[f"surface_data_{basis.lower()}"]["a"] = 1

    bitstring = " ".join(
        "".join(str(register_bits[name][label]) for label in reversed(labels))
        for name, labels in reversed(expected_register_labels.items())
    )
    measurements = d5.parse_transfer_measurements(
        bitstring,
        patch_sites=14,
        merge_schedule="hirano",
        include_cultivation=True,
        measure_encoder_flags=True,
        measure_cultivation_flags=True,
        surface_readout_basis=basis,
    )

    registers = cast(dict[str, dict[str, object]], measurements["registers"])
    assert set(registers) == set(expected_register_labels)
    assert all(registers[name]["bits"] == register_bits[name] for name in expected_register_labels)
    assert measurements["enc_flags"] == register_bits["enc_flags"]
    assert measurements["post_injection_qed"] == register_bits["steane_qed"]
    assert measurements["cultivation"] == register_bits["cult"]
    assert measurements["steane_verification"] == register_bits["steane_verify"]
    assert measurements["steane_bell_partner_flags"] == register_bits["steane_bell_partner_flags"]
    assert measurements["steane_x"] == register_bits["steane_x"]
    assert measurements["surface_x_recovery"] == {"X_ab": 1}
    assert measurements["interface_z_rounds"] == [
        {"Z1a": 1, "Z35bc": 0, "Zde": 0},
        {"Z1a": 1, "Z35bc": 0, "Zde": 0},
        {"Z1a": 1, "Z35bc": 0, "Zde": 0},
    ]
    assert common.merged_boundary_checks_stable(measurements)
    assert common.merged_boundary_logical_zz_outcome(measurements) == 1
    assert not common.transfer_acceptance_succeeds(measurements)

    zero_z_delta = dict.fromkeys(z_labels, 0)
    zero_merged_x_delta = dict.fromkeys(merged_x_labels, 0)
    assert measurements["merged_surface_syndrome_delta_history"] == [
        {"z": {**zero_z_delta, "Z_abfg": 1}, "x": zero_merged_x_delta},
        {"z": zero_z_delta, "x": zero_merged_x_delta},
        {"z": zero_z_delta, "x": zero_merged_x_delta},
    ]
    merged_recovery = cast(
        SpaceTimeRotatedDetectedErrors,
        measurements["merged_surface_qec_recovery"],
    )
    assert merged_recovery.x_error_sites == {"a"}
    assert merged_recovery.z_error_sites == {"y"}
    assert measurements["merged_surface_logical_x_byproduct"] == 1
    assert measurements["merged_surface_logical_z_byproduct"] == 0

    assert measurements["final_x_verification_checks"] == {
        "X0235": 0,
        "X0246": 0,
        "X0145ab": 1,
    }
    assert measurements["surface_data_readout"] == register_bits[f"surface_data_{basis.lower()}"]
    if basis == "X":
        assert measurements["surface_readout_stabilizers"] == {
            label: int(label == "X_ab") for label in x_labels
        }
        assert measurements["surface_logical_readout"] == {
            "a": 1,
            "f": 0,
            "k": 0,
            "p": 0,
            "u": 0,
        }
        post_transfer_recovery = cast(
            SpaceTimeRotatedDetectedErrors,
            measurements["post_transfer_surface_qec_recovery"],
        )
        assert post_transfer_recovery.x_error_sites == set()
        assert post_transfer_recovery.z_error_sites == {"y"}
    elif basis == "Z":
        assert measurements["surface_readout_stabilizers"] == {
            label: int(label == "Z_abfg") for label in z_labels
        }
        assert measurements["surface_logical_readout"] == {
            "a": 1,
            "b": 0,
            "c": 0,
            "d": 0,
            "e": 0,
        }
        post_transfer_recovery = cast(
            SpaceTimeRotatedDetectedErrors,
            measurements["post_transfer_surface_qec_recovery"],
        )
        assert post_transfer_recovery.is_trivial
    else:
        raise ValueError(f"Unsupported tomography basis: {basis!r}.")
    assert measurements["post_transfer_surface_logical_x_byproduct"] == 0
    assert measurements["post_transfer_surface_logical_z_byproduct"] == 0
    assert common.corrected_surface_logical_readout(measurements, basis=basis) == 1


def test_ty_tomography_builds_full_x_and_z_readout_circuits() -> None:
    """Both experiments include continuation rounds and all-data readout."""
    circuits = build_ty_tomography_circuits(
        target_distance=3,
        patch_sites=14,
        merge_schedule="hirano",
        full_post_selection=False,
        measure_encoder_flags=True,
        measure_cultivation_flags=True,
        post_injection_qed_checks="full",
        post_transfer_stabilizer_rounds=3,
        add_barriers=False,
    )

    assert tuple(circuits) == TOMOGRAPHY_BASES
    for basis, circuit in circuits.items():
        registers = {register.name: register for register in circuit.cregs}
        assert "post_transfer_surface_z" in registers
        assert "post_transfer_surface_x" in registers
        assert len(registers[f"surface_data_{basis.lower()}"]) == 9


@pytest.mark.slow
def test_noiseless_transferred_ty_tomography_has_unit_fidelity() -> None:
    """The paired full circuits recover the cultivated Ty Bloch vector.

    Uses the production ``minimal`` post-injection QED preset so the default
    protocol is what gets fidelity-checked.
    """
    build_and_analysis_params = dict(
        target_distance=3,
        patch_sites=14,
        merge_schedule="minimal_z_only",
        full_post_selection=False,
        persistent_surface_x_post_selection=True,
        measure_encoder_flags=False,
        measure_cultivation_flags=False,
        post_injection_qed_checks="minimal",
        post_transfer_stabilizer_rounds=3,
    )
    circuits = build_ty_tomography_circuits(**build_and_analysis_params, add_barriers=False)
    result = (
        AerSimulator(method="matrix_product_state")
        .run(
            [circuits[basis] for basis in TOMOGRAPHY_BASES],
            shots=128,
            seed_simulator=42,
        )
        .result()
    )
    summary = analyze_ty_tomography_counts(
        {basis: result.get_counts(index) for index, basis in enumerate(TOMOGRAPHY_BASES)},
        **build_and_analysis_params,
    )

    assert summary.accepted_x_shots == 128
    assert summary.accepted_z_shots == 128
    assert summary.expectation_x == pytest.approx(1 / math.sqrt(2), abs=0.25)
    assert summary.expectation_z == pytest.approx(1 / math.sqrt(2), abs=0.25)
    assert summary.fidelity == pytest.approx(1.0, abs=0.18)
