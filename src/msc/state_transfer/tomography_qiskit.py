"""Logical X/Z tomography after Qiskit Steane-to-surface state transfer."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import yaml
from qiskit import QuantumCircuit

from msc.rotated_surface_code.decoding.circuit_level_decoder import CircuitLevelDecoder
from msc.state_transfer import common
from msc.state_transfer import to_d3_rotated_qiskit as d3
from msc.state_transfer import to_d5_rotated_qiskit as d5
from msc.steane_code.state_injection_common import (
    PostInjectionQedChecks,
    resolve_post_injection_qed_checks,
)
from msc.utils.normalize import qiskit_bit_value

if TYPE_CHECKING:
    from msc.experiment.results_io import DeviceCalibration

TomographyTargetDistance = Literal[3, 5]
TOMOGRAPHY_BASES: tuple[common.SurfaceReadoutBasis, ...] = ("X", "Z")
TY_TOMOGRAPHY_CONFIG_PATH = Path(__file__).with_name("tomography_config.yaml")


@dataclass(frozen=True)
class TyTomographyConfig:
    """Validated settings loaded from a full-circuit tomography YAML file.

    They fix construction and the record layout. Analysis-only settings, such
    as persistent surface-X postselection and the complementary-gap threshold,
    are arguments of :func:`analyze_ty_tomography_counts` instead.
    """

    target_distance: TomographyTargetDistance
    patch_sites: common.SteanePatchSites
    merge_schedule: common.MergeSchedule
    full_post_selection: bool
    measure_encoder_flags: bool
    measure_cultivation_flags: bool
    post_injection_qed_checks: PostInjectionQedChecks
    post_transfer_stabilizer_rounds: int | None
    add_barriers: bool

    def __post_init__(self) -> None:
        if type(self.target_distance) is not int or self.target_distance not in (3, 5):
            raise ValueError("target_distance must be 3 or 5.")
        if type(self.patch_sites) is not int or self.patch_sites not in (11, 14):
            raise ValueError("patch_sites must be 11 or 14.")
        if self.merge_schedule not in ("hirano", "minimal_z_only"):
            raise ValueError("merge_schedule must be 'hirano' or 'minimal_z_only'.")
        if self.post_injection_qed_checks not in ("minimal", "full"):
            raise ValueError("post_injection_qed_checks must be 'minimal' or 'full'.")
        for name in (
            "full_post_selection",
            "measure_encoder_flags",
            "measure_cultivation_flags",
            "add_barriers",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be true or false.")
        if self.post_transfer_stabilizer_rounds is not None and (
            type(self.post_transfer_stabilizer_rounds) is not int
            or self.post_transfer_stabilizer_rounds < 1
        ):
            raise ValueError("post_transfer_stabilizer_rounds must be null or a positive integer.")

    @property
    def resolved_post_transfer_stabilizer_rounds(self) -> int:
        """Use the historical branch default unless the YAML gives a count."""
        if self.post_transfer_stabilizer_rounds is not None:
            return self.post_transfer_stabilizer_rounds
        return 1 if self.full_post_selection else self.target_distance

    @property
    def resolved_post_injection_qed_checks(self) -> tuple[str, ...] | None:
        """Concrete post-injection QED check names for the configured preset."""
        return resolve_post_injection_qed_checks(self.post_injection_qed_checks)


@dataclass(frozen=True)
class TyTomographySummary:
    """Postselected logical Bloch components and fidelity with ``|Ty>``."""

    expectation_x: float
    expectation_z: float
    fidelity: float
    accepted_x_shots: int
    accepted_z_shots: int
    total_x_shots: int
    total_z_shots: int


@dataclass(frozen=True)
class CircuitLevelTomographyVariant:
    """Inputs needed to decode one submitted circuit variant."""

    counts: Mapping[str, int]
    submitted_circuit: QuantumCircuit
    calibration: DeviceCalibration
    measurement_schema: Mapping[str, Sequence[str]]


def load_ty_tomography_config(
    circuit_config_path: str | Path = TY_TOMOGRAPHY_CONFIG_PATH,
) -> TyTomographyConfig:
    """Load and validate one full-circuit tomography YAML configuration."""
    circuit_config_path = Path(circuit_config_path)
    try:
        raw_config = yaml.safe_load(circuit_config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ValueError(f"invalid tomography YAML in {circuit_config_path}") from error
    if not isinstance(raw_config, dict):
        raise ValueError(f"tomography config {circuit_config_path} must contain a mapping.")

    expected_fields = set(TyTomographyConfig.__annotations__)
    if set(raw_config) != expected_fields:
        raise ValueError(
            f"tomography config {circuit_config_path} fields must be {sorted(expected_fields)}."
        )
    return TyTomographyConfig(**raw_config)


def resolve_ty_tomography_config(
    *,
    circuit_config_path: str | Path = TY_TOMOGRAPHY_CONFIG_PATH,
    target_distance: TomographyTargetDistance | None = None,
    patch_sites: common.SteanePatchSites | None = None,
    merge_schedule: common.MergeSchedule | None = None,
    full_post_selection: bool | None = None,
    measure_encoder_flags: bool | None = None,
    measure_cultivation_flags: bool | None = None,
    post_injection_qed_checks: PostInjectionQedChecks | None = None,
    post_transfer_stabilizer_rounds: int | None = None,
    add_barriers: bool | None = None,
) -> TyTomographyConfig:
    """Load the tomography YAML and apply per-call overrides."""
    config = load_ty_tomography_config(circuit_config_path)
    overrides = {
        name: value
        for name, value in (
            ("target_distance", target_distance),
            ("patch_sites", patch_sites),
            ("merge_schedule", merge_schedule),
            ("full_post_selection", full_post_selection),
            ("measure_encoder_flags", measure_encoder_flags),
            ("measure_cultivation_flags", measure_cultivation_flags),
            ("post_injection_qed_checks", post_injection_qed_checks),
            ("post_transfer_stabilizer_rounds", post_transfer_stabilizer_rounds),
            ("add_barriers", add_barriers),
        )
        if value is not None
    }
    return replace(config, **overrides)


def build_ty_tomography_circuits(
    *,
    circuit_config_path: str | Path = TY_TOMOGRAPHY_CONFIG_PATH,
    target_distance: TomographyTargetDistance | None = None,
    patch_sites: common.SteanePatchSites | None = None,
    merge_schedule: common.MergeSchedule | None = None,
    full_post_selection: bool | None = None,
    measure_encoder_flags: bool | None = None,
    measure_cultivation_flags: bool | None = None,
    post_injection_qed_checks: PostInjectionQedChecks | None = None,
    post_transfer_stabilizer_rounds: int | None = None,
    add_barriers: bool | None = None,
) -> dict[common.SurfaceReadoutBasis, QuantumCircuit]:
    """Build paired full logical-X/Z readout circuits from one YAML config.

    Each circuit includes physical ``|Ty>`` injection, post-injection Steane
    QED, logical-H_XZ cultivation, all three lattice-surgery merge rounds, the
    configured post-transfer rounds, and destructive readout of every
    surface-code data qubit. Per-call keywords remain available as config
    overrides for compatibility.
    """
    config = resolve_ty_tomography_config(
        circuit_config_path=circuit_config_path,
        target_distance=target_distance,
        patch_sites=patch_sites,
        merge_schedule=merge_schedule,
        full_post_selection=full_post_selection,
        measure_encoder_flags=measure_encoder_flags,
        measure_cultivation_flags=measure_cultivation_flags,
        post_injection_qed_checks=post_injection_qed_checks,
        post_transfer_stabilizer_rounds=post_transfer_stabilizer_rounds,
        add_barriers=add_barriers,
    )
    if config.target_distance == 3:
        build_transfer_circuit = d3.build_steane_to_rotated_d3_lattice_surgery_circuit
    elif config.target_distance == 5:
        build_transfer_circuit = d5.build_steane_to_rotated_d5_lattice_surgery_circuit
    else:  # pragma: no cover - validated by TyTomographyConfig.
        raise ValueError("target_distance must be 3 or 5.")

    circuits = {}
    for basis in TOMOGRAPHY_BASES:
        circuit = build_transfer_circuit(
            patch_sites=config.patch_sites,
            merge_schedule=config.merge_schedule,
            injected_state="Ty",
            cultivated_state="Ty",
            include_cultivation=True,
            measure_encoder_flags=config.measure_encoder_flags,
            measure_cultivation_flags=config.measure_cultivation_flags,
            post_injection_qed_checks=config.resolved_post_injection_qed_checks,
            surface_readout_basis=basis,
            full_post_selection=config.full_post_selection,
            post_transfer_stabilizer_rounds=(config.resolved_post_transfer_stabilizer_rounds),
            add_barriers=config.add_barriers,
        )
        circuit.metadata = {
            **(circuit.metadata or {}),
            "measurement_schema": ty_tomography_measurement_schema(config, basis),
        }
        circuits[basis] = circuit
    return circuits


def ty_tomography_measurement_schema(
    config: TyTomographyConfig,
    basis: common.SurfaceReadoutBasis,
) -> dict[str, tuple[str, ...]]:
    """Return one tomography circuit's ordered classical-bit labels.

    Builders attach this schema to circuit metadata, saved-job loaders identify
    submitted circuits by it, and phenomenological analysis parses count keys
    with it.
    """
    if config.target_distance == 3:
        transfer_spec = d3.TRANSFER_SPEC
    elif config.target_distance == 5:
        transfer_spec = d5.TRANSFER_SPEC
    else:  # pragma: no cover - validated by TyTomographyConfig.
        raise ValueError("target_distance must be 3 or 5.")
    return common.transfer_measurement_labels(
        transfer_spec,
        patch_sites=config.patch_sites,
        include_cultivation=True,
        merge_schedule=config.merge_schedule,
        surface_readout_basis=basis,
        measure_encoder_flags=config.measure_encoder_flags,
        measure_cultivation_flags=config.measure_cultivation_flags,
        post_injection_qed_checks=config.resolved_post_injection_qed_checks,
        post_transfer_stabilizer_rounds=config.resolved_post_transfer_stabilizer_rounds,
        full_data_readout=True,
    )


def analyze_ty_tomography_counts(
    counts_by_basis: Mapping[common.SurfaceReadoutBasis, Mapping[str, int]],
    *,
    tomography_config: TyTomographyConfig | None = None,
    circuit_config_path: str | Path = TY_TOMOGRAPHY_CONFIG_PATH,
    target_distance: TomographyTargetDistance | None = None,
    patch_sites: common.SteanePatchSites | None = None,
    merge_schedule: common.MergeSchedule | None = None,
    full_post_selection: bool | None = None,
    measure_encoder_flags: bool | None = None,
    measure_cultivation_flags: bool | None = None,
    post_injection_qed_checks: PostInjectionQedChecks | None = None,
    post_transfer_stabilizer_rounds: int | None = None,
    persistent_surface_x_post_selection: bool,
    complementary_gap_threshold: float | None,
    measurement_schemas: (
        Mapping[common.SurfaceReadoutBasis, Mapping[str, Sequence[str]]] | None
    ) = None,
    submitted_circuits_by_basis: Mapping[common.SurfaceReadoutBasis, QuantumCircuit] | None = None,
    calibrations_by_basis: Mapping[common.SurfaceReadoutBasis, DeviceCalibration] | None = None,
    circuit_level_variants_by_basis: (
        Mapping[common.SurfaceReadoutBasis, Sequence[CircuitLevelTomographyVariant]] | None
    ) = None,
) -> TyTomographySummary:
    """Analyze paired counts using the same YAML config as construction.

    Analysis-only postselection is chosen here, so one saved job can be
    re-analyzed under different settings. ``persistent_surface_x_post_selection``
    rejects shots that fire persistent surface-X detectors, and
    ``complementary_gap_threshold`` (``None`` disables it) rejects circuit-level
    shots whose complementary gap falls below it.

    Hardware counts are decoded circuit-level: pass, by basis, the submitted
    circuit, its calibration, and its saved measurement schema. Without
    submitted circuits and calibrations, such as for simulated logical
    circuits, each count key is parsed and decoded phase by phase instead, without
    complementary-gap postselection.
    """
    config = tomography_config or resolve_ty_tomography_config(
        circuit_config_path=circuit_config_path,
        target_distance=target_distance,
        patch_sites=patch_sites,
        merge_schedule=merge_schedule,
        full_post_selection=full_post_selection,
        measure_encoder_flags=measure_encoder_flags,
        measure_cultivation_flags=measure_cultivation_flags,
        post_injection_qed_checks=post_injection_qed_checks,
        post_transfer_stabilizer_rounds=post_transfer_stabilizer_rounds,
        add_barriers=None,
    )
    if complementary_gap_threshold is not None and (
        type(complementary_gap_threshold) not in (int, float)
        or not math.isfinite(complementary_gap_threshold)
        or complementary_gap_threshold < 0
    ):
        raise ValueError("complementary_gap_threshold must be None or a nonnegative number.")
    circuit_level = (
        submitted_circuits_by_basis is not None
        or calibrations_by_basis is not None
        or circuit_level_variants_by_basis is not None
    )
    if circuit_level and (
        circuit_level_variants_by_basis is None
        and (
            submitted_circuits_by_basis is None
            or calibrations_by_basis is None
            or measurement_schemas is None
        )
    ):
        raise ValueError(
            "circuit-level decoding requires submitted circuits, calibrations, and "
            "measurement schemas by basis."
        )
    if not circuit_level and complementary_gap_threshold is not None:
        raise ValueError("complementary_gap_threshold requires circuit-level decoding inputs.")

    expectations: dict[common.SurfaceReadoutBasis, float] = {}
    accepted_shots: dict[common.SurfaceReadoutBasis, int] = {}
    total_shots: dict[common.SurfaceReadoutBasis, int] = {}
    for basis in TOMOGRAPHY_BASES:
        if basis not in counts_by_basis:
            raise ValueError(f"counts_by_basis is missing the {basis} experiment.")
        counts = counts_by_basis[basis]
        total_shots[basis] = sum(counts.values())
        if circuit_level_variants_by_basis is not None:
            logical_zero_shots = 0
            logical_one_shots = 0
            variants = circuit_level_variants_by_basis[basis]
            for variant in variants:
                variant_zero_shots, variant_one_shots = CircuitLevelDecoder(
                    variant.submitted_circuit,
                    calibration=variant.calibration,
                    tomography_config=config,
                    measurement_schema=variant.measurement_schema,
                    readout_basis=basis,
                    persistent_surface_x_post_selection=persistent_surface_x_post_selection,
                    complementary_gap_threshold=complementary_gap_threshold,
                ).decode_counts(variant.counts)
                logical_zero_shots += variant_zero_shots
                logical_one_shots += variant_one_shots
            if sum(sum(variant.counts.values()) for variant in variants) != total_shots[basis]:
                raise ValueError(f"{basis}-basis variant counts do not match pooled counts.")
        elif circuit_level:
            logical_zero_shots, logical_one_shots = CircuitLevelDecoder(
                submitted_circuits_by_basis[basis],
                calibration=calibrations_by_basis[basis],
                tomography_config=config,
                measurement_schema=measurement_schemas[basis],
                readout_basis=basis,
                persistent_surface_x_post_selection=persistent_surface_x_post_selection,
                complementary_gap_threshold=complementary_gap_threshold,
            ).decode_counts(counts)
        else:
            logical_zero_shots, logical_one_shots = _decode_phenomenological_counts(
                counts,
                config=config,
                basis=basis,
                persistent_surface_x_post_selection=persistent_surface_x_post_selection,
                measurement_schema=(
                    None if measurement_schemas is None else measurement_schemas[basis]
                ),
            )
        accepted_shots[basis] = logical_zero_shots + logical_one_shots
        if accepted_shots[basis] == 0:
            raise ValueError(f"no accepted {basis}-basis tomography shots.")
        expectations[basis] = (logical_zero_shots - logical_one_shots) / accepted_shots[basis]

    expectation_x = expectations["X"]
    expectation_z = expectations["Z"]
    linear_fidelity = (1.0 + (expectation_x + expectation_z) / 2**0.5) / 2.0
    return TyTomographySummary(
        expectation_x=expectation_x,
        expectation_z=expectation_z,
        # Independent finite-shot estimates can fall outside the Bloch ball.
        fidelity=min(1.0, max(0.0, linear_fidelity)),
        accepted_x_shots=accepted_shots["X"],
        accepted_z_shots=accepted_shots["Z"],
        total_x_shots=total_shots["X"],
        total_z_shots=total_shots["Z"],
    )


def _decode_phenomenological_counts(
    counts: Mapping[str, int],
    *,
    config: TyTomographyConfig,
    basis: common.SurfaceReadoutBasis,
    persistent_surface_x_post_selection: bool,
    measurement_schema: Mapping[str, Sequence[str]] | None,
) -> tuple[int, int]:
    """Return accepted logical-zero and logical-one shots from per-phase decoding.

    Raw Steane, interface, and final-X checks reject first; surviving keys are
    parsed, postselected, and spacetime-decoded phase by phase.
    """
    if config.target_distance == 3:
        parse_transfer_measurements = d3.parse_transfer_measurements
        transfer_spec = d3.TRANSFER_SPEC
    elif config.target_distance == 5:
        parse_transfer_measurements = d5.parse_transfer_measurements
        transfer_spec = d5.TRANSFER_SPEC
    else:  # pragma: no cover - validated by TyTomographyConfig.
        raise ValueError("target_distance must be 3 or 5.")
    labels = (
        ty_tomography_measurement_schema(config, basis)
        if measurement_schema is None
        else measurement_schema
    )
    positions_by_family: dict[str, dict[str, int]] = {}
    offset = 0
    for family, family_labels in labels.items():
        positions_by_family[family] = {
            label: offset + index for index, label in enumerate(family_labels)
        }
        offset += len(family_labels)
    raw_zero_positions = tuple(
        position
        for family in common.STEANE_POSTSELECTION_FAMILIES
        for position in positions_by_family.get(family, {}).values()
    )
    interface_positions = tuple(positions_by_family["interface_z"].values())
    interface_width = len(transfer_spec.merged_boundary_checks)
    final_x_positions = {
        **positions_by_family["steane_x"],
        **positions_by_family["surface_x_recovery"],
    }
    logical_zero_shots = logical_one_shots = 0
    for bitstring, count in counts.items():
        if any(qiskit_bit_value(bitstring, position) for position in raw_zero_positions):
            continue
        if any(
            qiskit_bit_value(bitstring, interface_positions[index])
            != qiskit_bit_value(bitstring, interface_positions[index + interface_width])
            or qiskit_bit_value(bitstring, interface_positions[index])
            != qiskit_bit_value(bitstring, interface_positions[index + 2 * interface_width])
            for index in range(interface_width)
        ):
            continue
        if any(
            sum(qiskit_bit_value(bitstring, final_x_positions[label]) for label in support) % 2
            for support in common.FINAL_X_VERIFICATION_CHECKS.values()
        ):
            continue
        measurements = parse_transfer_measurements(
            bitstring,
            patch_sites=config.patch_sites,
            merge_schedule=config.merge_schedule,
            include_cultivation=True,
            measure_encoder_flags=config.measure_encoder_flags,
            measure_cultivation_flags=config.measure_cultivation_flags,
            post_injection_qed_checks=config.resolved_post_injection_qed_checks,
            surface_readout_basis=basis,
            full_post_selection=config.full_post_selection,
            persistent_surface_x_post_selection=persistent_surface_x_post_selection,
            post_transfer_stabilizer_rounds=config.resolved_post_transfer_stabilizer_rounds,
            measurement_labels=labels,
        )
        if not common.transfer_acceptance_succeeds(measurements):
            continue
        if common.corrected_surface_logical_readout(measurements, basis=basis):
            logical_one_shots += count
        else:
            logical_zero_shots += count
    return logical_zero_shots, logical_one_shots
