"""Logical X/Z tomography after Qiskit Steane-to-surface state transfer."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import yaml
from qiskit import QuantumCircuit

from msc.state_transfer import common
from msc.state_transfer import to_d3_rotated_qiskit as d3
from msc.state_transfer import to_d5_rotated_qiskit as d5

TomographyTargetDistance = Literal[3, 5]
TOMOGRAPHY_BASES: tuple[common.SurfaceReadoutBasis, ...] = ("X", "Z")
TY_TOMOGRAPHY_CONFIG_PATH = Path(__file__).with_name("tomography_config.yaml")


@dataclass(frozen=True)
class TyTomographyConfig:
    """Validated settings loaded from a full-circuit tomography YAML file."""

    target_distance: TomographyTargetDistance
    patch_sites: common.SteanePatchSites
    full_post_selection: bool
    persistent_surface_x_post_selection: bool
    post_transfer_stabilizer_rounds: int | None
    add_barriers: bool

    def __post_init__(self) -> None:
        if type(self.target_distance) is not int or self.target_distance not in (3, 5):
            raise ValueError("target_distance must be 3 or 5.")
        if type(self.patch_sites) is not int or self.patch_sites not in (11, 14):
            raise ValueError("patch_sites must be 11 or 14.")
        for name in ("full_post_selection", "persistent_surface_x_post_selection", "add_barriers"):
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


def load_ty_tomography_config(
    config_path: str | Path = TY_TOMOGRAPHY_CONFIG_PATH,
) -> TyTomographyConfig:
    """Load and validate one full-circuit tomography YAML configuration."""
    config_path = Path(config_path)
    try:
        raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ValueError(f"invalid tomography YAML in {config_path}") from error
    if not isinstance(raw_config, dict):
        raise ValueError(f"tomography config {config_path} must contain a mapping.")

    expected_fields = set(TyTomographyConfig.__annotations__)
    if set(raw_config) != expected_fields:
        raise ValueError(
            f"tomography config {config_path} fields must be {sorted(expected_fields)}."
        )
    return TyTomographyConfig(**raw_config)


def resolve_ty_tomography_config(
    *,
    config_path: str | Path = TY_TOMOGRAPHY_CONFIG_PATH,
    target_distance: TomographyTargetDistance | None = None,
    patch_sites: common.SteanePatchSites | None = None,
    full_post_selection: bool | None = None,
    persistent_surface_x_post_selection: bool | None = None,
    post_transfer_stabilizer_rounds: int | None = None,
    add_barriers: bool | None = None,
) -> TyTomographyConfig:
    """Load YAML and apply backward-compatible per-call overrides."""
    config = load_ty_tomography_config(config_path)
    overrides = {
        name: value
        for name, value in (
            ("target_distance", target_distance),
            ("patch_sites", patch_sites),
            ("full_post_selection", full_post_selection),
            ("persistent_surface_x_post_selection", persistent_surface_x_post_selection),
            ("post_transfer_stabilizer_rounds", post_transfer_stabilizer_rounds),
            ("add_barriers", add_barriers),
        )
        if value is not None
    }
    return replace(config, **overrides)


def build_ty_tomography_circuits(
    *,
    config_path: str | Path = TY_TOMOGRAPHY_CONFIG_PATH,
    target_distance: TomographyTargetDistance | None = None,
    patch_sites: common.SteanePatchSites | None = None,
    full_post_selection: bool | None = None,
    persistent_surface_x_post_selection: bool | None = None,
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
        config_path=config_path,
        target_distance=target_distance,
        patch_sites=patch_sites,
        full_post_selection=full_post_selection,
        persistent_surface_x_post_selection=persistent_surface_x_post_selection,
        post_transfer_stabilizer_rounds=post_transfer_stabilizer_rounds,
        add_barriers=add_barriers,
    )
    if config.target_distance == 3:
        build_transfer_circuit = d3.build_steane_to_rotated_d3_lattice_surgery_circuit
    elif config.target_distance == 5:
        build_transfer_circuit = d5.build_steane_to_rotated_d5_lattice_surgery_circuit
    else:  # pragma: no cover - validated by TyTomographyConfig.
        raise ValueError("target_distance must be 3 or 5.")

    return {
        basis: build_transfer_circuit(
            patch_sites=config.patch_sites,
            injected_state="Ty",
            cultivated_state="Ty",
            include_cultivation=True,
            surface_readout_basis=basis,
            full_post_selection=config.full_post_selection,
            post_transfer_stabilizer_rounds=(config.resolved_post_transfer_stabilizer_rounds),
            add_barriers=config.add_barriers,
        )
        for basis in TOMOGRAPHY_BASES
    }


def analyze_ty_tomography_counts(
    counts_by_basis: Mapping[common.SurfaceReadoutBasis, Mapping[str, int]],
    *,
    config_path: str | Path = TY_TOMOGRAPHY_CONFIG_PATH,
    target_distance: TomographyTargetDistance | None = None,
    patch_sites: common.SteanePatchSites | None = None,
    full_post_selection: bool | None = None,
    persistent_surface_x_post_selection: bool | None = None,
    post_transfer_stabilizer_rounds: int | None = None,
) -> TyTomographySummary:
    """Analyze paired counts using the same YAML config as construction."""
    config = resolve_ty_tomography_config(
        config_path=config_path,
        target_distance=target_distance,
        patch_sites=patch_sites,
        full_post_selection=full_post_selection,
        persistent_surface_x_post_selection=persistent_surface_x_post_selection,
        post_transfer_stabilizer_rounds=post_transfer_stabilizer_rounds,
        add_barriers=None,
    )
    if config.target_distance == 3:
        parse_transfer_measurements = d3.parse_transfer_measurements
    elif config.target_distance == 5:
        parse_transfer_measurements = d5.parse_transfer_measurements
    else:  # pragma: no cover - validated by TyTomographyConfig.
        raise ValueError("target_distance must be 3 or 5.")

    expectations: dict[common.SurfaceReadoutBasis, float] = {}
    accepted_shots: dict[common.SurfaceReadoutBasis, int] = {}
    total_shots: dict[common.SurfaceReadoutBasis, int] = {}
    for basis in TOMOGRAPHY_BASES:
        if basis not in counts_by_basis:
            raise ValueError(f"counts_by_basis is missing the {basis} experiment.")
        counts = counts_by_basis[basis]
        total_shots[basis] = sum(counts.values())
        signed_accepted_shots = 0
        accepted_shots[basis] = 0
        for bitstring, count in counts.items():
            measurements = parse_transfer_measurements(
                bitstring,
                patch_sites=config.patch_sites,
                include_cultivation=True,
                surface_readout_basis=basis,
                full_post_selection=config.full_post_selection,
                persistent_surface_x_post_selection=(config.persistent_surface_x_post_selection),
                post_transfer_stabilizer_rounds=(config.resolved_post_transfer_stabilizer_rounds),
            )
            if not common.transfer_acceptance_succeeds(measurements):
                continue
            logical_bit = common.corrected_surface_logical_readout(
                measurements,
                basis=basis,
            )
            accepted_shots[basis] += count
            signed_accepted_shots += (1 if logical_bit == 0 else -1) * count
        if accepted_shots[basis] == 0:
            raise ValueError(f"no accepted {basis}-basis tomography shots.")
        expectations[basis] = signed_accepted_shots / accepted_shots[basis]

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
