"""Framework-neutral Steane-to-surface lattice-surgery protocol semantics.

This module is the single source of truth for paper/Hirano labels, target
specifications, measurement-family labels, detector reconstruction, transfer
acceptance, temporal surface decoding, and logical Pauli-frame interpretation.
Qiskit and Stim allocation and construction live in their matching
``state_transfer_helper_<framework>`` modules; raw result parsing lives in
``state_transfer_analysis_<framework>``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, assert_never

from msc.decoding.pauli_frame import PauliFrame
from msc.decoding.stabilizer_frame import StabilizerFrame
from msc.rotated_surface_code import common
from msc.rotated_surface_code.decoding.decoder import (
    TemporalRotatedDecoder,
    TemporalRotatedDetectedErrors,
)
from msc.steane_code.decoding.decoder import (
    X_STABILIZER_LABELS,
    Z_STABILIZER_LABELS,
)
from msc.utils.circuit_common import resolve_logical_measurement_terms
from msc.utils.normalize import normalize_pauli_char

SurfaceReadoutBasis = Literal["X", "Y", "Z"]
LATTICE_SURGERY_ROUNDS = 3
MergedBoundaryCheck = tuple[str, tuple[str, ...], tuple[str, ...]]
CircuitAppender = Callable[..., None]

HIRANO_STEANE_DATA_ORDER: tuple[str, ...] = ("0", "1", "2", "3", "4", "5", "6")
HIRANO_STEANE_GEOMETRIC_ORDER: tuple[str, ...] = ("6", "4", "0", "2", "1", "5", "3")
HIRANO_STEANE_LOGICAL_X: tuple[str, ...] = ("1", "4", "6")
HIRANO_STEANE_LOGICAL_Z: tuple[str, ...] = ("1", "5", "3")
POST_INJECTION_QED_STABILIZER_LABELS: tuple[str, ...] = X_STABILIZER_LABELS + Z_STABILIZER_LABELS

STEANE_LS_VERIFICATION_STABILIZER_INDICES: dict[str, int] = {
    "Z0145": 1,
    "Z0235": 2,
    "X0235": 2,
    "Z0246": 0,
    "X0246": 0,
}
STEANE_LS_VERIFICATION_ROUNDS: tuple[tuple[str, ...], ...] = (
    ("Z0145", "Z0235", "X0235", "Z0246", "X0246"),
    ("Z0145", "Z0235", "Z0246"),
)
STEANE_LS_VERIFICATION_LABELS: tuple[str, ...] = tuple(
    f"round{round_index + 1}_{check_name}"
    for round_index, round_checks in enumerate(STEANE_LS_VERIFICATION_ROUNDS)
    for check_name in round_checks
)
FINAL_X_VERIFICATION_CHECKS: dict[str, tuple[str, ...]] = {
    "X0235": ("0", "2", "3", "5"),
    "X0246": ("0", "2", "4", "6"),
    "X0145ab": ("0", "1", "4", "5", "X_ab"),
}
GOOGLE_SITE_TO_HIRANO_LABEL: dict[int, str] = {
    0: "6",
    3: "4",
    4: "0",
    5: "2",
    8: "1",
    9: "5",
    10: "3",
}
HIRANO_LABEL_TO_GOOGLE_SITE: dict[str, int] = {
    label: site for site, label in GOOGLE_SITE_TO_HIRANO_LABEL.items()
}
STEANE_HELPER_PATCH_LABELS: tuple[str, ...] = ("h1", "h2", "h6", "h7")
STEANE_HELPER_LABEL_TO_GOOGLE_SITE: dict[str, int] = {
    "h1": 1,
    "h2": 2,
    "h6": 6,
    "h7": 7,
}


@dataclass(frozen=True)
class MeasurementFamilySpec:
    """Framework-neutral description of one measurement family.

    ``name`` is both the Qiskit classical-register name and the Stim record-
    family name. ``labels`` are the semantic bit labels used after parsing.
    Keeping this metadata with the protocol prevents builders, parsers, and
    tests from duplicating naming and ordering rules.
    """

    name: str
    labels: tuple[str, ...]


@dataclass(frozen=True)
class SurfaceCodeOperations:
    """Distance-specific surface-code operations used by transfer builders.

    Grouping these callables keeps ownership explicit without adding transfer-
    level wrappers around the d=3 and d=5 modules. Each framework supplies its
    own matching operation set.
    """

    prepare_product_plus: CircuitAppender
    append_scheduled_stabilizer_round: CircuitAppender
    append_logical_measurement: CircuitAppender


@dataclass(frozen=True)
class SurfaceCodeTransferSpec:
    """Distance-specific data and circuit operations for one transfer target.

    The d=3 and d=5 targets differ only in patch geometry, logical supports,
    merged-boundary checks, and top-edge X checks. Binding those values once
    avoids duplicate protocol builders. Distance- and framework-specific
    circuit callables are namespaced under ``surface_code_operations``.

    ``top_x_recovery_labels`` preserves paper-style measurement order, such as
    ``X_ab`` and ``X_cd``. Canonical supports and schedules remain owned by the
    distance-specific surface-code modules.
    """

    name: str
    distance: int
    surface_order: tuple[str, ...]
    logical_x: tuple[str, ...]
    logical_z: tuple[str, ...]
    z_stabilizers: Mapping[str, tuple[str, ...]]
    x_stabilizers: Mapping[str, tuple[str, ...]]
    merged_boundary_checks: tuple[MergedBoundaryCheck, ...]
    top_x_recovery_labels: tuple[str, ...]
    surface_code_operations: SurfaceCodeOperations

    @property
    def z_labels(self) -> tuple[str, ...]:
        """All ordinary surface Z-stabilizer labels in circuit order."""
        return tuple(self.z_stabilizers)

    @property
    def x_labels(self) -> tuple[str, ...]:
        """All ordinary surface X-stabilizer labels in circuit order."""
        return tuple(self.x_stabilizers)

    @property
    def merged_x_labels(self) -> tuple[str, ...]:
        """Surface X checks commuting with the active boundary-Z checks."""
        withheld = set(self.top_x_recovery_labels)
        return tuple(label for label in self.x_stabilizers if label not in withheld)

    @property
    def z_ancilla_labels(self) -> tuple[str, ...]:
        """Framework register or index labels for surface Z-check ancillas."""
        return tuple(label.lower() for label in self.z_labels)

    @property
    def x_ancilla_labels(self) -> tuple[str, ...]:
        """Framework register or index labels for surface X-check ancillas."""
        return tuple(label.lower() for label in self.x_labels)


def parity(bits: Mapping[str, int], labels: Sequence[str]) -> int:
    """Return the mod-2 parity over selected entries of a labeled bit mapping."""
    return sum(bits[label] for label in labels) % 2


def round_labels(labels: Sequence[str]) -> tuple[str, ...]:
    """Prefix three merge-round copies with ``roundN_`` for flat containers."""
    return common.repeated_round_labels(labels, rounds=LATTICE_SURGERY_ROUNDS)


def transfer_measurement_specs(
    spec: SurfaceCodeTransferSpec,
    *,
    include_cultivation: bool,
    surface_readout_basis: SurfaceReadoutBasis | None,
    include_post_transfer_stabilizer_rounds: bool = False,
) -> tuple[MeasurementFamilySpec, ...]:
    """Return measurement families for the fixed transfer sequence.

    The result is authoritative for Qiskit register allocation and parsing and
    for Stim family labels. Stim separately maps these labels to global record
    positions because physical append order interleaves families within every
    merged round.
    """
    measurement_specs = [
        MeasurementFamilySpec("enc_helpers", ("enc_h1", "enc_h2", "enc_h6", "enc_h7")),
        MeasurementFamilySpec("steane_qed", POST_INJECTION_QED_STABILIZER_LABELS),
    ]
    if include_cultivation:
        measurement_specs.append(
            MeasurementFamilySpec(
                "cult",
                ("root_x", "helper_h1", "helper_h2", "helper_h6", "helper_h7"),
            )
        )
    measurement_specs.extend(
        (
            MeasurementFamilySpec("surface_init_z", spec.z_labels),
            MeasurementFamilySpec(
                "ls_zz",
                round_labels(tuple(name for name, _, _ in spec.merged_boundary_checks)),
            ),
            MeasurementFamilySpec("merged_surface_z", round_labels(spec.z_labels)),
            MeasurementFamilySpec("merged_surface_x", round_labels(spec.merged_x_labels)),
            MeasurementFamilySpec("steane_verify", STEANE_LS_VERIFICATION_LABELS),
            MeasurementFamilySpec("steane_x", HIRANO_STEANE_DATA_ORDER),
            MeasurementFamilySpec("surface_x_recovery", spec.top_x_recovery_labels),
        )
    )
    if include_post_transfer_stabilizer_rounds:
        measurement_specs.extend(
            (
                MeasurementFamilySpec(
                    "post_transfer_surface_z",
                    common.repeated_round_labels(
                        spec.z_labels,
                        rounds=spec.distance,
                    ),
                ),
                MeasurementFamilySpec(
                    "post_transfer_surface_x",
                    common.repeated_round_labels(
                        spec.x_labels,
                        rounds=spec.distance,
                    ),
                ),
            )
        )
    if surface_readout_basis is not None:
        surface_readout_basis = normalize_pauli_char(surface_readout_basis)
        measurement_terms, _ = resolve_logical_measurement_terms(
            surface_readout_basis,
            logical_x_support=spec.logical_x,
            logical_z_support=spec.logical_z,
        )
        measurement_specs.append(
            MeasurementFamilySpec(
                f"surface_logical_{surface_readout_basis.lower()}",
                tuple(support_item for support_item, _ in measurement_terms),
            )
        )
    return tuple(measurement_specs)


def interpret_transfer_measurement_families(
    measurement_families: Mapping[str, Mapping[str, object]],
    *,
    spec: SurfaceCodeTransferSpec,
    include_cultivation: bool = True,
    include_post_transfer_stabilizer_rounds: bool = False,
    surface_readout_basis: SurfaceReadoutBasis | None = None,
) -> dict[str, object]:
    """Interpret one transfer shot from already-labeled measurement families.

    The returned ``registers`` entry preserves each parser's raw family
    metadata. Other entries expose the three merged rounds, Steane checks,
    demolition outcomes, recovered top-edge X checks, surface detector
    histories, decoded merge recovery, and complete post-transfer surface
    stabilizer reference.
    """

    def family_bits(name: str) -> Mapping[str, int]:
        bits = measurement_families[name]["bits"]
        if not isinstance(bits, Mapping):
            raise TypeError(f"measurement family {name!r} bits must be a mapping.")
        return bits

    surface_init_z_bits = family_bits("surface_init_z")
    merged_z_bits = family_bits("merged_surface_z")
    merged_x_bits = family_bits("merged_surface_x")
    merged_zz_bits = family_bits("ls_zz")
    verification_bits = family_bits("steane_verify")
    parsed: dict[str, object] = {
        "registers": dict(measurement_families),
        "enc_helpers": family_bits("enc_helpers"),
        "surface_init_z": surface_init_z_bits,
        "surface_x_recovery": family_bits("surface_x_recovery"),
        "steane_x": family_bits("steane_x"),
        "steane_verification": verification_bits,
        "steane_verification_rounds": [
            {
                check_name: verification_bits[f"round{round_index + 1}_{check_name}"]
                for check_name in round_checks
            }
            for round_index, round_checks in enumerate(STEANE_LS_VERIFICATION_ROUNDS)
        ],
        "merged_zz_rounds": [
            {
                check_name: merged_zz_bits[f"round{round_index + 1}_{check_name}"]
                for check_name, _, _ in spec.merged_boundary_checks
            }
            for round_index in range(LATTICE_SURGERY_ROUNDS)
        ],
        "merged_surface_z_rounds": [
            {label: merged_z_bits[f"round{round_index + 1}_{label}"] for label in spec.z_labels}
            for round_index in range(LATTICE_SURGERY_ROUNDS)
        ],
        "merged_surface_x_rounds": [
            {
                label: merged_x_bits[f"round{round_index + 1}_{label}"]
                for label in spec.merged_x_labels
            }
            for round_index in range(LATTICE_SURGERY_ROUNDS)
        ],
    }
    if include_cultivation:
        parsed["cultivation"] = family_bits("cult")
    parsed["post_injection_qed"] = family_bits("steane_qed")
    if include_post_transfer_stabilizer_rounds:
        post_z_bits = family_bits("post_transfer_surface_z")
        post_x_bits = family_bits("post_transfer_surface_x")
        parsed["post_transfer_surface_z_rounds"] = [
            {label: post_z_bits[f"round{round_index + 1}_{label}"] for label in spec.z_labels}
            for round_index in range(spec.distance)
        ]
        parsed["post_transfer_surface_x_rounds"] = [
            {label: post_x_bits[f"round{round_index + 1}_{label}"] for label in spec.x_labels}
            for round_index in range(spec.distance)
        ]
    if surface_readout_basis is not None:
        parsed["surface_logical_readout"] = family_bits(
            f"surface_logical_{surface_readout_basis.lower()}"
        )

    parsed["merged_surface_syndrome_delta_history"] = merged_surface_syndrome_delta_history(
        parsed,
        merged_surface_z_labels=spec.z_labels,
        merged_surface_x_labels=spec.merged_x_labels,
    )
    parsed["post_transfer_surface_stabilizer_frame"] = post_transfer_surface_stabilizer_frame(
        parsed,
        spec=spec,
    )
    if include_post_transfer_stabilizer_rounds:
        parsed["post_transfer_surface_syndrome_delta_history"] = (
            post_transfer_surface_syndrome_delta_history(
                parsed,
                initial_frame=parsed["post_transfer_surface_stabilizer_frame"],
            )
        )
    parsed["final_x_verification_checks"] = final_x_verification_checks(parsed)
    parsed["x_verification_succeeds"] = x_verification_succeeds(parsed)
    recovery = merged_surface_qec_recovery(
        parsed,
        surface_order=spec.surface_order,
        z_stabilizers=spec.z_stabilizers,
        x_stabilizers=spec.x_stabilizers,
    )
    parsed["merged_surface_qec_recovery"] = recovery
    parsed["merged_surface_qec_recoveries"] = (recovery,)
    parsed["merged_surface_logical_x_byproduct"] = (
        len(recovery.x_error_sites & frozenset(spec.logical_z)) % 2
    )
    parsed["merged_surface_logical_z_byproduct"] = (
        len(recovery.z_error_sites & frozenset(spec.logical_x)) % 2
    )
    return parsed


def post_transfer_surface_stabilizer_frame(
    measurements: dict[str, object],
    *,
    spec: SurfaceCodeTransferSpec,
) -> StabilizerFrame:
    """Return the complete surface-check reference after boundary restoration.

    The final merged round supplies the latest ordinary surface-Z and
    compatible surface-X values. After Steane demolition, separately restored
    top-edge X measurements supply first reference values for checks withheld
    during the merge. Those restored values are not a detector layer; later
    fixed-patch rounds compare against this frame.
    """
    final_z = measurements["merged_surface_z_rounds"][-1]
    final_compatible_x = measurements["merged_surface_x_rounds"][-1]
    restored_x = measurements["surface_x_recovery"]
    if set(final_z) != set(spec.z_labels):
        raise ValueError("final merged surface-Z round does not match the transfer spec.")
    if set(final_compatible_x) != set(spec.merged_x_labels):
        raise ValueError("final compatible surface-X round does not match the transfer spec.")
    if set(restored_x) != set(spec.top_x_recovery_labels):
        raise ValueError("restored top-edge surface-X checks do not match the transfer spec.")
    frame = StabilizerFrame()
    frame.record_many(final_z)
    frame.record_many(final_compatible_x)
    frame.record_many(restored_x)
    if set(frame.as_dict()) != set(spec.z_labels) | set(spec.x_labels):
        raise ValueError("post-transfer surface stabilizer frame is incomplete.")
    return frame


def merged_boundary_checks_stable(measurements: dict[str, object]) -> bool:
    """Return whether every repeated merged-boundary check is time-stable."""
    rounds = measurements["merged_zz_rounds"]
    if not rounds:
        return False
    check_names = tuple(rounds[0])
    if any(set(round_bits) != set(check_names) for round_bits in rounds[1:]):
        return False
    return all(
        round_bits[check_name] == rounds[0][check_name]
        for check_name in check_names
        for round_bits in rounds[1:]
    )


def merged_boundary_logical_zz_outcome(measurements: dict[str, object]) -> int:
    """Return the common logical ``Z_L Z_L`` parity of accepted boundary checks.

    Every individual check must be identical in all three merge rounds.
    Unstable histories are postselection failures and are never interpreted as
    a teleportation outcome. A measurement bit ``b`` represents eigenvalue
    ``(-1)^b``; multiplying stable check eigenvalues is therefore their XOR.
    """
    if not merged_boundary_checks_stable(measurements):
        raise ValueError("Merged-boundary checks are not individually stable.")
    stable_boundary_bits = measurements["merged_zz_rounds"][0]
    return parity(stable_boundary_bits, tuple(stable_boundary_bits))


def steane_logical_x_outcome(measurements: dict[str, object]) -> int:
    """Return the Steane logical-X outcome from paper support ``X146``."""
    return parity(measurements["steane_x"], HIRANO_STEANE_LOGICAL_X)


def final_x_verification_checks(measurements: dict[str, object]) -> dict[str, int]:
    """Reconstruct the paper's final X checks from demolition data.

    Products of Steane X-basis demolition outcomes give color-code stabilizers
    ``X0235`` and ``X0246``. The mixed check ``X0145ab`` is reconstructed from
    Steane ``X0145`` times the recovered surface ``X_ab`` result.
    """
    bits = {
        **measurements["steane_x"],
        "X_ab": measurements["surface_x_recovery"]["X_ab"],
    }
    return {name: parity(bits, support) for name, support in FINAL_X_VERIFICATION_CHECKS.items()}


def x_verification_succeeds(measurements: dict[str, object]) -> bool:
    """Return whether all deterministic final X checks are trivial."""
    return all(bit == 0 for bit in final_x_verification_checks(measurements).values())


def steane_verification_succeeds(measurements: dict[str, object]) -> bool:
    """Return whether non-destructive Steane LS checks are all trivial.

    No separate reference round is needed: an accepted encoded Steane state is
    in the +1 eigenspace of every code stabilizer regardless of its logical
    state, so these absolute outcomes have a known zero frame.
    """
    return all(bit == 0 for bit in measurements["steane_verification"].values())


def merged_surface_syndrome_delta_history(
    measurements: dict[str, object],
    *,
    merged_surface_z_labels: Sequence[str],
    merged_surface_x_labels: Sequence[str],
) -> list[dict[str, dict[str, int]]]:
    """Return change-detected surface history measured during the merge.

    Product ``|+>`` preparation gives compatible X checks known +1 values, so
    all three merged X rounds produce detectors. The pre-merge Z measurement
    prepares logical ``|+_L>`` and establishes the nondeterministic Z
    reference, so all three merged Z rounds also produce change detectors.
    Top-edge X checks that anticommute with boundary-Z checks are absent and
    recovered only after Steane demolition.
    """
    x_frame = StabilizerFrame()
    x_frame.record_many({label: 0 for label in merged_surface_x_labels})
    z_frame = StabilizerFrame()
    z_frame.record_many(
        {label: measurements["surface_init_z"][label] for label in merged_surface_z_labels}
    )
    delta_history: list[dict[str, dict[str, int]]] = []
    for merged_z, merged_x in zip(
        measurements["merged_surface_z_rounds"],
        measurements["merged_surface_x_rounds"],
        strict=True,
    ):
        z_delta_layer = z_frame.advance(merged_z)
        x_delta_layer = x_frame.compare_many(merged_x, update=True)
        delta_history.append(
            {
                "z": dict(z_delta_layer),
                "x": {label: x_delta_layer[label] for label in merged_surface_x_labels},
            }
        )
    return delta_history


def post_transfer_surface_syndrome_delta_history(
    measurements: dict[str, object],
    *,
    initial_frame: StabilizerFrame,
) -> list[dict[str, dict[str, int]]]:
    """Return full-check delta layers after the transfer boundary.

    Restored top-edge checks and final compatible merge-round values form the
    initial reference. Each optional full surface-code round advances that
    reference in both CSS sectors.
    """
    frame = StabilizerFrame()
    frame.record_many(initial_frame.as_dict())
    return [
        {"z": frame.advance(z_round), "x": frame.advance(x_round)}
        for z_round, x_round in zip(
            measurements["post_transfer_surface_z_rounds"],
            measurements["post_transfer_surface_x_rounds"],
            strict=True,
        )
    ]


def merged_surface_qec_recovery(
    measurements: dict[str, object],
    *,
    surface_order: Sequence[str],
    z_stabilizers: Mapping[str, tuple[str, ...]],
    x_stabilizers: Mapping[str, tuple[str, ...]],
) -> TemporalRotatedDetectedErrors:
    """Jointly decode all surface detector layers measured during the merge.

    Compatible X checks use a known preparation boundary; nondeterministic Z
    checks use the pre-merge initialization result. Both sectors therefore
    contribute three detector layers. Timelike matching edges distinguish
    repeated measurement faults from persistent data errors.
    """
    delta_history = measurements["merged_surface_syndrome_delta_history"]
    measured_z_labels = tuple(
        label
        for label in z_stabilizers
        if any(label in delta_layer["z"] for delta_layer in delta_history)
    )
    measured_x_labels = tuple(
        label
        for label in x_stabilizers
        if any(label in delta_layer["x"] for delta_layer in delta_history)
    )
    decoder = TemporalRotatedDecoder(
        data_qubits=surface_order,
        z_stabilizers={label: z_stabilizers[label] for label in measured_z_labels},
        x_stabilizers={label: x_stabilizers[label] for label in measured_x_labels},
    )
    return decoder.decode(
        z_delta_history=[delta_layer["z"] for delta_layer in delta_history if delta_layer["z"]],
        x_delta_history=[delta_layer["x"] for delta_layer in delta_history],
    )


def cultivation_succeeds(measurements: dict[str, object]) -> bool:
    """Return whether optional cultivation postselection bits are all zero."""
    cultivation = measurements.get("cultivation")
    return cultivation is None or sum(cultivation.values()) == 0


def post_injection_qed_succeeds(measurements: dict[str, object]) -> bool:
    """Return whether the post-injection Steane QED syndrome is trivial."""
    return all(bit == 0 for bit in measurements["post_injection_qed"].values())


def encoder_helpers_succeed(measurements: dict[str, object]) -> bool:
    """Return whether encoder-helper postselection bits are all zero."""
    return sum(measurements["enc_helpers"].values()) == 0


def transfer_acceptance_succeeds(measurements: dict[str, object]) -> bool:
    """Return whether transfer-local fault-detecting checks accept.

    Acceptance requires encoder helpers, post-injection QED, optional
    cultivation, repeated boundary consistency, Steane LS verification, and
    final deterministic X checks. Surface syndromes are corrected rather than
    hard-postselected, as required by Section III A of the protocol.
    """
    return (
        encoder_helpers_succeed(measurements)
        and post_injection_qed_succeeds(measurements)
        and cultivation_succeeds(measurements)
        and merged_boundary_checks_stable(measurements)
        and steane_verification_succeeds(measurements)
        and x_verification_succeeds(measurements)
    )


def transfer_pauli_frame(measurements: dict[str, object]) -> PauliFrame:
    """Build the target Pauli frame from transfer and decoded syndrome outcomes.

    The merged-boundary ``Z_L Z_L`` outcome toggles an X byproduct, while the
    destructive Steane ``X146`` outcome toggles a Z byproduct. The temporal
    surface decoder independently supplies logical X and Z corrections from
    its three detector layers. Future logical readout should be interpreted
    through this frame rather than corrected with physical gates.
    """
    frame = PauliFrame()
    frame.register("surface")
    frame.update_x("surface", bool(merged_boundary_logical_zz_outcome(measurements)))
    frame.update_x("surface", bool(measurements["merged_surface_logical_x_byproduct"]))
    frame.update_z("surface", bool(steane_logical_x_outcome(measurements)))
    frame.update_z("surface", bool(measurements["merged_surface_logical_z_byproduct"]))
    return frame


def raw_surface_logical_readout(measurements: dict[str, object]) -> int:
    """Return the parity of the destructively measured logical terms."""
    if "surface_logical_readout" not in measurements:
        raise ValueError("measurements do not include a surface logical readout.")
    return sum(measurements["surface_logical_readout"].values()) % 2


def corrected_surface_logical_readout(
    measurements: dict[str, object],
    *,
    basis: SurfaceReadoutBasis,
) -> int:
    """Return the surface logical readout corrected by the transfer Pauli frame.

    The raw bit is the parity of the destructively measured physical terms;
    this helper applies transfer and decoded-QEC byproducts from the same shot.
    """
    raw_bit = raw_surface_logical_readout(measurements)
    pauli_frame = transfer_pauli_frame(measurements)
    match basis:
        case "X":
            return pauli_frame.interpret_x_readout("surface", raw_bit)
        case "Y":
            return pauli_frame.interpret_y_readout("surface", raw_bit)
        case "Z":
            return pauli_frame.interpret_z_readout("surface", raw_bit)
    assert_never(basis)


def logical_expectation_from_logical_counts(logical_counts: Mapping[int, int]) -> float:
    """Return the expectation represented by logical zero/one outcome counts."""
    total = sum(logical_counts.values())
    if total == 0:
        raise ValueError("logical_counts must contain at least one accepted shot.")
    return (logical_counts.get(0, 0) - logical_counts.get(1, 0)) / total
