"""Classical interpretation of Steane-to-surface lattice-surgery results.

Circuit and register construction live in ``state_transfer_common_qiskit``.
This module parses Qiskit count keys, reconstructs paper-defined detectors,
applies acceptance policy, decodes surface syndromes, and builds the logical
Pauli frame.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from research_qec_haoran.decoding.pauli_frame import PauliFrame
from research_qec_haoran.decoding.stabilizer_frame import StabilizerFrame
from research_qec_haoran.state_transfer_by_lattice_surgery.state_transfer_common_qiskit import (
    HIRANO_STEANE_LOGICAL_X,
    LOCAL_STEANE_X_STABILIZER_CHECKS,
    LOCAL_X_AB_CORRELATION_CHECKS,
    STEANE_LS_VERIFICATION_ROUNDS,
    STEANE_X_RECONSTRUCTED_CHECKS,
    ClassicalRegisterSpec,
    SurfaceCodeTransferSpec,
    SurfaceReadoutBasis,
    SurfaceReadoutMode,
    parity,
    surface_logical_readout_label,
    transfer_classical_register_specs,
)
from research_qec_haoran.surface_code.decoding.rotated_decoder import (
    RotatedDecoder,
    RotatedDetectedErrors,
)
from research_qec_haoran.utils.normalize import parse_qiskit_registers

SyndromeItem = tuple[str, int]
SyndromeSignature = tuple[int, ...]


@dataclass(frozen=True)
class LookupTableEarlyRejection:
    """Paper-style early-rejection table over pre-expansion syndromes.

    This is separate from Steane syndrome postselection. The table stores
    pre-expansion signatures that offline sampling predicts will fail final
    complementary-gap postselection. ``remove_all_nontrivial`` implements the
    paper's special policy of rejecting every nonzero signature.
    """

    rejected_signatures: frozenset[SyndromeSignature] = frozenset()
    remove_all_nontrivial: bool = False

    @classmethod
    def from_signatures(
        cls,
        signatures: Iterable[Iterable[int]],
        *,
        remove_all_nontrivial: bool = False,
    ) -> LookupTableEarlyRejection:
        """Build a rejection table from iterable bit signatures."""
        return cls(
            rejected_signatures=frozenset(tuple(int(bit) for bit in sig) for sig in signatures),
            remove_all_nontrivial=remove_all_nontrivial,
        )

    def rejects(self, signature: Iterable[int]) -> bool:
        """Return whether the table rejects ``signature``."""
        normalized = tuple(int(bit) for bit in signature)
        if self.remove_all_nontrivial and any(normalized):
            return True
        return normalized in self.rejected_signatures


def parse_transfer_measurements_common(
    bitstring: str,
    *,
    specs: Sequence[ClassicalRegisterSpec],
    include_cultivation: bool,
    merged_rounds: int,
    include_merged_surface_syndromes: bool,
    include_steane_verification: bool,
    include_surface_qec: bool,
    surface_qec_rounds: int,
    surface_readout_basis: SurfaceReadoutBasis | None,
    surface_readout_mode: SurfaceReadoutMode,
    merged_boundary_checks: Sequence[tuple[str, tuple[str, ...], tuple[str, ...]]],
    merged_surface_z_labels: Sequence[str],
    merged_surface_x_labels: Sequence[str],
    surface_post_z_labels: Sequence[str],
    surface_post_x_labels: Sequence[str],
) -> dict[str, object]:
    """Parse target-independent transfer measurement registers.

    The returned dictionary has both raw register information and derived
    protocol fields.  For example:

    - `registers[name]["raw"]`: the raw Qiskit count-key chunk.
    - `registers[name]["labels"]`: semantic labels assigned to the chunk.
    - `registers[name]["bits"]`: parsed label-to-bit mapping.
    - `merged_zz_rounds`: one dictionary per merged-boundary round.
    - `merged_surface_*_rounds`: surface syndromes measured during the merge.
    - `steane_verification_rounds`: paper check groups on the color code.
    - `surface_qec_deltas`: optional decoded-reference syndrome changes.

    This parser does not decide acceptance by itself.  It normalizes the raw
    data so the distance-specific module can add reconstructed X checks,
    acceptance booleans, and optional QEC summaries.
    """
    register_labels = [(spec.name, spec.labels) for spec in specs]
    parsed_register_bits = parse_qiskit_registers(bitstring, register_labels)

    register_chunks: dict[str, dict[str, object]] = {}
    for spec in specs:
        register_chunks[spec.name] = {
            "raw": "".join(
                str(parsed_register_bits[spec.name][label]) for label in reversed(spec.labels)
            ),
            "labels": spec.labels,
            "bits": parsed_register_bits[spec.name],
        }

    parsed: dict[str, object] = {"registers": register_chunks}

    def register_bits(name: str) -> dict[str, int]:
        return register_chunks[name]["bits"]

    parsed["surface_init_z"] = register_bits("init_z")

    if surface_readout_basis is not None:
        if surface_readout_mode == "data":
            parsed["surface_readout"] = register_bits(f"surface_{surface_readout_basis.lower()}")
        else:
            parsed["surface_logical_readout"] = register_bits(
                f"surface_logical_{surface_readout_basis.lower()}"
            )

    if include_surface_qec:
        post_z_bits = register_bits("surface_post_z")
        post_x_bits = register_bits("surface_post_x")
        parsed["surface_post_z_rounds"] = [
            {
                label: post_z_bits[f"round{round_index + 1}_{label}"]
                for label in surface_post_z_labels
            }
            for round_index in range(surface_qec_rounds)
        ]
        parsed["surface_post_x_rounds"] = [
            {
                label: post_x_bits[f"round{round_index + 1}_{label}"]
                for label in surface_post_x_labels
            }
            for round_index in range(surface_qec_rounds)
        ]
        parsed["surface_post_z"] = parsed["surface_post_z_rounds"][-1]
        parsed["surface_post_x"] = parsed["surface_post_x_rounds"][-1]

    if include_merged_surface_syndromes:
        merged_surface_z_bits = register_bits("merged_surface_z")
        merged_surface_x_bits = register_bits("merged_surface_x")
        parsed["merged_surface_z_rounds"] = [
            {
                label: merged_surface_z_bits[f"round{round_index + 1}_{label}"]
                for label in merged_surface_z_labels
            }
            for round_index in range(merged_rounds)
        ]
        parsed["merged_surface_x_rounds"] = [
            {
                label: merged_surface_x_bits[f"round{round_index + 1}_{label}"]
                for label in merged_surface_x_labels
            }
            for round_index in range(merged_rounds)
        ]
        parsed["merged_surface_z"] = parsed["merged_surface_z_rounds"][-1]
        parsed["merged_surface_x"] = parsed["merged_surface_x_rounds"][-1]
        parsed["merged_surface_syndrome_deltas"] = merged_surface_syndrome_deltas(
            parsed,
            merged_surface_z_labels=merged_surface_z_labels,
            merged_surface_x_labels=merged_surface_x_labels,
        )

    if "surface_x_recovery" in register_chunks:
        parsed["surface_x_recovery"] = register_bits("surface_x_recovery")

    parsed["steane_x"] = register_bits("steane_x")
    if include_steane_verification:
        verification_bits = register_bits("steane_verify")
        parsed["steane_verification"] = verification_bits
        parsed["steane_verification_rounds"] = [
            {
                check_name: verification_bits[f"round{round_index + 1}_{check_name}"]
                for check_name in round_checks
            }
            for round_index, round_checks in enumerate(STEANE_LS_VERIFICATION_ROUNDS)
        ]

    merged_bits = register_bits("ls_zz")
    parsed["merged_zz_rounds"] = [
        {
            check_name: merged_bits[f"round{round_index + 1}_{check_name}"]
            for check_name, _, _ in merged_boundary_checks
        }
        for round_index in range(merged_rounds)
    ]
    parsed["enc_helpers"] = register_bits("enc_helpers")
    if include_cultivation:
        parsed["cultivation"] = register_bits("cult")
    return parsed


def parse_transfer_measurements(
    bitstring: str,
    *,
    spec: SurfaceCodeTransferSpec,
    include_cultivation: bool = True,
    merged_rounds: int = 3,
    include_merged_surface_syndromes: bool = True,
    include_steane_verification: bool = True,
    include_surface_qec: bool = False,
    surface_qec_rounds: int = 3,
    surface_readout_basis: SurfaceReadoutBasis | None = None,
    surface_readout_mode: SurfaceReadoutMode = "data",
) -> dict[str, object]:
    """Parse one transfer shot using the same patch spec as the builder."""
    specs = transfer_classical_register_specs(
        spec,
        include_cultivation=include_cultivation,
        merged_rounds=merged_rounds,
        include_merged_surface_syndromes=include_merged_surface_syndromes,
        include_steane_verification=include_steane_verification,
        include_surface_qec=include_surface_qec,
        surface_qec_rounds=surface_qec_rounds,
        surface_readout_basis=surface_readout_basis,
        surface_readout_mode=surface_readout_mode,
    )
    parsed = parse_transfer_measurements_common(
        bitstring,
        specs=specs,
        include_cultivation=include_cultivation,
        merged_rounds=merged_rounds,
        include_merged_surface_syndromes=include_merged_surface_syndromes,
        include_steane_verification=include_steane_verification,
        include_surface_qec=include_surface_qec,
        surface_qec_rounds=surface_qec_rounds,
        surface_readout_basis=surface_readout_basis,
        surface_readout_mode=surface_readout_mode,
        merged_boundary_checks=spec.merged_boundary_checks,
        merged_surface_z_labels=spec.z_labels,
        merged_surface_x_labels=spec.merged_x_labels,
        surface_post_z_labels=spec.z_labels,
        surface_post_x_labels=spec.x_labels,
    )

    if "surface_x_recovery" in parsed:
        parsed["reconstructed_x_checks"] = reconstructed_x_checks(parsed)
        parsed["local_x_verification_checks"] = local_x_verification_checks(parsed)
        parsed["x_verification_succeeds"] = x_verification_succeeds(parsed)
    if include_steane_verification:
        parsed["steane_verification_succeeds"] = steane_verification_succeeds(parsed)
    if include_merged_surface_syndromes:
        recoveries = merged_surface_qec_recoveries(
            parsed,
            surface_order=spec.surface_order,
            z_stabilizers=spec.z_stabilizers,
            x_stabilizers=spec.x_stabilizers,
            logical_x=spec.logical_x,
            logical_z=spec.logical_z,
        )
        parsed["merged_surface_qec_recoveries"] = recoveries
        parsed["merged_surface_qec_syndrome_is_trivial"] = all(
            recovery.is_trivial for recovery in recoveries
        )
    if include_surface_qec:
        parsed["surface_qec_deltas"] = surface_qec_deltas(
            parsed,
            surface_post_z_labels=spec.z_labels,
            surface_post_x_labels=spec.x_labels,
        )
        recoveries = surface_qec_recoveries(
            parsed,
            surface_order=spec.surface_order,
            z_stabilizers=spec.z_stabilizers,
            x_stabilizers=spec.x_stabilizers,
            logical_x=spec.logical_x,
            logical_z=spec.logical_z,
            surface_post_z_labels=spec.z_labels,
            surface_post_x_labels=spec.x_labels,
        )
        parsed["surface_qec_recoveries"] = recoveries
        parsed["surface_qec_syndrome_is_trivial"] = all(
            recovery.is_trivial for recovery in recoveries
        )
    return parsed


def merged_boundary_logical_zz_outcome(measurements: dict[str, object]) -> int:
    """Return the logical ``Z_L Z_L`` parity inferred from merged checks.

    The function first takes the parity of the merged-boundary checks inside
    each repeated round.  All rounds must agree.  The resulting bit is the
    lattice-surgery `Z_L Z_L` measurement and is recorded as an X byproduct on
    the target surface code by `transfer_pauli_frame`.
    """
    round_parities = [
        sum(round_bits.values()) % 2 for round_bits in measurements["merged_zz_rounds"]
    ]
    if len(set(round_parities)) != 1:
        raise ValueError("Merged-boundary rounds disagree.")
    return round_parities[0]


def merged_boundary_checks_stable(
    measurements: dict[str, object],
    merged_boundary_checks: Sequence[tuple[str, tuple[str, ...], tuple[str, ...]]],
) -> bool:
    """Return whether every repeated merged-boundary check is time-stable."""
    rounds = measurements["merged_zz_rounds"]
    return all(
        len({round_bits[check_name] for round_bits in rounds}) == 1
        for check_name, _, _ in merged_boundary_checks
    )


def steane_logical_x_outcome(measurements: dict[str, object]) -> int:
    """Return the Steane logical-X outcome from paper support ``X146``."""
    steane_x = measurements["steane_x"]
    return sum(steane_x[label] for label in HIRANO_STEANE_LOGICAL_X) % 2


def reconstructed_x_checks(measurements: dict[str, object]) -> dict[str, int]:
    """Reconstruct paper-style final X checks from demolition data.

    After the Steane block is measured in the X basis, products of those
    physical outcomes give the final color-code X stabilizers `X0235` and
    `X0246`.  If a recovered surface top-edge bit is present, the mixed check
    `X0145ab` is reconstructed from Steane `X0145` times recovered `X_ab`.
    """
    steane_x = measurements["steane_x"]
    checks = {
        name: parity(steane_x, support)
        for name, support in STEANE_X_RECONSTRUCTED_CHECKS.items()
        if name != "X0145"
    }
    surface_x_recovery = measurements.get("surface_x_recovery")
    if surface_x_recovery is not None:
        checks["X0145ab"] = (
            parity(steane_x, STEANE_X_RECONSTRUCTED_CHECKS["X0145"]) ^ surface_x_recovery["X_ab"]
        )
    return checks


def local_x_verification_checks(measurements: dict[str, object]) -> dict[str, int]:
    """Return deterministic final X checks used by transfer postselection.

    These checks are deterministic on accepted noiseless transfers and are used
    as local fault-detection predicates.  They are not decoded as surface-code
    syndromes; any nonzero value makes `x_verification_succeeds` false.
    """
    steane_x = measurements["steane_x"]
    checks = {
        name: parity(steane_x, support)
        for name, support in LOCAL_STEANE_X_STABILIZER_CHECKS.items()
    }
    surface_x_recovery = measurements.get("surface_x_recovery")
    if surface_x_recovery is None:
        return checks

    mixed_bits = {**steane_x, "X_ab": surface_x_recovery["X_ab"]}
    checks.update(
        {
            name: parity(mixed_bits, support)
            for name, support in LOCAL_X_AB_CORRELATION_CHECKS.items()
        }
    )
    return checks


def x_verification_succeeds(measurements: dict[str, object]) -> bool:
    """Return whether all deterministic final X checks are trivial."""
    return all(bit == 0 for bit in local_x_verification_checks(measurements).values())


def steane_verification_succeeds(measurements: dict[str, object]) -> bool:
    """Return whether non-destructive Steane LS checks are all trivial."""
    verification = measurements.get("steane_verification")
    if verification is None:
        return True
    return all(bit == 0 for bit in verification.values())


def surface_qec_deltas(
    measurements: dict[str, object],
    *,
    surface_post_z_labels: Sequence[str],
    surface_post_x_labels: Sequence[str],
) -> list[dict[str, dict[str, int]]]:
    """Return change-detected post-transfer surface-code stabilizer deltas.

    This is for optional full surface-code QEC rounds after transfer, not for
    the paper's merged-interval stabilizer measurements. References continue
    from the last merged-interval round when available. Recovered top-edge X
    checks seed the X frame after Steane demolition; only checks with no earlier
    measurement use the first post-transfer round as their initial reference.
    """
    init_z = measurements["surface_init_z"]
    post_z_rounds = measurements["surface_post_z_rounds"]
    post_x_rounds = measurements["surface_post_x_rounds"]

    z_reference = dict(init_z)
    if measurements.get("merged_surface_z_rounds"):
        z_reference.update(measurements["merged_surface_z_rounds"][-1])
    z_frame = StabilizerFrame()
    z_frame.record_many(z_reference)

    x_reference: dict[str, int] = {}
    if measurements.get("merged_surface_x_rounds"):
        x_reference.update(measurements["merged_surface_x_rounds"][-1])
    surface_x_recovery = measurements.get("surface_x_recovery")
    if surface_x_recovery is not None:
        x_reference.update(surface_x_recovery)
    for label in surface_post_x_labels:
        x_reference.setdefault(label, post_x_rounds[0][label])
    x_frame = StabilizerFrame()
    x_frame.record_many(x_reference)

    deltas: list[dict[str, dict[str, int]]] = []
    for post_z, post_x in zip(post_z_rounds, post_x_rounds, strict=True):
        z_deltas = z_frame.compare_many(post_z, update=True)
        x_deltas = x_frame.compare_many(post_x, update=True)
        deltas.append(
            {
                "z": {label: z_deltas[label] for label in surface_post_z_labels},
                "x": {label: x_deltas[label] for label in surface_post_x_labels},
            }
        )
    return deltas


def merged_surface_syndrome_deltas(
    measurements: dict[str, object],
    *,
    merged_surface_z_labels: Sequence[str],
    merged_surface_x_labels: Sequence[str],
) -> list[dict[str, dict[str, int]]]:
    """Return change-detected surface-code deltas measured during the merge.

    These are the compatible surface stabilizers extracted while the Steane and
    surface patches are connected by merged-boundary Z checks.  Z deltas compare
    against the initial surface Z frame.  X deltas compare against the expected
    `|+>` value for the compatible X stabilizers measured during the merge.

    Top-edge X checks that do not commute with active merged-boundary Z checks
    are intentionally absent here and recovered only after Steane demolition.
    """
    z_frame = StabilizerFrame()
    z_frame.record_many(
        {label: measurements["surface_init_z"][label] for label in merged_surface_z_labels}
    )

    x_frame = StabilizerFrame()
    x_frame.record_many({label: 0 for label in merged_surface_x_labels})

    deltas: list[dict[str, dict[str, int]]] = []
    for merged_z, merged_x in zip(
        measurements["merged_surface_z_rounds"],
        measurements["merged_surface_x_rounds"],
        strict=True,
    ):
        z_deltas = z_frame.compare_many(merged_z, update=True)
        x_deltas = x_frame.compare_many(merged_x, update=True)
        deltas.append(
            {
                "z": {label: z_deltas[label] for label in merged_surface_z_labels},
                "x": {label: x_deltas[label] for label in merged_surface_x_labels},
            }
        )
    return deltas


def surface_decoder(
    *,
    surface_order: Sequence[str],
    z_stabilizers: Mapping[str, tuple[str, ...]],
    x_stabilizers: Mapping[str, tuple[str, ...]],
    logical_x: Sequence[str],
    logical_z: Sequence[str],
) -> RotatedDecoder:
    """Return a matching decoder for a rotated surface-code target."""
    return RotatedDecoder(
        data_qubits=surface_order,
        z_stabilizers=z_stabilizers,
        x_stabilizers=x_stabilizers,
        logical_x=logical_x,
        logical_z=logical_z,
    )


def merged_surface_qec_recoveries(
    measurements: dict[str, object],
    *,
    surface_order: Sequence[str],
    z_stabilizers: Mapping[str, tuple[str, ...]],
    x_stabilizers: Mapping[str, tuple[str, ...]],
    logical_x: Sequence[str],
    logical_z: Sequence[str],
) -> list[RotatedDetectedErrors]:
    """Decode surface-code deltas measured while the patches are merged.

    The decoder is restricted to checks that commute with the active merged
    boundary: all measured Z checks and the measured subset of X checks.  This
    reuses the repository's rotated-code decoder for the Qiskit abstraction.
    A circuit-level noisy study should instead decode the complete detector
    error model, as in the paper's Stim implementation.
    """
    deltas_by_round = measurements.get("merged_surface_syndrome_deltas", [])
    if not deltas_by_round:
        return []

    first_round = deltas_by_round[0]
    measured_z_stabilizers = {label: z_stabilizers[label] for label in first_round["z"]}
    measured_x_stabilizers = {label: x_stabilizers[label] for label in first_round["x"]}
    decoder = surface_decoder(
        surface_order=surface_order,
        z_stabilizers=measured_z_stabilizers,
        x_stabilizers=measured_x_stabilizers,
        logical_x=logical_x,
        logical_z=logical_z,
    )
    return [
        decoder.decode(z_deltas=deltas["z"], x_deltas=deltas["x"]) for deltas in deltas_by_round
    ]


def merged_surface_qec_recovery(measurements: dict[str, object], **kwargs) -> RotatedDetectedErrors:
    """Return the last decoded merged-interval surface recovery."""
    recoveries = merged_surface_qec_recoveries(measurements, **kwargs)
    if not recoveries:
        raise ValueError("measurements do not include merged surface syndrome rounds.")
    return recoveries[-1]


def surface_qec_recoveries(
    measurements: dict[str, object],
    *,
    surface_order: Sequence[str],
    z_stabilizers: Mapping[str, tuple[str, ...]],
    x_stabilizers: Mapping[str, tuple[str, ...]],
    logical_x: Sequence[str],
    logical_z: Sequence[str],
    surface_post_z_labels: Sequence[str],
    surface_post_x_labels: Sequence[str],
) -> list[RotatedDetectedErrors]:
    """Decode all post-transfer surface-code stabilizer deltas."""
    decoder = surface_decoder(
        surface_order=surface_order,
        z_stabilizers=z_stabilizers,
        x_stabilizers=x_stabilizers,
        logical_x=logical_x,
        logical_z=logical_z,
    )
    return [
        decoder.decode(z_deltas=deltas["z"], x_deltas=deltas["x"])
        for deltas in surface_qec_deltas(
            measurements,
            surface_post_z_labels=surface_post_z_labels,
            surface_post_x_labels=surface_post_x_labels,
        )
    ]


def surface_qec_recovery(measurements: dict[str, object], **kwargs) -> RotatedDetectedErrors:
    """Return the final decoded post-transfer surface-code recovery."""
    recoveries = surface_qec_recoveries(measurements, **kwargs)
    if not recoveries:
        raise ValueError("measurements do not include post-transfer surface QEC rounds.")
    return recoveries[-1]


def post_transfer_surface_syndrome_schedule(
    surface_qec_rounds: int,
    *,
    surface_init_z_labels: Sequence[str],
    surface_post_z_labels: Sequence[str],
    surface_post_x_labels: Sequence[str],
    surface_x_recovery_labels: Sequence[str] = (),
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Return the ordered surface-code syndrome schedule after transfer."""
    if surface_qec_rounds < 1:
        raise ValueError("surface_qec_rounds must be at least 1.")
    schedule: list[tuple[str, tuple[str, ...]]] = [("surface_init_z", tuple(surface_init_z_labels))]
    if surface_x_recovery_labels:
        schedule.append(("surface_top_x_recovery", tuple(surface_x_recovery_labels)))
    for round_index in range(surface_qec_rounds):
        schedule.append((f"round{round_index + 1}_surface_post_z", tuple(surface_post_z_labels)))
        schedule.append((f"round{round_index + 1}_surface_post_x", tuple(surface_post_x_labels)))
    return tuple(schedule)


def early_rejection_syndrome_items(
    measurements: dict[str, object],
    *,
    merged_surface_z_labels: Sequence[str],
    merged_surface_x_labels: Sequence[str],
    surface_post_z_labels: Sequence[str],
    surface_post_x_labels: Sequence[str],
) -> tuple[SyndromeItem, ...]:
    """Return the pre-expansion syndrome bits used by lookup-table rejection.

    The table acts only on correctable surface-code detector events available
    before expansion. Color-code checks, final X checks, and repeated merged
    boundary parities are mandatory postselection predicates; accepted shots
    have zero values for them, so including those bits would only pad every
    lookup key with constants.

    The names are kept with the bits so tests and notebooks can display which
    physical/protocol stage triggered rejection.
    """
    items: list[SyndromeItem] = []

    for round_index, deltas in enumerate(
        measurements.get("merged_surface_syndrome_deltas", []),
        start=1,
    ):
        items.extend(
            (f"merged_surface:round{round_index}:z:{label}", deltas["z"][label])
            for label in merged_surface_z_labels
        )
        items.extend(
            (f"merged_surface:round{round_index}:x:{label}", deltas["x"][label])
            for label in merged_surface_x_labels
        )

    for round_index, deltas in enumerate(measurements.get("surface_qec_deltas", []), start=1):
        items.extend(
            (f"surface_qec:round{round_index}:z:{label}", deltas["z"][label])
            for label in surface_post_z_labels
        )
        items.extend(
            (f"surface_qec:round{round_index}:x:{label}", deltas["x"][label])
            for label in surface_post_x_labels
        )

    return tuple(items)


def early_rejection_syndrome_signature(
    items: Iterable[SyndromeItem],
) -> SyndromeSignature:
    """Return only the ordered bits of ``early_rejection_syndrome_items``."""
    return tuple(bit for _, bit in items)


def coerce_lookup_table(
    lookup_table: LookupTableEarlyRejection | Iterable[Iterable[int]] | None,
) -> LookupTableEarlyRejection | None:
    """Normalize lookup-table input to ``LookupTableEarlyRejection``."""
    if lookup_table is None:
        return None
    if isinstance(lookup_table, LookupTableEarlyRejection):
        return lookup_table
    return LookupTableEarlyRejection.from_signatures(lookup_table)


def lookup_table_early_rejection_accepts(
    signature: SyndromeSignature,
    *,
    lookup_table: LookupTableEarlyRejection | Iterable[Iterable[int]] | None = None,
) -> bool:
    """Return whether the paper-style early-rejection table accepts a shot."""
    table = coerce_lookup_table(lookup_table)
    if table is None:
        return True
    return not table.rejects(signature)


def cultivation_succeeds(measurements: dict[str, object]) -> bool:
    """Return whether optional cultivation postselection bits are all zero."""
    cultivation = measurements.get("cultivation")
    if cultivation is None:
        return True
    return sum(cultivation.values()) == 0


def encoder_helpers_succeed(measurements: dict[str, object]) -> bool:
    """Return whether the encoder helper postselection bits are all zero."""
    enc_helpers = measurements.get("enc_helpers")
    if enc_helpers is None:
        return True
    return sum(enc_helpers.values()) == 0


def transfer_acceptance_succeeds(
    measurements: dict[str, object],
    *,
    merged_boundary_checks: Sequence[tuple[str, tuple[str, ...], tuple[str, ...]]],
    signature: SyndromeSignature,
    lookup_table: LookupTableEarlyRejection | Iterable[Iterable[int]] | None = None,
) -> bool:
    """Return whether transfer-local fault-detecting checks accept.

    Acceptance here means all hard fault-detection predicates are satisfied:
    encoder helper checks, optional cultivation checks, repeated merged-boundary
    consistency, Steane LS verification, final deterministic X checks, and the
    optional paper-style lookup-table early-rejection hook.

    Optional surface-code QEC rounds are not hard rejection by default; if they
    are included, their deltas enter the lookup-table signature but recovery is
    applied through the Pauli frame rather than by rejecting every nonzero
    surface syndrome.
    """
    return (
        encoder_helpers_succeed(measurements)
        and cultivation_succeeds(measurements)
        and merged_boundary_checks_stable(measurements, merged_boundary_checks)
        and steane_verification_succeeds(measurements)
        and x_verification_succeeds(measurements)
        and lookup_table_early_rejection_accepts(signature, lookup_table=lookup_table)
    )


def transfer_pauli_frame(
    measurements: dict[str, object],
    *,
    surface_order: Sequence[str],
    z_stabilizers: Mapping[str, tuple[str, ...]],
    x_stabilizers: Mapping[str, tuple[str, ...]],
    logical_x: Sequence[str],
    logical_z: Sequence[str],
    surface_post_z_labels: Sequence[str],
    surface_post_x_labels: Sequence[str],
    apply_merged_surface_phase_recovery: bool = True,
    apply_surface_qec: bool = True,
) -> PauliFrame:
    """Build the target Pauli frame from transfer and surface-syndrome outcomes.

    Lattice surgery leaves known Pauli byproducts on the target surface patch:

    - the merged-boundary logical `Z_L Z_L` outcome toggles an X byproduct;
    - the destructive Steane logical `X146` outcome toggles a Z byproduct.

    Compatible X-stabilizer syndromes measured during the merge diagnose target
    Z errors, whose logical effect is folded into the frame by default. The
    analogous spatial X-error recovery is deliberately not applied here: a
    target X error already present at the logical ZZ measurement is absorbed by
    that teleportation outcome, and applying it again double-counts the error.
    Distinguishing such an error from one occurring later in the merged cycle
    requires the paper's joint, time-resolved detector-error model.

    Optional post-transfer QEC occurs after the teleportation measurements, so
    both recovery sectors can be applied normally. Future logical readout should
    be interpreted through this frame rather than by physically applying gates.
    """
    frame = PauliFrame()
    frame.register("surface")
    frame.update_x("surface", bool(merged_boundary_logical_zz_outcome(measurements)))
    frame.update_z("surface", bool(steane_logical_x_outcome(measurements)))

    decoder: RotatedDecoder | None = None
    if apply_merged_surface_phase_recovery and "merged_surface_syndrome_deltas" in measurements:
        logical_x_support = frozenset(logical_x)
        for recovery in merged_surface_qec_recoveries(
            measurements,
            surface_order=surface_order,
            z_stabilizers=z_stabilizers,
            x_stabilizers=x_stabilizers,
            logical_x=logical_x,
            logical_z=logical_z,
        ):
            frame.update_z(
                "surface",
                len(recovery.z_error_sites & logical_x_support) % 2 == 1,
            )

    if apply_surface_qec and "surface_post_z" in measurements:
        if decoder is None:
            decoder = surface_decoder(
                surface_order=surface_order,
                z_stabilizers=z_stabilizers,
                x_stabilizers=x_stabilizers,
                logical_x=logical_x,
                logical_z=logical_z,
            )
        for deltas in surface_qec_deltas(
            measurements,
            surface_post_z_labels=surface_post_z_labels,
            surface_post_x_labels=surface_post_x_labels,
        ):
            decoder.update_pauli_frame(
                frame,
                "surface",
                z_deltas=deltas["z"],
                x_deltas=deltas["x"],
            )
    return frame


def raw_surface_logical_readout(
    measurements: dict[str, object],
    *,
    basis: SurfaceReadoutBasis,
    logical_x: Sequence[str],
    logical_z: Sequence[str],
) -> int:
    """Return the uncorrected surface logical readout bit."""
    if "surface_readout" not in measurements:
        if "surface_logical_readout" not in measurements:
            raise ValueError("measurements do not include a surface logical readout.")
        logical_dict = measurements["surface_logical_readout"]
        combined_label = surface_logical_readout_label(
            basis,
            logical_x=logical_x,
            logical_z=logical_z,
        )
        if combined_label in logical_dict:
            return logical_dict[combined_label]
        return sum(logical_dict.values()) % 2

    support = logical_x if basis.upper() == "X" else logical_z
    return sum(measurements["surface_readout"][label] for label in support) % 2


def corrected_surface_logical_readout(
    measurements: dict[str, object],
    *,
    basis: SurfaceReadoutBasis,
    frame: PauliFrame,
    logical_x: Sequence[str],
    logical_z: Sequence[str],
) -> int:
    """Return the surface logical readout corrected by the Pauli frame.

    Raw logical readout can come from data-basis measurement, destructive
    logical-support measurement, or one-bit indirect logical Pauli measurement.
    This helper extracts the raw logical bit and then applies the transfer
    byproducts stored in `frame`.
    """
    raw_bit = raw_surface_logical_readout(
        measurements,
        basis=basis,
        logical_x=logical_x,
        logical_z=logical_z,
    )
    if basis.upper() == "X":
        return frame.interpret_x_readout("surface", raw_bit)
    return frame.interpret_z_readout("surface", raw_bit)
