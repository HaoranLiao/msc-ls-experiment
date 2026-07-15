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
    FINAL_X_VERIFICATION_CHECKS,
    HIRANO_STEANE_LOGICAL_X,
    LATTICE_SURGERY_ROUNDS,
    STEANE_LS_VERIFICATION_ROUNDS,
    SurfaceCodeTransferSpec,
    SurfaceReadoutBasis,
    SurfaceReadoutMode,
    parity,
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


def parse_transfer_measurements(
    bitstring: str,
    *,
    spec: SurfaceCodeTransferSpec,
    include_cultivation: bool = True,
    surface_readout_basis: SurfaceReadoutBasis | None = None,
    surface_readout_mode: SurfaceReadoutMode = "data",
) -> dict[str, object]:
    """Parse and interpret one fixed-protocol transfer shot.

    The returned ``registers`` entry preserves each raw Qiskit count-key chunk,
    its semantic labels, and its parsed bits. The other entries expose the
    paper's three merged rounds, Steane checks, demolition outcomes, recovered
    top-edge X checks, surface detector deltas, and decoded recoveries.
    """
    specs = transfer_classical_register_specs(
        spec,
        include_cultivation=include_cultivation,
        surface_readout_basis=surface_readout_basis,
        surface_readout_mode=surface_readout_mode,
    )
    labels_by_register = [(register.name, register.labels) for register in specs]
    bits_by_register = parse_qiskit_registers(bitstring, labels_by_register)
    register_chunks = {
        register.name: {
            "raw": "".join(
                str(bits_by_register[register.name][label]) for label in reversed(register.labels)
            ),
            "labels": register.labels,
            "bits": bits_by_register[register.name],
        }
        for register in specs
    }

    def register_bits(name: str) -> dict[str, int]:
        return register_chunks[name]["bits"]

    merged_z_bits = register_bits("merged_surface_z")
    merged_x_bits = register_bits("merged_surface_x")
    merged_zz_bits = register_bits("ls_zz")
    verification_bits = register_bits("steane_verify")
    parsed: dict[str, object] = {
        "registers": register_chunks,
        "enc_helpers": register_bits("enc_helpers"),
        "surface_init_z": register_bits("init_z"),
        "surface_x_recovery": register_bits("surface_x_recovery"),
        "steane_x": register_bits("steane_x"),
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
        parsed["cultivation"] = register_bits("cult")
    if surface_readout_basis is not None:
        register_name = (
            f"surface_{surface_readout_basis.lower()}"
            if surface_readout_mode == "data"
            else f"surface_logical_{surface_readout_basis.lower()}"
        )
        output_name = (
            "surface_readout" if surface_readout_mode == "data" else "surface_logical_readout"
        )
        parsed[output_name] = register_bits(register_name)

    parsed["merged_surface_syndrome_deltas"] = merged_surface_syndrome_deltas(
        parsed,
        merged_surface_z_labels=spec.z_labels,
        merged_surface_x_labels=spec.merged_x_labels,
    )
    parsed["final_x_verification_checks"] = final_x_verification_checks(parsed)
    parsed["x_verification_succeeds"] = x_verification_succeeds(parsed)
    recoveries = merged_surface_qec_recoveries(
        parsed,
        surface_order=spec.surface_order,
        z_stabilizers=spec.z_stabilizers,
        x_stabilizers=spec.x_stabilizers,
        logical_x=spec.logical_x,
        logical_z=spec.logical_z,
    )
    parsed["merged_surface_qec_recoveries"] = recoveries
    logical_x_support = frozenset(spec.logical_x)
    parsed["merged_surface_logical_z_byproduct"] = (
        sum(len(recovery.z_error_sites & logical_x_support) for recovery in recoveries) % 2
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
) -> bool:
    """Return whether every repeated merged-boundary check is time-stable."""
    rounds = measurements["merged_zz_rounds"]
    return all(
        len({round_bits[check_name] for round_bits in rounds}) == 1 for check_name in rounds[0]
    )


def steane_logical_x_outcome(measurements: dict[str, object]) -> int:
    """Return the Steane logical-X outcome from paper support ``X146``."""
    steane_x = measurements["steane_x"]
    return sum(steane_x[label] for label in HIRANO_STEANE_LOGICAL_X) % 2


def final_x_verification_checks(measurements: dict[str, object]) -> dict[str, int]:
    """Reconstruct the paper's final X checks from demolition data.

    After the Steane block is measured in the X basis, products of those
    physical outcomes give the final color-code X stabilizers `X0235` and
    `X0246`.  If a recovered surface top-edge bit is present, the mixed check
    `X0145ab` is reconstructed from Steane `X0145` times recovered `X_ab`.
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

    No separate reference round is needed here. An accepted encoded Steane
    state is in the +1 eigenspace of every code stabilizer regardless of its
    logical magic state, so these absolute outcomes have a known zero frame.
    """
    return all(bit == 0 for bit in measurements["steane_verification"].values())


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
    deltas_by_round = measurements["merged_surface_syndrome_deltas"]
    first_round = deltas_by_round[0]
    measured_z_stabilizers = {label: z_stabilizers[label] for label in first_round["z"]}
    measured_x_stabilizers = {label: x_stabilizers[label] for label in first_round["x"]}
    decoder = RotatedDecoder(
        data_qubits=surface_order,
        z_stabilizers=measured_z_stabilizers,
        x_stabilizers=measured_x_stabilizers,
        logical_x=logical_x,
        logical_z=logical_z,
    )
    return [
        decoder.decode(z_deltas=deltas["z"], x_deltas=deltas["x"]) for deltas in deltas_by_round
    ]


def early_rejection_syndrome_items(
    measurements: dict[str, object],
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
        measurements["merged_surface_syndrome_deltas"],
        start=1,
    ):
        items.extend(
            (f"merged_surface:round{round_index}:z:{label}", bit)
            for label, bit in deltas["z"].items()
        )
        items.extend(
            (f"merged_surface:round{round_index}:x:{label}", bit)
            for label, bit in deltas["x"].items()
        )

    return tuple(items)


def early_rejection_syndrome_signature(
    measurements: dict[str, object],
) -> SyndromeSignature:
    """Return only the ordered bits of ``early_rejection_syndrome_items``."""
    return tuple(bit for _, bit in early_rejection_syndrome_items(measurements))


def lookup_table_early_rejection_accepts(
    measurements: dict[str, object],
    *,
    lookup_table: LookupTableEarlyRejection | Iterable[Iterable[int]] | None = None,
) -> bool:
    """Return whether the paper-style early-rejection table accepts a shot."""
    if lookup_table is None:
        return True
    table = (
        lookup_table
        if isinstance(lookup_table, LookupTableEarlyRejection)
        else LookupTableEarlyRejection.from_signatures(lookup_table)
    )
    return not table.rejects(early_rejection_syndrome_signature(measurements))


def cultivation_succeeds(measurements: dict[str, object]) -> bool:
    """Return whether optional cultivation postselection bits are all zero."""
    cultivation = measurements.get("cultivation")
    if cultivation is None:
        return True
    return sum(cultivation.values()) == 0


def encoder_helpers_succeed(measurements: dict[str, object]) -> bool:
    """Return whether the encoder helper postselection bits are all zero."""
    return sum(measurements["enc_helpers"].values()) == 0


def transfer_acceptance_succeeds(
    measurements: dict[str, object],
    *,
    lookup_table: LookupTableEarlyRejection | Iterable[Iterable[int]] | None = None,
) -> bool:
    """Return whether transfer-local fault-detecting checks accept.

    Acceptance here means all hard fault-detection predicates are satisfied:
    encoder helper checks, optional cultivation checks, repeated merged-boundary
    consistency, Steane LS verification, final deterministic X checks, and the
    optional paper-style lookup-table early-rejection hook. Surface syndromes
    are corrected rather than hard-postselected, as required by Section III A.
    """
    return (
        encoder_helpers_succeed(measurements)
        and cultivation_succeeds(measurements)
        and merged_boundary_checks_stable(measurements)
        and steane_verification_succeeds(measurements)
        and x_verification_succeeds(measurements)
        and lookup_table_early_rejection_accepts(measurements, lookup_table=lookup_table)
    )


def transfer_pauli_frame(
    measurements: dict[str, object],
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

    Future logical readout should be interpreted through this frame rather than
    by physically applying gates.
    """
    frame = PauliFrame()
    frame.register("surface")
    frame.update_x("surface", bool(merged_boundary_logical_zz_outcome(measurements)))
    frame.update_z("surface", bool(steane_logical_x_outcome(measurements)))

    frame.update_z(
        "surface",
        bool(measurements["merged_surface_logical_z_byproduct"]),
    )
    return frame


def raw_surface_logical_readout(
    measurements: dict[str, object],
    *,
    basis: SurfaceReadoutBasis,
    spec: SurfaceCodeTransferSpec,
) -> int:
    """Return the uncorrected surface logical readout bit."""
    if "surface_readout" not in measurements:
        if "surface_logical_readout" not in measurements:
            raise ValueError("measurements do not include a surface logical readout.")
        return sum(measurements["surface_logical_readout"].values()) % 2

    support = spec.logical_x if basis.upper() == "X" else spec.logical_z
    return sum(measurements["surface_readout"][label] for label in support) % 2


def corrected_surface_logical_readout(
    measurements: dict[str, object],
    *,
    basis: SurfaceReadoutBasis,
    spec: SurfaceCodeTransferSpec,
) -> int:
    """Return the surface logical readout corrected by the Pauli frame.

    Raw logical readout can come from data-basis measurement or destructive
    logical-support measurement. This helper extracts the raw logical bit and
    applies the transfer byproducts inferred from the same shot.
    """
    raw_bit = raw_surface_logical_readout(
        measurements,
        basis=basis,
        spec=spec,
    )
    frame = transfer_pauli_frame(measurements)
    if basis.upper() == "X":
        return frame.interpret_x_readout("surface", raw_bit)
    return frame.interpret_z_readout("surface", raw_bit)
