"""Classical interpretation of Steane-to-surface lattice-surgery results.

Circuit and register construction live in ``state_transfer_common_qiskit``.
This module parses Qiskit count keys, reconstructs paper-defined detectors,
applies acceptance policy, decodes surface syndromes, and builds the logical
Pauli frame.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from magic_state_cultivation.decoding.pauli_frame import PauliFrame
from magic_state_cultivation.decoding.stabilizer_frame import StabilizerFrame
from magic_state_cultivation.state_transfer_by_lattice_surgery.state_transfer_common_qiskit import (
    FINAL_X_VERIFICATION_CHECKS,
    HIRANO_STEANE_LOGICAL_X,
    LATTICE_SURGERY_ROUNDS,
    MergedBoundaryCheck,
    STEANE_LS_VERIFICATION_ROUNDS,
    SurfaceCodeTransferSpec,
    SurfaceReadoutBasis,
    parity,
    transfer_classical_register_specs,
)
from magic_state_cultivation.surface_code.decoding.rotated_decoder import (
    TemporalRotatedDecoder,
    TemporalRotatedDetectedErrors,
)
from magic_state_cultivation.utils.normalize import normalize_bit, parse_qiskit_registers


@dataclass(frozen=True)
class JointTransferDecodeResult:
    """Logical corrections decoded from the complete transfer record.

    ``surface_qec_logical_x_byproduct`` is deliberately separate from the
    teleportation X byproduct. It predicts only an X correction accumulated
    after the final merged ``Z_L Z_L`` measurement. Earlier target X errors
    are already represented in that measured teleportation outcome.

    Target Z errors are not absorbed by the merged-Z measurement, so their net
    logical effect is obtained from the ordinary temporal surface recovery.
    """

    surface_qec_recovery: TemporalRotatedDetectedErrors
    surface_qec_logical_x_byproduct: int
    surface_qec_logical_z_byproduct: int
    merged_boundary_delta_history: tuple[dict[str, int], ...]


class JointTransferDecoder:
    """Decode merge parity and surface syndromes as one transfer history.

    The modular circuit serializes each logical merge round as

    ``merged ZZ -> surface Z -> surface X``.

    Consequently, the X-error sector has six phenomenological data-fault
    intervals: before and after the merged-Z measurement in each of the three
    rounds. Surface-Z detector layers locate the round, while changes between
    consecutive merged-boundary records distinguish the two sides of the
    merged-Z measurement. Only the final interval occurs after the logical
    parity used for teleportation, so only its spatial edges carry the output
    logical-X fault ID.

    The graph also contains measurement-fault edges for both record families.
    This is a transfer-specific graph because the merged checks include Steane
    support and have a protocol-specific preparation boundary.
    """

    _X_ERROR_INTERVALS: tuple[tuple[int, int | None], ...] = (
        (0, None),
        (0, 0),
        (1, 0),
        (1, 1),
        (2, 1),
        (2, None),
    )

    def __init__(
        self,
        *,
        surface_order: Sequence[str],
        z_stabilizers: Mapping[str, tuple[str, ...]],
        logical_x: Sequence[str],
        logical_z: Sequence[str],
        merged_boundary_checks: Sequence[MergedBoundaryCheck],
        data_weight: float = 1.0,
        measurement_weight: float = 1.0,
    ) -> None:
        self.surface_order = tuple(surface_order)
        self.z_stabilizers = dict(z_stabilizers)
        self.z_labels = tuple(z_stabilizers)
        self.logical_x = frozenset(logical_x)
        self.logical_z = frozenset(logical_z)
        self.boundary_surface_supports = {
            name: tuple(surface_support)
            for name, _, surface_support in merged_boundary_checks
        }
        self.boundary_labels = tuple(self.boundary_surface_supports)
        self.data_weight = data_weight
        self.measurement_weight = measurement_weight

    @staticmethod
    def _validate_history(
        history: Sequence[Mapping[str, int | bool]],
        *,
        labels: Sequence[str],
        expected_layers: int,
        name: str,
    ) -> None:
        if len(history) != expected_layers:
            raise ValueError(f"{name} must contain exactly {expected_layers} layers.")
        expected = set(labels)
        if any(set(layer) != expected for layer in history):
            raise ValueError(f"every {name} layer must contain exactly the decoder checks.")

    @staticmethod
    def _raw_measurement_detector_layers(
        raw_index: int,
        *,
        raw_layers: int,
    ) -> tuple[int, ...]:
        """Return change-detector layers toggled by one raw measurement fault."""
        detector_layers: list[int] = []
        if raw_index > 0:
            detector_layers.append(raw_index - 1)
        if raw_index < raw_layers - 1:
            detector_layers.append(raw_index)
        return tuple(detector_layers)

    def _decode_post_merge_logical_x(
        self,
        *,
        surface_z_delta_history: Sequence[Mapping[str, int | bool]],
        merged_boundary_delta_history: Sequence[Mapping[str, int | bool]],
    ) -> int:
        """Predict the logical-X change after the final merged-Z measurement."""
        import pymatching

        self._validate_history(
            surface_z_delta_history,
            labels=self.z_labels,
            expected_layers=LATTICE_SURGERY_ROUNDS,
            name="surface Z delta history",
        )
        self._validate_history(
            merged_boundary_delta_history,
            labels=self.boundary_labels,
            expected_layers=LATTICE_SURGERY_ROUNDS - 1,
            name="merged-boundary delta history",
        )

        z_row_count = LATTICE_SURGERY_ROUNDS * len(self.z_labels)
        detector_count = z_row_count + (LATTICE_SURGERY_ROUNDS - 1) * len(
            self.boundary_labels
        )
        columns: list[np.ndarray] = []
        logical_faults: list[int] = []
        weights: list[float] = []

        def z_row(round_index: int, check_index: int) -> int:
            return round_index * len(self.z_labels) + check_index

        def boundary_row(delta_index: int, check_index: int) -> int:
            return z_row_count + delta_index * len(self.boundary_labels) + check_index

        def append_fault(
            detector_rows: Sequence[int],
            *,
            logical_fault: bool,
            weight: float,
        ) -> None:
            column = np.zeros(detector_count, dtype=np.uint8)
            column[list(detector_rows)] = 1
            detector_degree = int(column.sum())
            if not 1 <= detector_degree <= 2:
                raise ValueError(
                    "joint transfer matching requires every fault mechanism to touch "
                    "one or two detectors."
                )
            columns.append(column)
            logical_faults.append(int(logical_fault))
            weights.append(weight)

        # One data-X mechanism per surface site in each interval. The first
        # five intervals precede the final teleportation parity; the sixth is
        # between round-3 merged-Z and round-3 surface-Z extraction.
        for interval_index, (z_round_index, boundary_delta_index) in enumerate(
            self._X_ERROR_INTERVALS
        ):
            for site in self.surface_order:
                rows = [
                    z_row(z_round_index, check_index)
                    for check_index, label in enumerate(self.z_labels)
                    if site in self.z_stabilizers[label]
                ]
                if boundary_delta_index is not None:
                    rows.extend(
                        boundary_row(boundary_delta_index, check_index)
                        for check_index, label in enumerate(self.boundary_labels)
                        if site in self.boundary_surface_supports[label]
                    )
                append_fault(
                    rows,
                    logical_fault=(
                        interval_index == len(self._X_ERROR_INTERVALS) - 1
                        and site in self.logical_z
                    ),
                    weight=self.data_weight,
                )

        # A fault in a raw surface-Z result toggles the adjacent change
        # detectors. There are four raw layers: initialization plus rounds 1-3.
        for raw_index in range(LATTICE_SURGERY_ROUNDS + 1):
            for check_index in range(len(self.z_labels)):
                append_fault(
                    [
                        z_row(delta_index, check_index)
                        for delta_index in self._raw_measurement_detector_layers(
                            raw_index,
                            raw_layers=LATTICE_SURGERY_ROUNDS + 1,
                        )
                    ],
                    logical_fault=False,
                    weight=self.measurement_weight,
                )

        # The three raw merged-boundary layers produce two change layers.
        for raw_index in range(LATTICE_SURGERY_ROUNDS):
            for check_index in range(len(self.boundary_labels)):
                append_fault(
                    [
                        boundary_row(delta_index, check_index)
                        for delta_index in self._raw_measurement_detector_layers(
                            raw_index,
                            raw_layers=LATTICE_SURGERY_ROUNDS,
                        )
                    ],
                    logical_fault=False,
                    weight=self.measurement_weight,
                )

        check_matrix = np.stack(columns, axis=1)
        faults_matrix = np.array([logical_faults], dtype=np.uint8)
        syndrome = np.array(
            [
                normalize_bit(surface_z_delta_history[round_index][label])
                for round_index in range(LATTICE_SURGERY_ROUNDS)
                for label in self.z_labels
            ]
            + [
                normalize_bit(merged_boundary_delta_history[delta_index][label])
                for delta_index in range(LATTICE_SURGERY_ROUNDS - 1)
                for label in self.boundary_labels
            ],
            dtype=np.uint8,
        )
        matching = pymatching.Matching.from_check_matrix(
            check_matrix,
            weights=np.array(weights),
            faults_matrix=faults_matrix,
            use_virtual_boundary_node=True,
        )
        prediction = matching.decode(syndrome)
        return int(prediction[0])

    def decode(
        self,
        *,
        surface_qec_recovery: TemporalRotatedDetectedErrors,
        surface_z_delta_history: Sequence[Mapping[str, int | bool]],
        merged_boundary_rounds: Sequence[Mapping[str, int | bool]],
    ) -> JointTransferDecodeResult:
        """Return separate logical X/Z corrections for one transfer shot."""
        boundary_deltas = merged_boundary_syndrome_deltas(
            merged_boundary_rounds,
            merged_boundary_labels=self.boundary_labels,
        )
        logical_x_byproduct = self._decode_post_merge_logical_x(
            surface_z_delta_history=surface_z_delta_history,
            merged_boundary_delta_history=boundary_deltas,
        )
        logical_z_byproduct = len(surface_qec_recovery.z_error_sites & self.logical_x) % 2
        return JointTransferDecodeResult(
            surface_qec_recovery=surface_qec_recovery,
            surface_qec_logical_x_byproduct=logical_x_byproduct,
            surface_qec_logical_z_byproduct=logical_z_byproduct,
            merged_boundary_delta_history=tuple(dict(layer) for layer in boundary_deltas),
        )


def parse_transfer_measurements(
    bitstring: str,
    *,
    spec: SurfaceCodeTransferSpec,
    include_cultivation: bool = True,
    surface_readout_basis: SurfaceReadoutBasis | None = None,
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

    surface_init_z_bits = register_bits("surface_init_z")
    merged_z_bits = register_bits("merged_surface_z")
    merged_x_bits = register_bits("merged_surface_x")
    merged_zz_bits = register_bits("ls_zz")
    verification_bits = register_bits("steane_verify")
    parsed: dict[str, object] = {
        "registers": register_chunks,
        "enc_helpers": register_bits("enc_helpers"),
        "surface_init_z": surface_init_z_bits,
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
        parsed["surface_logical_readout"] = register_bits(
            f"surface_logical_{surface_readout_basis.lower()}"
        )

    parsed["merged_surface_syndrome_deltas"] = merged_surface_syndrome_deltas(
        parsed,
        merged_surface_z_labels=spec.z_labels,
        merged_surface_x_labels=spec.merged_x_labels,
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
    parsed["joint_transfer_decode_result"] = joint_transfer_decode(
        parsed,
        spec=spec,
        surface_qec_recovery=recovery,
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


def merged_boundary_syndrome_deltas(
    merged_boundary_rounds: Sequence[Mapping[str, int | bool]],
    *,
    merged_boundary_labels: Sequence[str],
) -> list[dict[str, int]]:
    """Return changes between consecutive merged-boundary records.

    The first merged-boundary measurement is the nondeterministic logical
    teleportation result, not a detector reference. The two later rounds each
    provide one change-detector layer relative to the preceding round.
    """
    if len(merged_boundary_rounds) != LATTICE_SURGERY_ROUNDS:
        raise ValueError(
            f"merged-boundary rounds must contain exactly {LATTICE_SURGERY_ROUNDS} layers."
        )
    expected = set(merged_boundary_labels)
    if any(set(round_bits) != expected for round_bits in merged_boundary_rounds):
        raise ValueError(
            "every merged-boundary round must contain exactly the decoder checks."
        )
    return [
        {
            label: int(
                normalize_bit(merged_boundary_rounds[round_index][label])
                ^ normalize_bit(merged_boundary_rounds[round_index - 1][label])
            )
            for label in merged_boundary_labels
        }
        for round_index in range(1, LATTICE_SURGERY_ROUNDS)
    ]


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
    surface patches are connected by merged-boundary Z checks. The product
    `|+>` preparation gives compatible X checks known +1 values, so all three
    merged X rounds produce detectors. A separate pre-merge Z-stabilizer
    measurement prepares logical `|+_L>` and establishes the nondeterministic
    Z reference, so all three merged Z rounds also produce change detectors.

    Top-edge X checks that do not commute with active merged-boundary Z checks
    are intentionally absent here and recovered only after Steane demolition.
    """
    x_frame = StabilizerFrame()
    x_frame.record_many({label: 0 for label in merged_surface_x_labels})

    z_frame = StabilizerFrame()
    z_frame.record_many(
        {
            label: measurements["surface_init_z"][label]
            for label in merged_surface_z_labels
        }
    )
    deltas: list[dict[str, dict[str, int]]] = []
    for merged_z, merged_x in zip(
        measurements["merged_surface_z_rounds"],
        measurements["merged_surface_x_rounds"],
        strict=True,
    ):
        z_deltas = z_frame.advance(merged_z)
        x_deltas = x_frame.compare_many(merged_x, update=True)
        deltas.append(
            {
                "z": dict(z_deltas),
                "x": {label: x_deltas[label] for label in merged_surface_x_labels},
            }
        )
    return deltas


def merged_surface_qec_recovery(
    measurements: dict[str, object],
    *,
    surface_order: Sequence[str],
    z_stabilizers: Mapping[str, tuple[str, ...]],
    x_stabilizers: Mapping[str, tuple[str, ...]],
) -> TemporalRotatedDetectedErrors:
    """Jointly decode all surface detector layers measured during the merge.

    Compatible X checks have a known preparation boundary and therefore three
    detector layers. Nondeterministic Z checks use the pre-merge initialization
    measurement as their reference and therefore also have three layers.
    Timelike matching edges distinguish repeated measurement faults from
    persistent data errors.
    """
    deltas_by_round = measurements["merged_surface_syndrome_deltas"]
    measured_z_labels = tuple(
        label for label in z_stabilizers if any(label in deltas["z"] for deltas in deltas_by_round)
    )
    measured_x_labels = tuple(
        label for label in x_stabilizers if any(label in deltas["x"] for deltas in deltas_by_round)
    )
    decoder = TemporalRotatedDecoder(
        data_qubits=surface_order,
        z_stabilizers={label: z_stabilizers[label] for label in measured_z_labels},
        x_stabilizers={label: x_stabilizers[label] for label in measured_x_labels},
    )
    return decoder.decode(
        z_delta_history=[deltas["z"] for deltas in deltas_by_round if deltas["z"]],
        x_delta_history=[deltas["x"] for deltas in deltas_by_round],
    )


def joint_transfer_decode(
    measurements: dict[str, object],
    *,
    spec: SurfaceCodeTransferSpec,
    surface_qec_recovery: TemporalRotatedDetectedErrors | None = None,
) -> JointTransferDecodeResult:
    """Decode final target corrections from parity and syndrome histories.

    The X-error sector jointly uses the repeated merged-boundary records and
    all surface-Z detector layers so that only errors after the final logical
    ``Z_L Z_L`` measurement contribute an additional logical-X correction.
    The Z-error sector uses the compatible surface-X history through the
    existing temporal recovery.
    """
    if surface_qec_recovery is None:
        surface_qec_recovery = merged_surface_qec_recovery(
            measurements,
            surface_order=spec.surface_order,
            z_stabilizers=spec.z_stabilizers,
            x_stabilizers=spec.x_stabilizers,
        )
    decoder = JointTransferDecoder(
        surface_order=spec.surface_order,
        z_stabilizers=spec.z_stabilizers,
        logical_x=spec.logical_x,
        logical_z=spec.logical_z,
        merged_boundary_checks=spec.merged_boundary_checks,
    )
    return decoder.decode(
        surface_qec_recovery=surface_qec_recovery,
        surface_z_delta_history=[
            deltas["z"] for deltas in measurements["merged_surface_syndrome_deltas"]
        ],
        merged_boundary_rounds=measurements["merged_zz_rounds"],
    )


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
) -> bool:
    """Return whether transfer-local fault-detecting checks accept.

    Acceptance here means all hard fault-detection predicates are satisfied:
    encoder helper checks, optional cultivation checks, repeated merged-boundary
    consistency, Steane LS verification, and final deterministic X checks.
    Surface syndromes are corrected rather than hard-postselected, as required
    by Section III A.
    """
    return (
        encoder_helpers_succeed(measurements)
        and cultivation_succeeds(measurements)
        and merged_boundary_checks_stable(measurements)
        and steane_verification_succeeds(measurements)
        and x_verification_succeeds(measurements)
    )


def transfer_pauli_frame(
    measurements: dict[str, object],
) -> PauliFrame:
    """Build the target Pauli frame from transfer and surface-syndrome outcomes.

    Lattice surgery leaves known Pauli byproducts on the target surface patch:

    - the merged-boundary logical `Z_L Z_L` outcome toggles an X byproduct;
    - the destructive Steane logical `X146` outcome toggles a Z byproduct.

    The joint transfer decoder combines changes in the repeated merged-boundary
    outcomes with the surface detector history. It adds a separate target-X
    correction only when the inferred logical effect occurs after the final
    merged ``Z_L Z_L`` measurement. Earlier target X errors are already
    absorbed into the teleportation outcome and are not counted again.

    Decoder-derived X and Z contributions remain separate Pauli-frame updates
    from the two teleportation byproducts. The frame performs the required
    modulo-two composition without obscuring their protocol origins.

    Future logical readout should be interpreted through this frame rather than
    by physically applying gates.
    """
    joint_result = measurements["joint_transfer_decode_result"]
    frame = PauliFrame()
    frame.register("surface")

    # Teleportation X byproduct from the logical Z_L Z_L measurement.
    frame.update_x("surface", bool(merged_boundary_logical_zz_outcome(measurements)))

    # Surface-QEC X correction accumulated after the final parity measurement.
    frame.update_x(
        "surface",
        bool(joint_result.surface_qec_logical_x_byproduct),
    )

    # Teleportation Z byproduct from the destructive Steane X_L measurement.
    frame.update_z("surface", bool(steane_logical_x_outcome(measurements)))

    # Surface-QEC Z correction decoded from compatible X-check history.
    frame.update_z(
        "surface",
        bool(joint_result.surface_qec_logical_z_byproduct),
    )
    return frame


def raw_surface_logical_readout(
    measurements: dict[str, object],
) -> int:
    """Return the parity of the destructively measured logical support."""
    if "surface_logical_readout" not in measurements:
        raise ValueError("measurements do not include a surface logical readout.")
    return sum(measurements["surface_logical_readout"].values()) % 2


def corrected_surface_logical_readout(
    measurements: dict[str, object],
    *,
    basis: SurfaceReadoutBasis,
) -> int:
    """Return the surface logical readout corrected by the Pauli frame.

    The raw bit is the parity of the destructively measured logical support.
    This helper applies the transfer byproducts inferred from the same shot.
    """
    raw_bit = raw_surface_logical_readout(measurements)
    frame = transfer_pauli_frame(measurements)
    if basis.upper() == "X":
        return frame.interpret_x_readout("surface", raw_bit)
    return frame.interpret_z_readout("surface", raw_bit)
