"""Stim circuit helpers for Steane-to-rotated-surface state transfer.

Distance-specific target modules define patch geometry and available checks.
This module owns the target-independent Stim pieces: deterministic qubit-index
allocation, the global measurement-record layout, Hirano-to-Crumble mapping,
mixed Steane/surface checks, and the caller-owned appender and circuit-owning
builder. Framework-neutral labels and analysis live in
``ls_common``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import stim

from msc.lattice_surgery import (
    ls_common as common,
)
from msc.rotated_surface_code import helper_stim as helper
from msc.rotated_surface_code.post_transfer import post_transfer_stim
from msc.steane_code import magic_state_cultivation_stim as cultivation
from msc.steane_code import state_injection_stim as steane
from msc.utils import circuit_stim
from msc.utils.normalize import normalize_pauli_char

# ---------------------------------------------------------------------------
# Measurement-record layout and detector counts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HiranoLabeledQubits:
    """Paper-labeled Stim qubit indices used by the transfer circuit.

    ``steane_data`` uses Hirano labels ``"0"`` through ``"6"``;
    ``steane_helpers`` uses the 11-site encoder's helper labels;
    ``surface_data`` uses target labels such as ``a..y``; surface ancillas are
    reused across initialization, merge, and top-edge recovery; merge ancillas
    measure mixed Steane/surface ``ZZ`` checks.
    """

    steane_data: dict[str, int]
    steane_helpers: dict[str, int]
    surface_data: dict[str, int]
    surface_z_ancillas: dict[str, int]
    surface_x_ancillas: dict[str, int]
    merge_ancillas: dict[str, int]


@dataclass(frozen=True)
class MeasurementRecordSpec:
    """Semantic labels and global Stim positions for one measurement family.

    Physical append order interleaves some families across merged rounds, so
    ``positions`` need not be contiguous even though ``labels`` retain their
    protocol order.
    """

    name: str
    labels: tuple[str, ...]
    positions: tuple[int, ...]

    def bits(self, shot: Sequence[bool | int]) -> dict[str, int]:
        """Return this measurement family's labeled bits from one shot."""
        return {
            label: int(bool(shot[position]))
            for label, position in zip(self.labels, self.positions, strict=True)
        }

    def positions_by_label(self) -> dict[str, int]:
        """Return global measurement positions keyed by semantic label."""
        return dict(zip(self.labels, self.positions, strict=True))

    def round_positions(self, *, rounds: int) -> tuple[tuple[int, ...], ...]:
        """Return round-major position rows for a repeated family."""
        if rounds <= 0:
            raise ValueError("rounds must be positive.")
        if len(self.positions) % rounds:
            raise ValueError(f"{self.name} cannot be divided into {rounds} rounds.")
        round_width = len(self.positions) // rounds
        return tuple(
            self.positions[round_index * round_width : (round_index + 1) * round_width]
            for round_index in range(rounds)
        )


@dataclass(frozen=True)
class TransferMeasurementLayout:
    """Authoritative semantic layout of the Stim global measurement record.

    ``records`` provides named views used by parsing and tests, while
    ``num_measurements`` validates that the layout and built circuit agree.
    """

    records: tuple[MeasurementRecordSpec, ...]
    num_measurements: int

    def record(self, name: str) -> MeasurementRecordSpec:
        """Return the named measurement family."""
        try:
            return next(record for record in self.records if record.name == name)
        except StopIteration as error:
            raise KeyError(name) from error

    def with_appended_records(
        self,
        records: Sequence[tuple[str, Sequence[str]]],
    ) -> TransferMeasurementLayout:
        """Return a layout extended by measurement families appended in order."""
        existing_names = {record.name for record in self.records}
        selected_records = tuple((name, tuple(labels)) for name, labels in records)
        selected_names = tuple(name for name, _ in selected_records)
        if len(selected_names) != len(set(selected_names)):
            raise ValueError("appended measurement-family names must be unique.")
        if existing_names & set(selected_names):
            raise ValueError("an appended measurement-family name already exists.")

        next_position = self.num_measurements
        appended_records: list[MeasurementRecordSpec] = []
        for name, labels in selected_records:
            positions = tuple(range(next_position, next_position + len(labels)))
            appended_records.append(
                MeasurementRecordSpec(name=name, labels=labels, positions=positions)
            )
            next_position += len(labels)
        return TransferMeasurementLayout(
            records=self.records + tuple(appended_records),
            num_measurements=next_position,
        )

    def shifted(self, measurement_offset: int) -> TransferMeasurementLayout:
        """Return this layout translated into a larger circuit record."""
        if measurement_offset < 0:
            raise ValueError("measurement_offset must be nonnegative.")
        return TransferMeasurementLayout(
            records=tuple(
                MeasurementRecordSpec(
                    name=record.name,
                    labels=record.labels,
                    positions=tuple(position + measurement_offset for position in record.positions),
                )
                for record in self.records
            ),
            num_measurements=self.num_measurements + measurement_offset,
        )


@dataclass(frozen=True)
class TransferDetectorLayout:
    """Detector indices separated by postselection and correction policy."""

    protocol_postselection_indices: tuple[int, ...]
    surface_correction_indices: tuple[int, ...]


class _MeasurementLayoutBuilder:
    """Accumulate split measurement families in physical append order."""

    def __init__(self) -> None:
        self._next_position = 0
        self._family_order: list[str] = []
        self._labels: dict[str, list[str]] = {}
        self._positions: dict[str, list[int]] = {}

    def add(self, name: str, labels: Sequence[str]) -> None:
        selected_labels = tuple(labels)
        if name not in self._labels:
            self._family_order.append(name)
            self._labels[name] = []
            self._positions[name] = []
        positions = tuple(range(self._next_position, self._next_position + len(selected_labels)))
        self._labels[name].extend(selected_labels)
        self._positions[name].extend(positions)
        self._next_position += len(selected_labels)

    def finish(self) -> TransferMeasurementLayout:
        records = tuple(
            MeasurementRecordSpec(
                name=name,
                labels=tuple(self._labels[name]),
                positions=tuple(self._positions[name]),
            )
            for name in self._family_order
        )
        return TransferMeasurementLayout(records=records, num_measurements=self._next_position)


def transfer_measurement_layout(
    spec: common.SurfaceCodeTransferSpec,
    *,
    include_cultivation: bool,
    surface_readout_basis: common.SurfaceReadoutBasis | None,
    include_post_transfer_stabilizer_rounds: bool = False,
) -> TransferMeasurementLayout:
    """Return the authoritative Stim measurement-record layout.

    The common measurement specs own family labels. This helper assigns their
    actual global positions in physical append order: front-end measurements,
    surface initialization, per-round boundary/surface/Steane groups,
    demolition and recovery, optional full surface rounds, then optional
    logical readout.
    """
    if surface_readout_basis is not None:
        surface_readout_basis = normalize_pauli_char(surface_readout_basis)
    labels_by_family = {
        register_spec.name: register_spec.labels
        for register_spec in common.transfer_measurement_specs(
            spec,
            include_cultivation=include_cultivation,
            surface_readout_basis=surface_readout_basis,
            include_post_transfer_stabilizer_rounds=include_post_transfer_stabilizer_rounds,
        )
    }
    layout = _MeasurementLayoutBuilder()
    layout.add("enc_helpers", labels_by_family["enc_helpers"])
    layout.add("steane_qed", labels_by_family["steane_qed"])
    if include_cultivation:
        layout.add("cult", labels_by_family["cult"])
    layout.add("surface_init_z", labels_by_family["surface_init_z"])

    steane_verify_start = 0
    for round_index in range(common.LATTICE_SURGERY_ROUNDS):
        for family, round_width in (
            ("ls_zz", len(spec.merged_boundary_checks)),
            ("merged_surface_z", len(spec.z_labels)),
            ("merged_surface_x", len(spec.merged_x_labels)),
        ):
            round_start = round_index * round_width
            layout.add(
                family,
                labels_by_family[family][round_start : round_start + round_width],
            )
        if round_index < len(common.STEANE_LS_VERIFICATION_ROUNDS):
            round_width = len(common.STEANE_LS_VERIFICATION_ROUNDS[round_index])
            layout.add(
                "steane_verify",
                labels_by_family["steane_verify"][
                    steane_verify_start : steane_verify_start + round_width
                ],
            )
            steane_verify_start += round_width

    layout.add("steane_x", labels_by_family["steane_x"])
    layout.add("surface_x_recovery", labels_by_family["surface_x_recovery"])
    if include_post_transfer_stabilizer_rounds:
        post_z_labels = labels_by_family["post_transfer_surface_z"]
        post_x_labels = labels_by_family["post_transfer_surface_x"]
        for round_index in range(spec.distance):
            z_start = round_index * len(spec.z_labels)
            x_start = round_index * len(spec.x_labels)
            layout.add(
                "post_transfer_surface_z",
                post_z_labels[z_start : z_start + len(spec.z_labels)],
            )
            layout.add(
                "post_transfer_surface_x",
                post_x_labels[x_start : x_start + len(spec.x_labels)],
            )
    if surface_readout_basis is not None:
        family = f"surface_logical_{surface_readout_basis.lower()}"
        layout.add(family, labels_by_family[family])
    return layout.finish()


# Encoder, QED, cultivation, and Steane-verification helpers emit one known-zero
# detector per measured bit while the front end is built, so both the detector
# count and the annotation consistency check derive from the same families.
_LOCALLY_ANNOTATED_DETECTOR_FAMILIES = ("enc_helpers", "steane_qed", "cult", "steane_verify")


def _num_locally_annotated_detectors(layout: TransferMeasurementLayout) -> int:
    """Return detectors already emitted by encoder/QED/cultivation/verify helpers."""
    record_names = {record.name for record in layout.records}
    return sum(
        len(layout.record(family).positions)
        for family in _LOCALLY_ANNOTATED_DETECTOR_FAMILIES
        if family in record_names
    )


def num_transfer_protocol_postselection_detectors(
    *,
    spec: common.SurfaceCodeTransferSpec,
    layout: TransferMeasurementLayout,
) -> int:
    """Return the number of transfer-local hard-postselection detectors."""
    return (
        _num_locally_annotated_detectors(layout)
        + (common.LATTICE_SURGERY_ROUNDS - 1) * len(spec.merged_boundary_checks)
        + len(common.FINAL_X_VERIFICATION_CHECKS)
    )


# ---------------------------------------------------------------------------
# Detector and observable annotations
# ---------------------------------------------------------------------------


def _merge_round_positions(
    layout: TransferMeasurementLayout, name: str
) -> tuple[tuple[int, ...], ...]:
    """Return per-round position rows for a three-round merged family."""
    return layout.record(name).round_positions(rounds=common.LATTICE_SURGERY_ROUNDS)


def restored_surface_x_reference_positions(
    spec: common.SurfaceCodeTransferSpec,
    layout: TransferMeasurementLayout,
) -> tuple[int, ...]:
    """Return the restored top-edge X reference in ``spec.x_labels`` order.

    Combines the final merged X-check round with the top-edge checks that were
    withheld during merging and recovered after Steane demolition.
    """
    merged_x_final = {
        **dict(
            zip(
                spec.merged_x_labels,
                _merge_round_positions(layout, "merged_surface_x")[-1],
                strict=True,
            )
        ),
        **layout.record("surface_x_recovery").positions_by_label(),
    }
    return tuple(merged_x_final[label] for label in spec.x_labels)


def transfer_logical_frame_positions(
    layout: TransferMeasurementLayout,
    basis: common.SurfaceReadoutBasis,
) -> tuple[int, ...]:
    """Return the Pauli-frame correction records for a transfer logical readout.

    ``X`` uses the demolished Steane logical-X record, ``Z`` uses the final
    boundary-``ZZ`` round, and ``Y`` uses both.
    """
    basis = normalize_pauli_char(basis)
    boundary_positions = _merge_round_positions(layout, "ls_zz")[-1]
    steane_x_positions = layout.record("steane_x").positions_by_label()
    steane_logical_x_positions = tuple(
        steane_x_positions[label] for label in common.HIRANO_STEANE_LOGICAL_X
    )
    match basis:
        case "X":
            return steane_logical_x_positions
        case "Y":
            return boundary_positions + steane_logical_x_positions
        case _:
            return boundary_positions


def append_transfer_detectors(
    circuit: stim.Circuit,
    *,
    spec: common.SurfaceCodeTransferSpec,
    layout: TransferMeasurementLayout,
    first_protocol_detector: int,
) -> TransferDetectorLayout:
    """Annotate transfer postselection checks and merge surface history.

    Encoder, QED, cultivation, and Steane-verification helpers have already
    emitted their local known-zero detectors. This function completes the
    protocol layer with repeated-boundary changes and final reconstructed X
    checks, then appends the three merge-round X/Z syndrome-delta layers.
    ``first_protocol_detector`` preserves the caller's detector offset and
    keeps protocol postselection separate from surface-code correction.
    """
    if circuit.num_measurements != layout.num_measurements:
        raise ValueError("the circuit and transfer measurement layout do not match.")

    if circuit.num_detectors - first_protocol_detector != _num_locally_annotated_detectors(layout):
        raise RuntimeError("transfer-local detector annotations are inconsistent.")

    boundary_rounds = _merge_round_positions(layout, "ls_zz")
    helper.append_stabilizer_detectors(
        circuit,
        rounds=boundary_rounds[1:],
        first_round_reference=boundary_rounds[0],
    )

    final_x_positions = {
        **layout.record("steane_x").positions_by_label(),
        **layout.record("surface_x_recovery").positions_by_label(),
    }
    for support in common.FINAL_X_VERIFICATION_CHECKS.values():
        circuit_stim.append_detector(
            circuit,
            measurement_positions=tuple(final_x_positions[label] for label in support),
        )
    protocol_postselection_indices = tuple(range(first_protocol_detector, circuit.num_detectors))
    if len(protocol_postselection_indices) != (
        num_transfer_protocol_postselection_detectors(spec=spec, layout=layout)
    ):
        raise RuntimeError("transfer protocol detector layout is inconsistent.")

    first_surface_detector = circuit.num_detectors
    helper.append_stabilizer_detectors(
        circuit,
        rounds=_merge_round_positions(layout, "merged_surface_z"),
        first_round_reference=layout.record("surface_init_z").positions,
    )
    helper.append_stabilizer_detectors(
        circuit,
        rounds=_merge_round_positions(layout, "merged_surface_x"),
    )
    surface_correction_indices = tuple(range(first_surface_detector, circuit.num_detectors))
    return TransferDetectorLayout(
        protocol_postselection_indices=protocol_postselection_indices,
        surface_correction_indices=surface_correction_indices,
    )


def append_transfer_post_transfer_surface_detectors(
    circuit: stim.Circuit,
    *,
    spec: common.SurfaceCodeTransferSpec,
    layout: TransferMeasurementLayout,
    z_record_name: str = "post_transfer_surface_z",
    x_record_name: str = "post_transfer_surface_x",
    rounds: int | None = None,
) -> tuple[int, ...]:
    """Annotate full surface rounds after the restored transfer boundary."""
    if rounds is None:
        rounds = spec.distance
    first_detector = circuit.num_detectors
    post_z_rounds = layout.record(z_record_name).round_positions(rounds=rounds)
    helper.append_stabilizer_detectors(
        circuit,
        rounds=post_z_rounds,
        first_round_reference=_merge_round_positions(layout, "merged_surface_z")[-1],
    )

    post_x_rounds = layout.record(x_record_name).round_positions(rounds=rounds)
    helper.append_stabilizer_detectors(
        circuit,
        rounds=post_x_rounds,
        first_round_reference=restored_surface_x_reference_positions(spec, layout),
    )
    return tuple(range(first_detector, circuit.num_detectors))


def append_transfer_logical_observable(
    circuit: stim.Circuit,
    *,
    layout: TransferMeasurementLayout,
    basis: common.SurfaceReadoutBasis,
    observable_index: int = 0,
) -> None:
    """Include a transfer-corrected logical-support readout as an observable."""
    basis = normalize_pauli_char(basis)
    logical_positions = layout.record(f"surface_logical_{basis.lower()}").positions
    circuit_stim.append_observable_include(
        circuit,
        measurement_positions=logical_positions + transfer_logical_frame_positions(layout, basis),
        observable_index=observable_index,
    )


# ---------------------------------------------------------------------------
# Qubit registers and geometry
# ---------------------------------------------------------------------------


def build_transfer_qregs(spec: common.SurfaceCodeTransferSpec) -> HiranoLabeledQubits:
    """Allocate deterministic Stim qubit indices for the full transfer.

    The 11-site Steane patch retains indices ``0..10``. It is followed by
    ``distance**2`` surface data indices, one ancilla per surface Z and X
    stabilizer, and one ancilla per merged-boundary check.
    """
    next_qubit = 11
    surface_data = {label: next_qubit + index for index, label in enumerate(spec.surface_order)}
    next_qubit += len(surface_data)
    surface_z_ancillas = {
        label: next_qubit + index for index, label in enumerate(spec.z_ancilla_labels)
    }
    next_qubit += len(surface_z_ancillas)
    surface_x_ancillas = {
        label: next_qubit + index for index, label in enumerate(spec.x_ancilla_labels)
    }
    next_qubit += len(surface_x_ancillas)
    merge_ancillas = {
        f"zz_{name[1:].lower()}": next_qubit + index
        for index, (name, _, _) in enumerate(spec.merged_boundary_checks)
    }
    return HiranoLabeledQubits(
        steane_data={
            label: common.HIRANO_LABEL_TO_GOOGLE_SITE[label]
            for label in common.HIRANO_STEANE_DATA_ORDER
        },
        steane_helpers={
            label: common.STEANE_HELPER_LABEL_TO_GOOGLE_SITE[label]
            for label in common.STEANE_HELPER_PATCH_LABELS
        },
        surface_data=surface_data,
        surface_z_ancillas=surface_z_ancillas,
        surface_x_ancillas=surface_x_ancillas,
        merge_ancillas=merge_ancillas,
    )


def hirano_labeled_patch_qubits(registers: HiranoLabeledQubits) -> list[int]:
    """Return the 11-site cultivation patch qubits in patch-label order."""
    patch_qubits: list[int | None] = [None] * 11
    for site, label in common.GOOGLE_SITE_TO_HIRANO_LABEL.items():
        patch_qubits[site] = registers.steane_data[label]
    for label, site in common.STEANE_HELPER_LABEL_TO_GOOGLE_SITE.items():
        patch_qubits[site] = registers.steane_helpers[label]
    if any(qubit is None for qubit in patch_qubits):
        raise ValueError("Hirano-labeled registers do not cover all 11 patch sites.")
    return [int(qubit) for qubit in patch_qubits]


def surface_data_qubits(
    registers: HiranoLabeledQubits,
    surface_order: Sequence[str],
) -> list[int]:
    """Return surface data qubits in target-label order."""
    return [registers.surface_data[label] for label in surface_order]


# ---------------------------------------------------------------------------
# Measurement-round appenders
# ---------------------------------------------------------------------------


def append_boundary_zz_round(
    circuit: stim.Circuit,
    *,
    registers: HiranoLabeledQubits,
    merged_boundary_checks: Sequence[common.MergedBoundaryCheck],
    add_ticks: bool = True,
) -> None:
    """Append one round of merged-boundary ``ZZ`` checks.

    Each check may contain Steane/color-code labels, surface-code labels, or
    both. The d=5 paper schedule uses ``Z1a``, ``Z35bc``, and ``Zde``. Their
    parity is logical ``Z_L Z_L``; a nonzero accepted value becomes an X
    byproduct in the target Pauli frame. The builder repeats this round three
    times so disagreement can be detected.
    """
    ancillas = tuple(
        registers.merge_ancillas[f"zz_{name[1:].lower()}"] for name, _, _ in merged_boundary_checks
    )
    if ancillas:
        circuit.append("R", ancillas)
        if add_ticks:
            circuit.append("TICK", [])

    supports = tuple(
        tuple(registers.steane_data[label] for label in steane_labels)
        + tuple(registers.surface_data[label] for label in surface_labels)
        for _, steane_labels, surface_labels in merged_boundary_checks
    )
    max_weight = max((len(support) for support in supports), default=0)
    for support_index in range(max_weight):
        cx_targets: list[int] = []
        for support, ancilla in zip(supports, ancillas, strict=True):
            if support_index < len(support):
                cx_targets.extend((support[support_index], ancilla))
        if cx_targets:
            circuit.append("CX", cx_targets)
        if add_ticks:
            circuit.append("TICK", [])
    if ancillas:
        circuit.append("M", ancillas)
    if add_ticks:
        circuit.append("TICK", [])


def append_measure_steane_data_in_x_basis(
    circuit: stim.Circuit,
    *,
    registers: HiranoLabeledQubits,
    add_ticks: bool = True,
) -> None:
    """Demolish the Steane block by measuring all seven data qubits in X."""
    steane.append_patch_logical_measurement(
        circuit,
        "X",
        patch_qubits=hirano_labeled_patch_qubits(registers),
        support=tuple(
            common.HIRANO_LABEL_TO_GOOGLE_SITE[label] for label in common.HIRANO_STEANE_DATA_ORDER
        ),
        destructive=True,
    )
    if add_ticks:
        circuit.append("TICK", [])


def append_steane_lattice_surgery_verification_round(
    circuit: stim.Circuit,
    *,
    registers: HiranoLabeledQubits,
    round_index: int,
    add_ticks: bool = True,
) -> None:
    """Measure one paper-defined Steane check group during a merged round.

    Each commuting stabilizer uses the shared Steane measurement helper. The
    measured observables and their round ordering match the Qiskit model; this
    modular circuit does not reproduce the paper's hardware-level superdense
    gate interleaving.
    """
    if not 0 <= round_index < len(common.STEANE_LS_VERIFICATION_ROUNDS):
        raise ValueError(
            "round_index must be in " f"[0, {len(common.STEANE_LS_VERIFICATION_ROUNDS) - 1}]."
        )
    patch_qubits = hirano_labeled_patch_qubits(registers)
    for check_name in common.STEANE_LS_VERIFICATION_ROUNDS[round_index]:
        steane.append_patch_stabilizer_measurement(
            circuit,
            patch_qubits=patch_qubits,
            basis=check_name[0].lower(),
            stabilizer_indices=(common.STEANE_LS_VERIFICATION_STABILIZER_INDICES[check_name],),
            add_ticks=False,
        )
    if add_ticks:
        circuit.append("TICK", [])


def append_merged_surface_syndrome_round(
    circuit: stim.Circuit,
    *,
    spec: common.SurfaceCodeTransferSpec,
    registers: HiranoLabeledQubits,
    surface_qubits: Sequence[int],
    round_label: str,
    add_ticks: bool = True,
) -> None:
    """Measure all merge-compatible surface checks in one four-step round.

    Every ordinary Z check is interleaved with compatible X checks under their
    shared canonical schedule. Top-edge X checks in
    ``spec.top_x_recovery_labels`` anticommute with the boundary-Z checks, so
    they are withheld until after Steane demolition.
    """
    spec.surface_code_operations.append_scheduled_stabilizer_round(
        circuit,
        surface_qubits=surface_qubits,
        selected_stabilizer_names=spec.z_labels + spec.merged_x_labels,
        ancillas=[registers.surface_z_ancillas[label.lower()] for label in spec.z_labels]
        + [registers.surface_x_ancillas[label.lower()] for label in spec.merged_x_labels],
        round_label=f"{round_label} compatible surface checks",
        reset_ancillas_initially=True,
        add_ticks=add_ticks,
    )


def append_surface_top_x_recovery_measurements(
    circuit: stim.Circuit,
    *,
    spec: common.SurfaceCodeTransferSpec,
    registers: HiranoLabeledQubits,
    add_ticks: bool = True,
) -> None:
    """Recover surface top-edge X checks in parallel after Steane demolition."""
    spec.surface_code_operations.append_scheduled_stabilizer_round(
        circuit,
        surface_qubits=surface_data_qubits(registers, spec.surface_order),
        selected_stabilizer_names=spec.top_x_recovery_labels,
        ancillas=[
            registers.surface_x_ancillas[label.lower()] for label in spec.top_x_recovery_labels
        ],
        round_label="recover surface top edge",
        reset_ancillas_initially=False,
        add_ticks=add_ticks,
    )


# ---------------------------------------------------------------------------
# Circuit builders
# ---------------------------------------------------------------------------


def append_lattice_surgery_circuit(
    circuit: stim.Circuit,
    *,
    spec: common.SurfaceCodeTransferSpec,
    registers: HiranoLabeledQubits,
    injected_state: steane.StimInjectedStateName = steane.DEFAULT_STIM_INJECTED_STATE,
    include_cultivation: bool = True,
    include_post_transfer_stabilizer_rounds: bool = False,
    surface_readout_basis: common.SurfaceReadoutBasis | None = None,
    add_ticks: bool = True,
) -> None:
    """Append the complete Clifford-proxy transfer to caller-owned qubits.

    The fixed-patch sequence follows Section III A of Hirano et al. and is
    specialized by ``spec`` to the rotated target:

    1. Encode ``injected_state`` with the routed 11-site Crumble encoder and
       measure encoder-helper acceptance bits.
    2. Measure every Steane X and Z stabilizer for post-injection QED.
    3. Optionally apply the logical-Y Stim proxy for H_XY cultivation.
    4. Prepare the target in product ``|+>`` and measure all Z stabilizers,
       projecting logical ``|+_L>`` and establishing its Z reference.
    5. Perform exactly three merged rounds. Each measures boundary-Z checks,
       compatible surface Z/X checks, and the paper-defined Steane groups in
       rounds one and two.
    6. Demolish Steane data in X and recover withheld top-edge X checks.
    7. Optionally append ``distance`` full surface-stabilizer rounds.
    8. Optionally append a destructive surface logical readout.

    ``injected_state`` defaults to Stim's Clifford ``S|+>`` proxy. Disable
    cultivation only for transfer plumbing, stabilizer-basis channel tests, or
    an already-cultivated input. ``surface_readout_basis=None`` ends at the
    transfer boundary; an X, Y, or Z readout is an analysis addition.

    The merged surface measurements form temporal decoder input after
    execution. Boundary changes and Steane checks are strict postselection;
    surface syndromes are decoded. Corrections remain in a classical Pauli
    frame. The modular helper preserves measured observables and round order,
    but does not implement the paper's superdense interleaving, its exact
    circuit-level detector error model, later code expansion, or
    complementary-gap lookup.
    Protocol and surface-history detectors are always appended. A corrected
    logical observable is included whenever readout is requested. ``add_ticks``
    controls conceptual layer and stage separators.
    """
    measurement_offset = circuit.num_measurements
    first_protocol_detector = circuit.num_detectors
    patch_qubits = hirano_labeled_patch_qubits(registers)
    cultivation.append_steane_injection_and_cultivation(
        circuit,
        patch_qubits=patch_qubits,
        injected_state=injected_state,
        cultivation_rounds=(1 if include_cultivation else 0),
        measure_encoder_flags=True,
        add_qed_round=True,
        add_ticks=add_ticks,
        skip_cx_1_4=False,
    )

    surface_qubits = surface_data_qubits(registers, spec.surface_order)
    spec.surface_code_operations.prepare_product_plus(
        circuit,
        qubits=surface_qubits,
        add_ticks=add_ticks,
        tick_label=f"prepare {spec.name} |+> product state",
    )
    spec.surface_code_operations.append_scheduled_stabilizer_round(
        circuit,
        surface_qubits=surface_qubits,
        selected_stabilizer_names=spec.z_labels,
        ancillas=[registers.surface_z_ancillas[label.lower()] for label in spec.z_labels],
        round_label=f"initialize {spec.name} |+_L>",
        reset_ancillas_initially=True,
        add_ticks=add_ticks,
    )

    for round_index in range(common.LATTICE_SURGERY_ROUNDS):
        append_boundary_zz_round(
            circuit,
            registers=registers,
            merged_boundary_checks=spec.merged_boundary_checks,
            add_ticks=add_ticks,
        )
        append_merged_surface_syndrome_round(
            circuit,
            spec=spec,
            registers=registers,
            surface_qubits=surface_qubits,
            round_label=f"merged interval round {round_index + 1}",
            add_ticks=add_ticks,
        )
        if round_index < len(common.STEANE_LS_VERIFICATION_ROUNDS):
            append_steane_lattice_surgery_verification_round(
                circuit,
                registers=registers,
                round_index=round_index,
                add_ticks=add_ticks,
            )

    append_measure_steane_data_in_x_basis(circuit, registers=registers, add_ticks=add_ticks)
    append_surface_top_x_recovery_measurements(
        circuit,
        spec=spec,
        registers=registers,
        add_ticks=add_ticks,
    )
    post_transfer_stim.append_rotated_post_transfer(
        circuit,
        surface_qubits=surface_qubits,
        z_ancillas=[registers.surface_z_ancillas[label.lower()] for label in spec.z_labels],
        x_ancillas=[registers.surface_x_ancillas[label.lower()] for label in spec.x_labels],
        z_stabilizer_names=spec.z_labels,
        x_stabilizer_names=spec.x_labels,
        stabilizer_rounds=(spec.distance if include_post_transfer_stabilizer_rounds else 0),
        append_scheduled_stabilizer_round=(
            spec.surface_code_operations.append_scheduled_stabilizer_round
        ),
        logical_readout_basis=surface_readout_basis,
        append_logical_measurement=spec.surface_code_operations.append_logical_measurement,
        add_ticks=add_ticks,
    )
    layout = transfer_measurement_layout(
        spec,
        include_cultivation=include_cultivation,
        include_post_transfer_stabilizer_rounds=(include_post_transfer_stabilizer_rounds),
        surface_readout_basis=surface_readout_basis,
    ).shifted(measurement_offset)
    append_transfer_detectors(
        circuit,
        spec=spec,
        layout=layout,
        first_protocol_detector=first_protocol_detector,
    )
    if include_post_transfer_stabilizer_rounds:
        append_transfer_post_transfer_surface_detectors(
            circuit,
            spec=spec,
            layout=layout,
        )
    if surface_readout_basis is not None:
        append_transfer_logical_observable(
            circuit,
            layout=layout,
            basis=surface_readout_basis,
        )


def _append_transfer_qubit_coords(
    circuit: stim.Circuit,
    *,
    spec: common.SurfaceCodeTransferSpec,
    registers: HiranoLabeledQubits,
) -> None:
    """Annotate the transfer's conceptual Steane and surface geometry."""
    patch_coords = (
        (2.0, 0.0),
        (1.0, 1.0),
        (3.0, 1.0),
        (0.0, 2.0),
        (2.0, 2.0),
        (4.0, 2.0),
        (1.0, 3.0),
        (3.0, 3.0),
        (0.0, 4.0),
        (2.0, 4.0),
        (4.0, 4.0),
    )
    for qubit, coords in enumerate(patch_coords):
        circuit.append("QUBIT_COORDS", [qubit], coords)
    for surface_index, label in enumerate(spec.surface_order):
        circuit.append(
            "QUBIT_COORDS",
            [registers.surface_data[label]],
            (7.0 + surface_index % spec.distance, float(surface_index // spec.distance)),
        )


def build_lattice_surgery_circuit(
    spec: common.SurfaceCodeTransferSpec,
    *,
    injected_state: steane.StimInjectedStateName = steane.DEFAULT_STIM_INJECTED_STATE,
    include_cultivation: bool = True,
    include_post_transfer_stabilizer_rounds: bool = False,
    surface_readout_basis: common.SurfaceReadoutBasis | None = None,
    add_ticks: bool = True,
) -> stim.Circuit:
    """Build the authoritative register-owning Stim transfer circuit.

    Qubit allocation consists of 11 Steane patch sites, ``distance**2`` target
    data sites, one ancilla per target Z and X stabilizer, and one ancilla per
    merged-boundary check. The measurement record contains encoder helpers,
    six Steane QED bits, optional five-bit cultivation, surface initialization,
    all three interleaved merge rounds, eight Steane verification results,
    seven demolition bits, top-edge recovery, optional distance full surface
    rounds, and optional logical readout. ``transfer_measurement_layout`` is
    checked against the circuit as the authoritative record map.
    """
    if surface_readout_basis is not None:
        surface_readout_basis = normalize_pauli_char(surface_readout_basis)
    registers = build_transfer_qregs(spec)
    circuit = stim.Circuit()
    _append_transfer_qubit_coords(circuit, spec=spec, registers=registers)
    append_lattice_surgery_circuit(
        circuit,
        spec=spec,
        registers=registers,
        injected_state=injected_state,
        include_cultivation=include_cultivation,
        include_post_transfer_stabilizer_rounds=include_post_transfer_stabilizer_rounds,
        surface_readout_basis=surface_readout_basis,
        add_ticks=add_ticks,
    )
    layout = transfer_measurement_layout(
        spec,
        include_cultivation=include_cultivation,
        include_post_transfer_stabilizer_rounds=include_post_transfer_stabilizer_rounds,
        surface_readout_basis=surface_readout_basis,
    )
    if circuit.num_measurements != layout.num_measurements:
        raise RuntimeError("Transfer measurement layout does not match the Stim circuit.")
    return circuit
