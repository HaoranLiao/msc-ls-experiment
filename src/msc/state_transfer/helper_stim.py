from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import zip_longest

import stim

from msc.rotated_surface_code import helper_stim as surface_helper
from msc.rotated_surface_code.post_transfer import post_transfer_stim
from msc.state_transfer import common
from msc.steane_code import magic_state_cultivation_stim as cultivation
from msc.steane_code import state_injection_stim as steane
from msc.utils import circuit_stim
from msc.utils.normalize import normalize_pauli_char

_STEANE_POSTSELECTION_DETECTOR_FAMILIES = (
    "enc_flags",
    "steane_qed",
    "cult",
    "steane_verify",
    "steane_bell_partner_flags",
)


# =============================================================================
# Semantic qubits and measurement record
# =============================================================================


def transfer_qubits(
    spec: common.SurfaceCodeTransferSpec,
    *,
    patch_sites: common.SteanePatchSites = 14,
) -> dict[str, int]:
    """Allocate one paper-labeled Stim index for every transfer qubit.

    The selected Steane patch retains indices ``0..patch_sites-1``. Surface data
    follow in target order, then surface Z ancillas, surface X ancillas, and one
    ancilla per interface check.
    """
    patch_spec = common.resolve_steane_transfer_patch_spec(patch_sites)
    qubits = {label: site for site, label in enumerate(patch_spec.site_labels)}
    next_qubit = len(patch_spec.site_labels)
    for label in (
        *spec.surface_order,
        *(name.lower() for name in spec.z_labels),
        *(name.lower() for name in spec.x_labels),
        *(f"zz_{name[1:].lower()}" for name, _, _ in spec.merged_boundary_checks),
    ):
        qubits[label] = next_qubit
        next_qubit += 1
    return qubits


@dataclass(frozen=True)
class TransferMeasurementLayout:
    """Relate semantic measurement families to Stim's global record."""

    patch_sites: common.SteanePatchSites
    labels: dict[str, tuple[str, ...]]
    positions: dict[str, tuple[int, ...]]
    num_measurements: int

    def rounds(
        self,
        name: str,
        rounds: int = common.LATTICE_SURGERY_ROUNDS,
    ) -> tuple[tuple[int, ...], ...]:
        """Split one repeated family into equal round-major position rows."""
        positions = self.positions[name]
        if rounds <= 0 or len(positions) % rounds:
            raise ValueError(f"{name} cannot be divided into {rounds} rounds.")
        width = len(positions) // rounds
        return tuple(positions[start : start + width] for start in range(0, len(positions), width))

    def by_label(self, name: str) -> dict[str, int]:
        """Return one family's global positions keyed by semantic label."""
        return dict(zip(self.labels[name], self.positions[name], strict=True))

    def extract(self, shot: Sequence[bool | int]) -> dict[str, dict[str, int]]:
        """Extract every family as labeled bits from one measurement row."""
        return {
            name: {
                label: int(bool(shot[position]))
                for label, position in zip(labels, self.positions[name], strict=True)
            }
            for name, labels in self.labels.items()
        }

    def appended(
        self,
        records: Sequence[tuple[str, Sequence[str]]],
    ) -> TransferMeasurementLayout:
        """Return a layout extended by contiguous appended families."""
        labels = dict(self.labels)
        positions = dict(self.positions)
        next_position = self.num_measurements
        for name, family_labels in records:
            if name in labels:
                raise ValueError(f"duplicate measurement family {name!r}.")
            labels[name] = tuple(family_labels)
            positions[name] = tuple(range(next_position, next_position + len(family_labels)))
            next_position += len(family_labels)
        return TransferMeasurementLayout(
            patch_sites=self.patch_sites,
            labels=labels,
            positions=positions,
            num_measurements=next_position,
        )


def transfer_measurement_layout(
    spec: common.SurfaceCodeTransferSpec,
    *,
    patch_sites: common.SteanePatchSites = 14,
    include_cultivation: bool,
    surface_readout_basis: common.SurfaceReadoutBasis | None,
    include_post_transfer_stabilizer_rounds: bool = False,
) -> TransferMeasurementLayout:
    """Map shared measurement labels into Stim's physical append order.

    The shared schema groups measurements by meaning, whereas Stim stores one
    global record in circuit order. This layout explicitly interleaves each
    round as interface checks -> compatible surface checks -> color-code
    checks, then places demolition, top-edge recovery, optional continuation,
    and logical readout in their actual append order.
    """
    labels = common.transfer_measurement_labels(
        spec,
        patch_sites=patch_sites,
        include_cultivation=include_cultivation,
        surface_readout_basis=surface_readout_basis,
        include_post_transfer_stabilizer_rounds=include_post_transfer_stabilizer_rounds,
    )
    # Stim physically measures the known +1 surface-X checks before merging so
    # their circuit faults and first detector layer remain in the DEM.
    labels["surface_init_x"] = spec.x_labels
    measurement_chunks: list[tuple[str, tuple[str, ...]]] = [
        (name, labels[name]) for name in ("enc_flags", "steane_qed")
    ]
    if include_cultivation:
        measurement_chunks.append(("cult", labels["cult"]))
    measurement_chunks.extend((name, labels[name]) for name in ("surface_init_z", "surface_init_x"))

    cursors = {name: 0 for name in labels}
    for name, chunk in measurement_chunks:
        cursors[name] += len(chunk)

    def take_measurement_labels(name: str, width: int) -> tuple[str, ...]:
        start = cursors[name]
        cursors[name] += width
        return labels[name][start : start + width]

    for round_index in range(common.LATTICE_SURGERY_ROUNDS):
        measurement_chunks.extend(
            (
                (
                    "interface_z",
                    take_measurement_labels("interface_z", len(spec.merged_boundary_checks)),
                ),
                (
                    "merged_surface_z",
                    take_measurement_labels("merged_surface_z", len(spec.z_labels)),
                ),
                (
                    "merged_surface_x",
                    take_measurement_labels("merged_surface_x", len(spec.merged_x_labels)),
                ),
            )
        )
        verification_checks = common.STEANE_MERGE_VERIFICATION_ROUNDS[round_index]
        for check_name in verification_checks:
            verification_chunk = (
                "steane_verify",
                take_measurement_labels("steane_verify", 1),
            )
            if patch_sites == 11:
                measurement_chunks.append(verification_chunk)
            elif patch_sites == 14:
                bell_partner_flag_chunk = (
                    "steane_bell_partner_flags",
                    take_measurement_labels("steane_bell_partner_flags", 1),
                )
                if check_name.startswith("X"):
                    measurement_chunks.extend((verification_chunk, bell_partner_flag_chunk))
                elif check_name.startswith("Z"):
                    measurement_chunks.extend((bell_partner_flag_chunk, verification_chunk))
                else:
                    raise ValueError(f"unsupported Steane check {check_name!r}.")
            else:
                raise ValueError("patch_sites must be 11 or 14.")

    measurement_chunks.extend(
        (name, take_measurement_labels(name, len(labels[name])))
        for name in ("steane_x", "surface_x_recovery")
    )
    if include_post_transfer_stabilizer_rounds:
        for _ in range(spec.distance):
            measurement_chunks.extend(
                (
                    (
                        "post_transfer_surface_z",
                        take_measurement_labels("post_transfer_surface_z", len(spec.z_labels)),
                    ),
                    (
                        "post_transfer_surface_x",
                        take_measurement_labels("post_transfer_surface_x", len(spec.x_labels)),
                    ),
                )
            )
    if surface_readout_basis is not None:
        name = f"surface_logical_{normalize_pauli_char(surface_readout_basis).lower()}"
        measurement_chunks.append((name, take_measurement_labels(name, len(labels[name]))))

    family_labels: dict[str, list[str]] = {}
    family_positions: dict[str, list[int]] = {}
    next_position = 0
    for name, chunk in measurement_chunks:
        family_labels.setdefault(name, []).extend(chunk)
        family_positions.setdefault(name, []).extend(
            range(next_position, next_position + len(chunk))
        )
        next_position += len(chunk)
    if any(tuple(family_labels[name]) != expected for name, expected in labels.items()):
        raise RuntimeError("Stim measurement order does not cover the shared schema.")
    return TransferMeasurementLayout(
        patch_sites=patch_sites,
        labels=labels,
        positions={name: tuple(positions) for name, positions in family_positions.items()},
        num_measurements=next_position,
    )


# =============================================================================
# Circuit construction
# =============================================================================


def build_lattice_surgery_circuit(
    spec: common.SurfaceCodeTransferSpec,
    *,
    patch_sites: common.SteanePatchSites = 14,
    injected_state: steane.StimInjectedStateName = steane.DEFAULT_STIM_INJECTED_STATE,
    include_cultivation: bool = True,
    include_post_transfer_stabilizer_rounds: bool = False,
    surface_readout_basis: common.SurfaceReadoutBasis | None = None,
    add_ticks: bool = True,
) -> stim.Circuit:
    """Build the detector-annotated logical-Y-proxy transfer circuit.

    The fixed-patch sequence follows Section III A of Hirano et al. and is
    specialized by ``spec`` to the rotated target:

    1. Encode ``injected_state`` on the selected Steane patch and measure its
       encoder-flag acceptance bits. The default 14-site path uses the
       abstract-routed encoder; the 11-site path uses Rosenfeld's encoder.
    2. Measure every Steane X and Z stabilizer for post-injection QED.
    3. Optionally apply logical-Y cultivation, the Stim proxy for H_XY
       cultivation.
    4. Prepare the target in product ``|+>`` and measure a complete X/Z
       stabilizer round. Z checks project logical ``|+_L>`` and establish its
       Z reference; X checks are compared with their known +1 values.
    5. Perform exactly three merged rounds. Each measures interface-Z checks,
       compatible surface Z/X checks, and the paper-defined Steane groups in
       rounds one and two. On the 14-site patch, every selective check also
       measures the opposite-basis Bell partner as a known-zero fault flag.
    6. Demolish Steane data in X and recover withheld top-edge X checks.
    7. Optionally append ``distance`` full surface-stabilizer rounds.
    8. Optionally append a destructive surface logical readout.

    ``injected_state`` defaults to Stim's Clifford ``S|+>`` proxy. Disable
    cultivation only for transfer plumbing, stabilizer-basis channel tests, or
    an already-cultivated input. ``surface_readout_basis=None`` ends at the
    transfer boundary; an X, Y, or Z readout is an analysis addition.

    Each merge round appends interface checks -> compatible surface checks ->
    color-code verification. The 14-site Bell-partner flags close a two-fault path
    that is invisible if a selective superdense check reads only its requested
    Bell ancilla, retaining graphlike distance three in this serialized
    schedule. Mixed checks use Hirano's hook-safe alternating paths, especially
    ``3 -> b -> 5 -> c`` for ``Z35bc``. The merged surface measurements form
    spacetime-decoder input after execution. Interface changes, Steane checks,
    and Bell-partner flags are strict postselection; surface syndromes are decoded,
    and corrections remain in a classical Pauli frame.

    Compatible X checks use the known +1 product-preparation boundary;
    nondeterministic Z checks use the pre-merge initialization results as their
    reference. Both sectors therefore contribute three detector layers.
    Timelike matching edges distinguish repeated measurement faults from
    persistent data errors.

    The modular helper preserves measured observables and round order, but it
    does not implement the paper's superdense interleaving, exact circuit-level
    detector error model, later code expansion, or complementary-gap lookup.
    Protocol and surface-history detectors are always appended; a corrected
    logical observable is included whenever readout is requested.
    """
    if surface_readout_basis is not None:
        surface_readout_basis = normalize_pauli_char(surface_readout_basis)
    patch_spec = common.resolve_steane_transfer_patch_spec(patch_sites)
    qubits = transfer_qubits(spec, patch_sites=patch_sites)
    steane_patch = [qubits[label] for label in patch_spec.site_labels]
    surface_qubits = [qubits[label] for label in spec.surface_order]
    circuit = stim.Circuit()
    for index, label in enumerate(spec.surface_order):
        circuit.append(
            "QUBIT_COORDS",
            [qubits[label]],
            (7 + index % spec.distance, index // spec.distance),
        )

    # 1. Accepted input: injection -> QED -> optional logical-Y cultivation (H_XY proxy).
    if patch_sites == 11:
        cultivation.append_11_site_patch_injection_and_cultivation(
            circuit,
            patch_qubits=steane_patch,
            injected_state=injected_state,
            cultivation_rounds=int(include_cultivation),
            measure_encoder_flags=True,
            add_qed_round=True,
            add_ticks=add_ticks,
            skip_cx_1_4=False,
        )
        append_steane_logical_measurement = steane.append_11_site_patch_logical_measurement
    elif patch_sites == 14:
        cultivation.append_14_site_patch_injection_and_cultivation(
            circuit,
            patch_qubits=steane_patch,
            injected_state=injected_state,
            encoder="abstract_routed",
            cultivation_rounds=int(include_cultivation),
            measure_encoder_flags=True,
            add_qed_round=True,
            add_ticks=add_ticks,
        )
        append_steane_logical_measurement = steane.append_14_site_patch_logical_measurement
    else:
        raise ValueError("patch_sites must be 11 or 14.")

    # 2. Prepare surface |+_L>. Product |+> fixes every X check to +1; measure
    # them physically for Hirano's initial detector layer. The simultaneous Z
    # checks project into the code space and establish their reference.
    operations = spec.surface_code_operations
    operations.prepare_product_plus(
        circuit,
        qubits=surface_qubits,
        add_ticks=add_ticks,
        tick_label=f"prepare {spec.name} |+> product state",
    )
    operations.append_scheduled_stabilizer_round(
        circuit,
        surface_qubits=surface_qubits,
        selected_stabilizer_names=spec.z_labels + spec.x_labels,
        ancillas=[qubits[label.lower()] for label in spec.z_labels + spec.x_labels],
        round_label=f"initialize {spec.name} |+_L>",
        reset_ancillas_initially=True,
        add_ticks=add_ticks,
    )

    # 3. Three merged rounds in the observable order used by Hirano.
    for round_index in range(common.LATTICE_SURGERY_ROUNDS):
        append_interface_z_round(
            circuit,
            qubits=qubits,
            merged_boundary_checks=spec.merged_boundary_checks,
            add_ticks=add_ticks,
        )
        operations.append_scheduled_stabilizer_round(
            circuit,
            surface_qubits=surface_qubits,
            selected_stabilizer_names=spec.z_labels + spec.merged_x_labels,
            ancillas=[qubits[label.lower()] for label in spec.z_labels + spec.merged_x_labels],
            round_label=f"merged interval round {round_index + 1}",
            reset_ancillas_initially=True,
            add_ticks=add_ticks,
        )
        verification_checks = common.STEANE_MERGE_VERIFICATION_ROUNDS[round_index]
        for check_name in verification_checks:
            if patch_sites == 11:
                steane.append_11_site_patch_stabilizer_measurement(
                    circuit,
                    patch_qubits=steane_patch,
                    basis=check_name[0],
                    stabilizer_indices=(
                        patch_spec.merge_verification_stabilizer_indices[check_name],
                    ),
                    add_ticks=add_ticks,
                )
            elif patch_sites == 14:
                steane.append_14_site_patch_superdense_stabilizer_measurement(
                    circuit,
                    patch_qubits=steane_patch,
                    basis=check_name[0],
                    stabilizer_indices=(
                        patch_spec.merge_verification_stabilizer_indices[check_name],
                    ),
                    add_ticks=add_ticks,
                )
            else:
                raise ValueError("patch_sites must be 11 or 14.")

    # 4. Demolish Steane in X and restore the withheld surface-X boundary.
    append_steane_logical_measurement(
        circuit,
        "X",
        patch_qubits=steane_patch,
        support=patch_spec.data_sites_in_hirano_order,
        destructive=True,
    )
    if add_ticks:
        circuit.append("TICK", [])
    operations.append_scheduled_stabilizer_round(
        circuit,
        surface_qubits=surface_qubits,
        selected_stabilizer_names=spec.top_x_recovery_labels,
        ancillas=[qubits[label.lower()] for label in spec.top_x_recovery_labels],
        round_label="recover surface top edge",
        reset_ancillas_initially=False,
        add_ticks=add_ticks,
    )

    # 5. Optional fixed-patch history and terminal logical-support readout.
    post_transfer_stim.append_rotated_post_transfer(
        circuit,
        surface_qubits=surface_qubits,
        z_ancillas=[qubits[label.lower()] for label in spec.z_labels],
        x_ancillas=[qubits[label.lower()] for label in spec.x_labels],
        z_stabilizer_names=spec.z_labels,
        x_stabilizer_names=spec.x_labels,
        stabilizer_rounds=spec.distance if include_post_transfer_stabilizer_rounds else 0,
        append_scheduled_stabilizer_round=operations.append_scheduled_stabilizer_round,
        logical_readout_basis=surface_readout_basis,
        append_logical_measurement=operations.append_logical_measurement,
        add_ticks=add_ticks,
    )

    layout = transfer_measurement_layout(
        spec,
        patch_sites=patch_sites,
        include_cultivation=include_cultivation,
        include_post_transfer_stabilizer_rounds=include_post_transfer_stabilizer_rounds,
        surface_readout_basis=surface_readout_basis,
    )

    # 6. Hard-postselect color/interface checks, then retain surface history for decoding.
    local_detector_count = sum(
        len(layout.positions[name])
        for name in _STEANE_POSTSELECTION_DETECTOR_FAMILIES
        if name in layout.positions
    )
    if circuit.num_detectors != local_detector_count:
        raise RuntimeError("transfer-local detector annotations are inconsistent.")
    interface_rounds = layout.rounds("interface_z")
    surface_helper.append_stabilizer_detectors(
        circuit,
        rounds=interface_rounds[1:],
        first_round_reference=interface_rounds[0],
    )
    final_x_positions = layout.by_label("steane_x") | layout.by_label("surface_x_recovery")
    for support in common.FINAL_X_VERIFICATION_CHECKS.values():
        circuit_stim.append_detector(
            circuit,
            measurement_positions=tuple(final_x_positions[label] for label in support),
        )
    if circuit.num_detectors != num_transfer_protocol_postselection_detectors(layout):
        raise RuntimeError("transfer protocol detector layout is inconsistent.")

    surface_helper.append_stabilizer_detectors(
        circuit,
        rounds=(layout.positions["surface_init_x"],),
    )
    merged_z_rounds = layout.rounds("merged_surface_z")
    surface_helper.append_stabilizer_detectors(
        circuit,
        rounds=merged_z_rounds,
        first_round_reference=layout.positions["surface_init_z"],
    )
    surface_helper.append_stabilizer_detectors(
        circuit,
        rounds=layout.rounds("merged_surface_x"),
        first_round_reference=tuple(
            layout.by_label("surface_init_x")[label] for label in spec.merged_x_labels
        ),
    )
    if include_post_transfer_stabilizer_rounds:
        surface_helper.append_stabilizer_detectors(
            circuit,
            rounds=layout.rounds("post_transfer_surface_z", spec.distance),
            first_round_reference=merged_z_rounds[-1],
        )
        surface_helper.append_stabilizer_detectors(
            circuit,
            rounds=layout.rounds("post_transfer_surface_x", spec.distance),
            first_round_reference=restored_surface_x_reference_positions(spec, layout),
        )
    if surface_readout_basis is not None:
        append_transfer_logical_observable(circuit, layout=layout, basis=surface_readout_basis)
    if circuit.num_measurements != layout.num_measurements:
        raise RuntimeError("transfer measurement layout does not match the Stim circuit.")
    return circuit


def append_interface_z_round(
    circuit: stim.Circuit,
    *,
    qubits: Mapping[str, int],
    merged_boundary_checks: Sequence[common.MergedBoundaryCheck],
    add_ticks: bool = True,
) -> None:
    """Measure one interface-Z round using hook-safe alternating supports.

    The current Stim target is d=5 and measures three checks:

    - ``Z1a = Z_1 Z_a``, with CNOT support order ``1 -> a``;
    - ``Z35bc = Z_3 Z_5 Z_b Z_c``, ordered ``3 -> b -> 5 -> c``;
    - ``Zde = Z_d Z_e``, ordered ``d -> e``.

    Their product is the Steane/surface logical ``Z_L Z_L`` operator. The
    alternating mixed-code order for ``Z35bc`` avoids a damaging hook error.
    Three repeated rounds detect any interface-outcome change; an accepted
    nonzero product is an X byproduct in the target Pauli frame.
    """
    ancillas = tuple(qubits[f"zz_{name[1:].lower()}"] for name, _, _ in merged_boundary_checks)
    supports = tuple(
        tuple(
            qubits[label]
            for pair in zip_longest(steane_labels, surface_labels)
            for label in pair
            if label is not None
        )
        for _, steane_labels, surface_labels in merged_boundary_checks
    )
    circuit.append("R", ancillas)
    if add_ticks:
        circuit.append("TICK", [])
    for step in range(max(map(len, supports), default=0)):
        targets = [
            qubit
            for support, ancilla in zip(supports, ancillas, strict=True)
            if step < len(support)
            for qubit in (support[step], ancilla)
        ]
        circuit.append("CX", targets)
        if add_ticks:
            circuit.append("TICK", [])
    circuit.append("M", ancillas)
    if add_ticks:
        circuit.append("TICK", [])


# =============================================================================
# Detector and observable annotations
# =============================================================================


def num_transfer_protocol_postselection_detectors(layout: TransferMeasurementLayout) -> int:
    """Return the hard-postselection prefix length before surface history.

    The prefix contains Steane/color-code checks, changes between repeated
    interface rounds, and the three final X checks reconstructed at demolition.
    """
    steane_detector_count = sum(
        len(layout.positions[name])
        for name in _STEANE_POSTSELECTION_DETECTOR_FAMILIES
        if name in layout.positions
    )
    interface_width = len(layout.rounds("interface_z")[0])
    return (
        steane_detector_count
        + (common.LATTICE_SURGERY_ROUNDS - 1) * interface_width
        + len(common.FINAL_X_VERIFICATION_CHECKS)
    )


def append_transfer_logical_observable(
    circuit: stim.Circuit,
    *,
    layout: TransferMeasurementLayout,
    basis: common.SurfaceReadoutBasis,
    observable_index: int = 0,
) -> None:
    """Include a logical-support readout corrected by transfer-frame records.

    Stim observables combine the destructive support parity with the accepted
    interface and Steane-demolition outcomes that define its Pauli byproducts.
    Explicitly, the included measurement-record parities are:

    - X: surface logical-X readout plus Steane demolition ``X146``;
    - Z: surface logical-Z readout plus the final ``Z1a``, ``Z35bc``, and
      ``Zde`` interface outcomes;
    - Y: surface logical-Y readout plus all the X- and Z-correction records
      listed above.

    Here “plus” means addition modulo two in one Stim ``OBSERVABLE_INCLUDE``.
    """
    basis = normalize_pauli_char(basis)
    circuit_stim.append_observable_include(
        circuit,
        measurement_positions=(
            layout.positions[f"surface_logical_{basis.lower()}"]
            + transfer_logical_frame_positions(layout, basis)
        ),
        observable_index=observable_index,
    )


def restored_surface_x_reference_positions(
    spec: common.SurfaceCodeTransferSpec,
    layout: TransferMeasurementLayout,
) -> tuple[int, ...]:
    """Return the completed surface-X reference in canonical label order.

    Compatible X checks come from the final merge round. The top-edge checks
    that anticommute with interface Z are measured only after Steane demolition;
    those values fill the remaining reference positions for later full rounds.
    """
    positions = dict(
        zip(
            spec.merged_x_labels,
            layout.rounds("merged_surface_x")[-1],
            strict=True,
        )
    )
    positions.update(layout.by_label("surface_x_recovery"))
    return tuple(positions[label] for label in spec.x_labels)


def transfer_logical_frame_positions(
    layout: TransferMeasurementLayout,
    basis: common.SurfaceReadoutBasis,
) -> tuple[int, ...]:
    """Return the transfer records that correct one logical readout basis.

    Steane-X demolition determines the logical Z byproduct, the accepted final
    interface outcome determines the logical X byproduct, and Y needs both.
    """
    basis = normalize_pauli_char(basis)
    interface = layout.rounds("interface_z")[-1]
    steane_x = layout.by_label("steane_x")
    demolition = tuple(steane_x[label] for label in common.HIRANO_STEANE_LOGICAL_X)
    return demolition if basis == "X" else interface if basis == "Z" else interface + demolition
