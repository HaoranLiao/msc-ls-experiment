from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import zip_longest

import stim

from msc.state_transfer import common
from msc.steane_code import magic_state_cultivation_stim as cultivation
from msc.steane_code import state_injection_stim as steane
from msc.utils import circuit_stim
from msc.utils.normalize import normalize_pauli_char


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


def transfer_measurement_layout(
    spec: common.SurfaceCodeTransferSpec,
    *,
    patch_sites: common.SteanePatchSites = 14,
    include_cultivation: bool,
    measure_encoder_flags: bool = False,
    measure_cultivation_flags: bool = False,
    post_injection_qed_checks: Sequence[str] | None = None,
    merge_schedule: common.MergeSchedule = "hirano",
) -> TransferMeasurementLayout:
    """Map shared measurement labels into Stim's physical append order.

    The shared measurement labels group measurements by meaning, whereas Stim stores one
    global record in circuit order. This layout explicitly interleaves each
    round as interface checks -> compatible surface checks -> color-code
    checks, then places demolition and the ``X_ab`` recovery in their actual
    append order. ``X_cd`` first reappears in full surface round 1, which is
    appended by the d=5 continuation.
    """
    patch_spec = common.resolve_steane_transfer_patch_spec(
        patch_sites, merge_schedule=merge_schedule
    )
    labels = common.transfer_measurement_labels(
        spec,
        patch_sites=patch_sites,
        include_cultivation=include_cultivation,
        merge_schedule=merge_schedule,
        surface_readout_basis=None,
        measure_encoder_flags=measure_encoder_flags,
        measure_cultivation_flags=measure_cultivation_flags,
        post_injection_qed_checks=post_injection_qed_checks,
    )
    if patch_sites == 14:
        # Follow the parallel superdense readout order: same-face X/Z checks
        # share one Bell pair, and the D2' restoration flag follows the 0246
        # Bell-ancilla readouts whenever that face is measured.
        labels["steane_verify"] = tuple(
            f"round{round_index + 1}_{check_name}"
            for round_index, checks in enumerate(patch_spec.merge_verification_rounds)
            for check_name in sorted(
                checks,
                key=lambda name: (patch_spec.merge_verification_stabilizer_indices[name], name[0]),
            )
        )
    family_labels: dict[str, list[str]] = {}
    family_positions: dict[str, list[int]] = {}
    next_position = 0

    def record_measurements(name: str, width: int) -> None:
        """Assign the next chunk of one family's labels to the global Stim record."""
        nonlocal next_position
        recorded_labels = family_labels.setdefault(name, [])
        start = len(recorded_labels)
        chunk = labels[name][start : start + width]
        recorded_labels.extend(chunk)
        family_positions.setdefault(name, []).extend(
            range(next_position, next_position + len(chunk))
        )
        next_position += len(chunk)

    if measure_encoder_flags:
        record_measurements("enc_flags", len(labels["enc_flags"]))
    record_measurements("steane_qed", len(labels["steane_qed"]))
    if include_cultivation:
        record_measurements("cult", len(labels["cult"]))
    for name in ("surface_init_z", "surface_init_x"):
        record_measurements(name, len(labels[name]))

    for round_index in range(common.LATTICE_SURGERY_ROUNDS):
        record_measurements("interface_z", len(spec.merged_boundary_checks))
        record_measurements("merged_surface_z", len(spec.z_labels))
        record_measurements("merged_surface_x", len(spec.merged_x_labels))
        if patch_sites == 14:
            stabilizer_bases, stabilizer_indices = common.resolve_14_site_patch_stabilizer_checks(
                tuple(
                    f"{name[0]}_{name[1:]}"
                    for name in patch_spec.merge_verification_rounds[round_index]
                )
            )
            if stabilizer_indices:
                _, readouts = steane.select_14_site_patch_superdense_measurement_layers(
                    stabilizer_bases,
                    stabilizer_indices,
                )
                for _, _, readout_role in readouts:
                    if readout_role == "STABILIZER":
                        family = "steane_verify"
                    elif readout_role in ("BELL_PARTNER_FLAG", "D2_COPY_FLAG"):
                        family = "steane_bell_partner_flags"
                    else:  # pragma: no cover - assigned by the shared selector.
                        raise ValueError(f"Unsupported superdense readout role: {readout_role!r}.")
                    record_measurements(family, 1)
        elif patch_sites == 11:
            for _ in patch_spec.merge_verification_rounds[round_index]:
                record_measurements("steane_verify", 1)
        else:  # pragma: no cover - resolved patch specs support only 11/14 sites.
            raise ValueError("patch_sites must be 11 or 14.")

    for name in ("steane_x", "surface_x_recovery"):
        record_measurements(name, len(labels[name]))
    if any(tuple(family_labels[name]) != expected for name, expected in labels.items()):
        raise RuntimeError("Stim measurement order does not cover the shared measurement labels.")
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
    injected_state: steane.StimInjectedStateName = "S",
    projection_axis: cultivation.ProjectionAxis = "Y",
    include_cultivation: bool = True,
    measure_encoder_flags: bool = False,
    measure_cultivation_flags: bool = False,
    post_injection_qed_checks: Sequence[str] | None = None,
    merge_schedule: common.MergeSchedule = "hirano",
    add_ticks: bool = True,
) -> stim.Circuit:
    """Build a detector-annotated logical-Pauli projection and transfer circuit.

    1. Encode ``injected_state`` on the selected Steane patch and, when
       ``measure_encoder_flags`` is true, measure and postselect its encoder
       flags. The default 14-site path uses the abstract-routed encoder;
       the 11-site path uses Rosenfeld's encoder.
    2. Measure all six Steane stabilizers for post-injection QED.
    3. Optionally project the requested logical Pauli; Y is Stim's proxy for
       H_XY cultivation, while X and Z are diagnostic projections.
    4. Prepare the surface patch in product ``|+>`` and measure a complete X/Z
       round. Z checks project ``|+_L>`` and establish references; X checks are
       compared with their known +1 values.
    5. Perform three merge rounds of interface Z checks, compatible surface
       checks, and the prescribed Steane verification groups. Same-face X/Z
       requests on the 14-site patch share one Bell pair; single-basis
       extractions also read the unused Bell partner as a known-zero flag.
    6. Demolish Steane data in X, then measure only ``X_ab`` as Hirano's
       ``m_ab``. There is no standalone ``X_cd`` recovery.
    7. End at the transfer boundary. The d=5 wrapper owns the required full
       surface rounds and destructive readout.

    Each merge round appends interface Z checks, compatible surface checks,
    then Steane verification. Mixed interface check ``Z35bc`` retains Hirano's
    hook-safe ``3 -> b -> 5 -> c`` order. Full round 1 establishes the deferred
    ``X_cd`` stream when present.

    Known-+1 initial X results and measured initial Z results seed the surface
    change detectors. Protocol checks are postselected, surface histories are
    decoded, and transfer corrections remain in a classical Pauli frame.
    Y uses ``S|+>`` and logical-Y cultivation as Clifford proxies; X and Z are
    direct diagnostic projections. This serialized helper does not reproduce
    Hirano's exact concurrent timing, idle noise, later code expansion, or
    complementary-gap lookup.
    """
    patch_spec = common.resolve_steane_transfer_patch_spec(
        patch_sites, merge_schedule=merge_schedule
    )
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

    # 1. Accepted input: injection -> QED -> optional logical-Pauli projection.
    if patch_sites == 11:
        if projection_axis != "Y":
            raise ValueError("11-site Stim transfer supports only projection_axis='Y'.")
        if post_injection_qed_checks is not None:
            raise ValueError("post_injection_qed_checks subsetting requires the 14-site patch.")
        cultivation.append_11_site_patch_injection_and_cultivation(
            circuit,
            patch_qubits=steane_patch,
            injected_state=injected_state,
            cultivation_rounds=int(include_cultivation),
            measure_encoder_flags=measure_encoder_flags,
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
            projection_axis=projection_axis,
            encoder="abstract_routed",
            cultivation_rounds=int(include_cultivation),
            measure_encoder_flags=measure_encoder_flags,
            measure_cultivation_flags=measure_cultivation_flags,
            post_injection_qed_checks=post_injection_qed_checks,
            add_qed_round=True,
            add_ticks=add_ticks,
        )
        append_steane_logical_measurement = steane.append_14_site_patch_logical_measurement
    else:  # pragma: no cover - resolved above by the patch specification.
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
            add_ticks=add_ticks,
        )
        verification_checks = patch_spec.merge_verification_rounds[round_index]
        if patch_sites == 11:
            for check_name in verification_checks:
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
            stabilizer_bases, stabilizer_indices = common.resolve_14_site_patch_stabilizer_checks(
                tuple(f"{name[0]}_{name[1:]}" for name in verification_checks)
            )
            if stabilizer_indices:
                steane.append_14_site_patch_superdense_stabilizer_measurement(
                    circuit,
                    patch_qubits=steane_patch,
                    basis=stabilizer_bases,
                    stabilizer_indices=stabilizer_indices,
                    add_ticks=add_ticks,
                )
        else:  # pragma: no cover - resolved above by the patch specification.
            raise ValueError("patch_sites must be 11 or 14.")

    # 4. Demolish Steane in X and complete lattice surgery. In Hirano's Fig. 9
    # path the sole selected result is X_ab (m_ab); X_cd is deliberately absent
    # until the first complete noisy surface round.
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
        selected_stabilizer_names=common.SURFACE_RECOVERY_X_LABELS,
        ancillas=[qubits[label.lower()] for label in common.SURFACE_RECOVERY_X_LABELS],
        round_label="complete lattice surgery X references",
        # Hirano resets A_AB in X before coupling it to a,b and reading m_ab.
        # The Stim surface helper uses the equivalent native RX/CX/MX sequence.
        add_ticks=add_ticks,
    )

    layout = transfer_measurement_layout(
        spec,
        patch_sites=patch_sites,
        include_cultivation=include_cultivation,
        measure_encoder_flags=measure_encoder_flags,
        measure_cultivation_flags=measure_cultivation_flags,
        post_injection_qed_checks=post_injection_qed_checks,
        merge_schedule=merge_schedule,
    )

    # 6. Steane-local checks were annotated inline; append the shared transfer
    # detectors: postselected color/interface checks, then decodable surface history.
    local_detector_count = sum(
        len(layout.positions.get(name, ())) for name in common.STEANE_POSTSELECTION_FAMILIES
    )
    if circuit.num_detectors != local_detector_count:
        raise RuntimeError("transfer-local detector annotations are inconsistent.")
    append_labeled_detectors(circuit, layout=layout, detectors=common.transfer_detectors(spec))
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


def append_labeled_detectors(
    circuit: stim.Circuit,
    *,
    layout: TransferMeasurementLayout,
    detectors: Sequence[common.TransferDetector],
) -> None:
    """Resolve shared ``(family, label)`` detector rules into Stim record positions."""
    positions_by_family = {name: layout.by_label(name) for name in layout.labels}
    for records, _ in detectors:
        circuit_stim.append_detector(
            circuit,
            measurement_positions=tuple(
                positions_by_family[family][label] for family, label in records
            ),
        )


def num_transfer_protocol_postselection_detectors(layout: TransferMeasurementLayout) -> int:
    """Return the hard-postselection prefix length before surface history.

    The prefix contains Steane/color-code checks, changes between repeated
    interface rounds, and the three final X checks reconstructed at demolition.
    """
    steane_detector_count = sum(
        len(layout.positions.get(name, ())) for name in common.STEANE_POSTSELECTION_FAMILIES
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
    spec: common.SurfaceCodeTransferSpec,
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
    The layout's ``surface_logical_<basis>`` family identifies the physical
    measurement positions contributing to the logical-support parity.
    """
    basis = normalize_pauli_char(basis)
    circuit_stim.append_observable_include(
        circuit,
        measurement_positions=(
            layout.positions[f"surface_logical_{basis.lower()}"]
            + transfer_logical_frame_positions(spec, layout, basis)
        ),
        observable_index=observable_index,
    )


def transfer_logical_frame_positions(
    spec: common.SurfaceCodeTransferSpec,
    layout: TransferMeasurementLayout,
    basis: common.SurfaceReadoutBasis,
) -> tuple[int, ...]:
    """Resolve the shared transfer-frame records that correct one readout basis."""
    return tuple(
        layout.by_label(family)[label]
        for family, label in common.transfer_logical_frame_records(spec, basis)
    )
