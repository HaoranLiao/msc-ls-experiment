from __future__ import annotations

from itertools import zip_longest

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

from msc.rotated_surface_code.post_transfer import post_transfer_qiskit
from msc.state_transfer import common
from msc.steane_code import magic_state_cultivation_qiskit as cultivation
from msc.steane_code import state_injection_qiskit as steane
from msc.utils.normalize import normalize_pauli_char


def build_lattice_surgery_circuit(
    spec: common.SurfaceCodeTransferSpec,
    *,
    patch_sites: common.SteanePatchSites = 14,
    injected_state: steane.InjectedStateName = "T",
    include_cultivation: bool = True,
    include_post_transfer_stabilizer_rounds: bool = False,
    surface_readout_basis: common.SurfaceReadoutBasis | None = None,
    add_barriers: bool = True,
) -> QuantumCircuit:
    """Build injection -> QED/cultivation -> three-round transfer -> continuation.

    The circuit encodes the Steane input on the selected patch, performs
    post-injection QED and optional H_XY cultivation, prepares surface
    ``|+_L>``, executes the three-row merge table, demolishes Steane in X, and
    restores the withheld top-edge surface X checks. The default 14-site path
    uses the abstract-routed encoder; the 11-site path uses Rosenfeld's encoder.
    Optional fixed-patch syndrome rounds and destructive logical readout are
    analysis additions.

    Each merge round appends interface checks, compatible surface checks, then
    its color-code verification group. Mixed-check CNOTs alternate code sides,
    with the exact interface operators determined by the target:

    - d=3 measures ``Z1a = Z_1 Z_a`` and
      ``Z35bc = Z_3 Z_5 Z_b Z_c``;
    - d=5 measures those two checks plus ``Zde = Z_d Z_e``.

    Their CNOT support orders are respectively ``1 -> a``,
    ``3 -> b -> 5 -> c``, and, for d=5, ``d -> e``. The product of the target's
    interface checks is logical ``Z_L Z_L``; the alternating ``Z35bc`` order is
    Hirano's hook-safe path. Interface and color-code outcomes are later
    postselected. On the 14-site patch, same-face X/Z requests share one Bell-
    pair preparation and one simultaneous ``BOTH`` extraction. Single-basis
    requests omit the opposite-basis data-coupling phase but read its Bell
    partner as a known-zero Bell-correlation flag, not as the omitted face
    parity. This closes an otherwise-hidden two-fault path and retains graphlike
    distance three in the serialized schedule. Surface histories are
    spacetime-decoded into the Pauli frame.
    """
    if surface_readout_basis is not None:
        surface_readout_basis = normalize_pauli_char(surface_readout_basis)
    patch_spec = common.resolve_steane_transfer_patch_spec(patch_sites)

    # Quantum allocation uses one single-qubit register per physical site:
    # - one register per site of the selected Steane patch;
    # - distance**2 surface-data registers (9 for d=3, 25 for d=5);
    # - one ancilla per surface Z or X stabilizer
    #   (4 Z + 4 X for d=3, 12 Z + 12 X for d=5);
    # - one interface ancilla per merged-boundary check
    #   (2 for d=3, 3 for d=5).
    qubit_labels = (
        *patch_spec.site_labels,
        *spec.surface_order,
        *(name.lower() for name in spec.z_labels),
        *(name.lower() for name in spec.x_labels),
        *(f"zz_{name[1:].lower()}" for name, _, _ in spec.merged_boundary_checks),
    )
    qregs = {label: QuantumRegister(1, label) for label in qubit_labels}
    qubits = {label: register[0] for label, register in qregs.items()}
    steane_patch = [qubits[label] for label in patch_spec.site_labels]
    surface_qubits = [qubits[label] for label in spec.surface_order]

    # Classical allocation uses one register per measurement family:
    # - enc_flags: 4 Rosenfeld or 5 abstract-routed encoder-flag bits;
    # - steane_qed: 6 post-injection Steane stabilizer bits, plus the 14-site
    #   D2-copy flag when applicable;
    # - cult: 5 (11-site) or 7 (14-site) bits, when cultivation is enabled;
    # - surface_init_z/x: one bit per surface stabilizer
    #   (4 + 4 for d=3, 12 + 12 for d=5);
    # - interface_z: one bit per interface check in each of the 3 merge rounds
    #   (6 for d=3, 9 for d=5);
    # - merged_surface_z: one bit per Z stabilizer in each merge round
    #   (12 for d=3, 36 for d=5);
    # - merged_surface_x: one bit per compatible X stabilizer in each merge round
    #   (9 for d=3, 30 for d=5);
    # - steane_verify: 8 requested checks across the two verification groups;
    # - steane_bell_partner_flags: six Bell-partner/D2' flags for the
    #   14-site patch only;
    # - steane_x: 7 Steane demolition bits;
    # - surface_x_recovery: one bit for the X_ab completion measurement m_ab;
    # - post_transfer_surface_z/x: distance rounds of full syndromes, if enabled
    #   (12 Z + 12 X bits for d=3, 60 Z + 60 X bits for d=5);
    # - surface_logical_x/y/z: one bit per measured physical qubit, if requested
    #   (Y uses the union of the logical-X and logical-Z supports).
    measurement_labels = common.transfer_measurement_labels(
        spec,
        patch_sites=patch_sites,
        include_cultivation=include_cultivation,
        post_transfer_stabilizer_rounds=(
            spec.distance if include_post_transfer_stabilizer_rounds else 0
        ),
        surface_readout_basis=surface_readout_basis,
    )
    cregs = {
        name: ClassicalRegister(len(labels), name) for name, labels in measurement_labels.items()
    }
    circuit = QuantumCircuit(*qregs.values(), *cregs.values())

    # 1. Accepted Steane input: injection -> QED -> optional H_XY cultivation.
    if patch_sites == 11:
        cultivation.append_11_site_patch_injection_and_cultivation(
            circuit,
            patch_qubits=steane_patch,
            injected_state=injected_state,
            encoder_bits=list(cregs["enc_flags"]),
            qed_bits=list(cregs["steane_qed"]),
            cultivation_bits=[list(cregs["cult"])] if include_cultivation else [],
            cultivation_rounds=int(include_cultivation),
            add_qed_round=True,
            add_barriers=add_barriers,
            skip_cx_1_4=False,
        )
        append_steane_logical_measurement = steane.append_11_site_patch_logical_measurement
    elif patch_sites == 14:
        cultivation.append_14_site_patch_injection_and_cultivation(
            circuit,
            patch_qubits=steane_patch,
            injected_state=injected_state,
            encoder="abstract_routed",
            encoder_bits=list(cregs["enc_flags"]),
            qed_bits=list(cregs["steane_qed"]),
            cultivation_bits=[list(cregs["cult"])] if include_cultivation else [],
            cultivation_rounds=int(include_cultivation),
            add_qed_round=True,
            add_barriers=add_barriers,
        )
        append_steane_logical_measurement = steane.append_14_site_patch_logical_measurement
    else:
        raise ValueError("patch_sites must be 11 or 14.")

    # 2. Prepare surface |+_L>. Product |+> fixes every X check to +1; the
    # complete X/Z round records that X boundary while the Z checks project
    # into the code space and establish their nondeterministic reference.
    operations = spec.surface_code_operations
    operations.prepare_product_plus(
        circuit,
        qubits=surface_qubits,
        add_barriers=add_barriers,
        barrier_label=f"prepare {spec.name} |+> product state",
    )
    operations.append_scheduled_stabilizer_round(
        circuit,
        surface_qubits=surface_qubits,
        selected_stabilizer_names=spec.z_labels + spec.x_labels,
        ancillas=[qubits[name.lower()] for name in spec.z_labels + spec.x_labels],
        bits=list(cregs["surface_init_z"]) + list(cregs["surface_init_x"]),
        round_label=f"initialize {spec.name} |+_L>",
        reset_ancillas_initially=True,
        add_barriers=False,
    )
    if add_barriers:
        circuit.barrier(label=f"initialize {spec.name} |+_L>")

    # 3. Three merged rounds: interface -> surface -> Steane verification.
    round_widths = {
        "interface_z": len(spec.merged_boundary_checks),
        "merged_surface_z": len(spec.z_labels),
        "merged_surface_x": len(spec.merged_x_labels),
    }
    bits_by_family = {name: list(cregs[name]) for name in round_widths}
    verification_bit_by_label = dict(
        zip(measurement_labels["steane_verify"], cregs["steane_verify"], strict=True)
    )
    bell_partner_flag_bit_by_label = dict(
        zip(
            measurement_labels.get("steane_bell_partner_flags", ()),
            cregs.get("steane_bell_partner_flags", ()),
            strict=True,
        )
    )
    for round_index in range(common.LATTICE_SURGERY_ROUNDS):
        interface_start = round_index * round_widths["interface_z"]
        for (check_name, steane_labels, surface_labels), bit in zip(
            spec.merged_boundary_checks,
            bits_by_family["interface_z"][
                interface_start : interface_start + round_widths["interface_z"]
            ],
            strict=True,
        ):
            ancilla = qubits[f"zz_{check_name[1:].lower()}"]
            circuit.reset(ancilla)
            for steane_label, surface_label in zip_longest(steane_labels, surface_labels):
                for label in (steane_label, surface_label):
                    if label is not None:
                        circuit.cx(qubits[label], ancilla)
            circuit.measure(ancilla, bit)

        z_start = round_index * round_widths["merged_surface_z"]
        x_start = round_index * round_widths["merged_surface_x"]
        operations.append_scheduled_stabilizer_round(
            circuit,
            surface_qubits=surface_qubits,
            selected_stabilizer_names=spec.z_labels + spec.merged_x_labels,
            ancillas=[qubits[name.lower()] for name in spec.z_labels + spec.merged_x_labels],
            bits=(
                bits_by_family["merged_surface_z"][
                    z_start : z_start + round_widths["merged_surface_z"]
                ]
                + bits_by_family["merged_surface_x"][
                    x_start : x_start + round_widths["merged_surface_x"]
                ]
            ),
            round_label=f"merged interval round {round_index + 1}",
            reset_ancillas_initially=True,
            add_barriers=False,
        )
        verification_checks = common.STEANE_MERGE_VERIFICATION_ROUNDS[round_index]
        if patch_sites == 11:
            for check_name in verification_checks:
                steane.append_11_site_patch_stabilizer_measurement(
                    circuit,
                    patch_qubits=steane_patch,
                    bits=[verification_bit_by_label[f"round{round_index + 1}_{check_name}"]],
                    basis=check_name[0],
                    stabilizer_indices=(
                        patch_spec.merge_verification_stabilizer_indices[check_name],
                    ),
                    add_barriers=False,
                )
        elif patch_sites == 14:
            for group in common.group_14_site_superdense_verification_checks(verification_checks):
                measure_d2_copy_flag = (
                    patch_spec.merge_verification_stabilizer_indices["X0246"]
                    in group.stabilizer_indices
                )
                steane.append_14_site_patch_superdense_stabilizer_measurement(
                    circuit,
                    patch_qubits=steane_patch,
                    stabilizer_bits=[
                        verification_bit_by_label[f"round{round_index + 1}_{check_name}"]
                        for check_name in group.check_names_in_readout_order
                    ],
                    bell_partner_flag_bits=(
                        [
                            bell_partner_flag_bit_by_label[
                                f"round{round_index + 1}_{check_name}_bell_partner_flag"
                            ]
                            for check_name in group.check_names_in_readout_order
                        ]
                        if group.basis != "BOTH"
                        else None
                    ),
                    d2_copy_flag_bit=(
                        bell_partner_flag_bit_by_label[f"round{round_index + 1}_D2_copy_flag"]
                        if measure_d2_copy_flag
                        else None
                    ),
                    basis=group.basis,
                    stabilizer_indices=group.stabilizer_indices,
                    add_barriers=False,
                )
        else:
            raise ValueError("patch_sites must be 11 or 14.")
        if add_barriers:
            circuit.barrier(label=f"merged interval round {round_index + 1}")

    # 4. Demolish Steane in X, then measure X_ab as Hirano's m_ab. X_cd is
    # withheld by the d=5 merge but does not receive a standalone recovery.
    append_steane_logical_measurement(
        circuit,
        "X",
        patch_qubits=steane_patch,
        support=patch_spec.data_sites_in_hirano_order,
        bits=list(cregs["steane_x"]),
        destructive=True,
    )
    if add_barriers:
        circuit.barrier(label="measure Steane data in X")
    operations.append_scheduled_stabilizer_round(
        circuit,
        surface_qubits=surface_qubits,
        selected_stabilizer_names=common.SURFACE_RECOVERY_X_LABELS,
        ancillas=[qubits[name.lower()] for name in common.SURFACE_RECOVERY_X_LABELS],
        bits=list(cregs["surface_x_recovery"]),
        round_label="recover surface top edge",
        reset_ancillas_initially=True,
        add_barriers=add_barriers,
    )

    # 5. Optional fixed-patch history and destructive logical-support readout.
    post_transfer_qiskit.append_rotated_full_stabilizer_rounds(
        circuit,
        surface_qubits=surface_qubits,
        z_ancillas=[qubits[name.lower()] for name in spec.z_labels],
        x_ancillas=[qubits[name.lower()] for name in spec.x_labels],
        z_stabilizer_names=spec.z_labels,
        x_stabilizer_names=spec.x_labels,
        z_bits=(
            list(cregs["post_transfer_surface_z"])
            if include_post_transfer_stabilizer_rounds
            else ()
        ),
        x_bits=(
            list(cregs["post_transfer_surface_x"])
            if include_post_transfer_stabilizer_rounds
            else ()
        ),
        stabilizer_rounds=spec.distance if include_post_transfer_stabilizer_rounds else 0,
        append_scheduled_stabilizer_round=operations.append_scheduled_stabilizer_round,
        logical_readout_basis=surface_readout_basis,
        logical_bits=(
            list(cregs[f"surface_logical_{surface_readout_basis.lower()}"])
            if surface_readout_basis
            else ()
        ),
        append_logical_measurement=operations.append_logical_measurement,
        add_barriers=add_barriers,
    )
    return circuit
