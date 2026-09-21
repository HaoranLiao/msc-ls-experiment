"""Qiskit H_XZ/H_XY cultivation on the 11-site and 14-site Steane patches."""

from __future__ import annotations

from collections.abc import Sequence

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

from msc.steane_code.magic_state_cultivation_common import (
    PATCH_11_SITE_CULTIVATION_MEASUREMENTS_PER_ROUND,
    PATCH_11_SITE_FOLD_LAYERS,
    PATCH_11_SITE_FOLDED_MEASUREMENT_SITE,
    PATCH_14_SITE_CULTIVATION_MEASUREMENTS_PER_ROUND,
    PATCH_14_SITE_FOLD_LAYERS,
    PATCH_14_SITE_FOLDED_MEASUREMENT_SITE,
    PATCH_14_SITE_QED_MEASUREMENTS_PER_ROUND,
    STEANE_QED_MEASUREMENTS_PER_ROUND,
    CultivatedStateName,
    Patch14SiteEncoderName,
    filter_11_site_patch_fold_layer,
    patch_14_site_encoder_flag_count,
)
from msc.steane_code.state_injection_common import (
    NUM_QUBITS_PATCH_14_SITES,
    PATCH_11_SITE_DATA_SITES,
    PATCH_11_SITE_HELPER_SITES,
    PATCH_14_SITE_DATA_SITES,
    PATCH_14_SITE_HELPER_SITES,
)
from msc.steane_code.state_injection_qiskit import (
    NUM_QUBITS_PATCH_11_SITES,
    InjectedStateName,
    append_11_site_patch_rosenfeld_encoder,
    append_11_site_patch_stabilizer_measurement,
    append_14_site_patch_abstract_encoder,
    append_14_site_patch_abstract_routed_encoder,
    append_14_site_patch_hirano_encoder,
    append_14_site_patch_superdense_stabilizer_measurement,
)
from msc.utils.circuit_qiskit import apply_ty, apply_ty_dagger
from msc.utils.normalize import qiskit_bit_value, resolve_patch_qubits

# =============================================================================
# 1. Framework-neutral layout, fold schedule, and constants
# =============================================================================
#
# The patch layouts, fold/unfold CNOT schedules, measurement counts, and
# filter_11_site_patch_fold_layer are framework-neutral and live in
# magic_state_cultivation_common.py.


# =============================================================================
# 2. Fold / unfold network
# =============================================================================


def measure_in_x_basis(circuit: QuantumCircuit, qubit, bit) -> None:
    """Measure ``qubit`` in the X basis."""
    circuit.h(qubit)
    circuit.measure(qubit, bit)


def reset_to_plus(circuit: QuantumCircuit, qubit) -> None:
    """Reset a qubit and prepare ``|+>``."""
    circuit.reset(qubit)
    circuit.h(qubit)


def _append_transversal_cultivation_basis_rotation(
    circuit: QuantumCircuit,
    *,
    data_qubits: Sequence,
    cultivated_state: CultivatedStateName,
    inverse: bool,
) -> None:
    """Rotate the selected magic-state observable onto logical X."""
    if cultivated_state == "T":
        operation = circuit.tdg if inverse else circuit.t
        for data_qubit in data_qubits:
            operation(data_qubit)
    elif cultivated_state == "Ty":
        operation = apply_ty_dagger if inverse else apply_ty
        for data_qubit in data_qubits:
            operation(circuit, data_qubit)
    else:
        raise ValueError("cultivated_state must be 'T' or 'Ty'.")


def append_cnot_layer_from_site_labels(
    circuit: QuantumCircuit,
    patch_qubits: Sequence,
    cnot_layer: Sequence[tuple[int, int]],
) -> None:
    """Append one layer of CNOTs using 11-site patch labels.

    Parameters
    ----------
    circuit
        Circuit being modified.
    patch_qubits
        Sequence mapping integer site labels to actual Qiskit qubits.
    cnot_layer
        Iterable of ``(control_site, target_site)`` pairs.
    """
    for control_site, target_site in cnot_layer:
        circuit.cx(
            patch_qubits[control_site],
            patch_qubits[target_site],
        )


def append_11_site_patch_fold_network(
    circuit: QuantumCircuit,
    patch_qubits: Sequence,
    *,
    add_barriers: bool = True,
    skip_cx_1_4: bool = False,
    skip_cx: tuple[int, int] | None = None,
) -> None:
    """Append the forward 4-3-2-1 fold network.

    Implements Google paper Fig.1 panel-C steps 3--6:
    step 3: four CNOTs, step 4: three CNOTs, step 5: two CNOTs, step 6: one CNOT.

    Parameters
    ----------
    circuit
        Circuit being modified.
    patch_qubits
        Qubits indexed by the 11-site labels.
    add_barriers
        Whether to insert barriers after each fold layer.
    skip_cx_1_4
        If True, omit the CX 1->4 gate from the fold network.
    skip_cx
        If given, omit the specified (control, target) CX gate.
        Takes precedence over skip_cx_1_4.
    """
    for layer_index, layer in enumerate(PATCH_11_SITE_FOLD_LAYERS, start=3):
        filtered = filter_11_site_patch_fold_layer(layer, skip_cx_1_4=skip_cx_1_4, skip_cx=skip_cx)
        append_cnot_layer_from_site_labels(circuit, patch_qubits, filtered)
        if add_barriers:
            circuit.barrier(label=f"fold step {layer_index}")


def append_11_site_patch_unfold_network(
    circuit: QuantumCircuit,
    patch_qubits: Sequence,
    *,
    add_barriers: bool = True,
    skip_cx_1_4: bool = False,
    skip_cx: tuple[int, int] | None = None,
) -> None:
    """Append the inverse of the 4-3-2-1 fold network.

    Implements the time-reversed panel-C steps 8--12. Since CNOT is self-inverse,
    the inverse network is the same CNOTs in exactly reverse order.

    Parameters
    ----------
    circuit
        Circuit being modified.
    patch_qubits
        Qubits indexed by the 11-site labels.
    add_barriers
        Whether to insert barriers after each inverse layer.
    skip_cx_1_4
        If True, omit the CX 1->4 gate (matching the forward fold).
    skip_cx
        If given, omit the specified (control, target) CX gate.
        Takes precedence over skip_cx_1_4.
    """
    for reverse_step_index, layer in zip(
        range(8, 12),
        reversed(PATCH_11_SITE_FOLD_LAYERS),
        strict=False,
    ):
        filtered = filter_11_site_patch_fold_layer(layer, skip_cx_1_4=skip_cx_1_4, skip_cx=skip_cx)
        append_cnot_layer_from_site_labels(circuit, patch_qubits, list(reversed(filtered)))
        if add_barriers:
            circuit.barrier(label=f"unfold step {reverse_step_index}")


def append_14_site_patch_fold_network(
    circuit: QuantumCircuit,
    patch_qubits: Sequence,
    *,
    add_barriers: bool = True,
) -> None:
    """Append the forward 4-4-2-1 H10-root fold network."""
    for layer_index, layer in enumerate(PATCH_14_SITE_FOLD_LAYERS, start=1):
        append_cnot_layer_from_site_labels(circuit, patch_qubits, layer)
        if add_barriers:
            circuit.barrier(label=f"fold layer {layer_index}")


def append_14_site_patch_unfold_network(
    circuit: QuantumCircuit,
    patch_qubits: Sequence,
    *,
    add_barriers: bool = True,
) -> None:
    """Append the inverse 14-site fold network in fully reversed order."""
    for layer_index, layer in enumerate(reversed(PATCH_14_SITE_FOLD_LAYERS), start=1):
        append_cnot_layer_from_site_labels(circuit, patch_qubits, list(reversed(layer)))
        if add_barriers:
            circuit.barrier(label=f"unfold layer {layer_index}")


def append_11_site_patch_magic_state_projection_round(
    circuit: QuantumCircuit,
    *,
    patch_qubits: Sequence,
    bits: Sequence,
    cultivated_state: CultivatedStateName,
    add_barriers: bool = True,
    skip_cx_1_4: bool = False,
    skip_cx: tuple[int, int] | None = None,
) -> None:
    """Append one H_XY or H_XZ magic-state projection round.

    See the 11-site layout comment in magic_state_cultivation_common.

    This routine implements the Google paper Fig.1 panel-C cultivation sequence:

    1. Prepare the four helper sites in |+>.
    2. Rotate the chosen logical magic-state observable onto X.
    3. Apply the 4-CNOT fold layer.
    4. Apply the 3-CNOT fold layer.
    5. Apply the 2-CNOT fold layer.
    6. Apply the 1-CNOT fold layer.
    7. Measure site 7 in the X basis.
    8. Reset site 7 to |+> for the time-reversed unfold.
    9. Apply the inverse 1-CNOT layer.
    10. Apply the inverse 2-CNOT layer.
    11. Apply the inverse 3-CNOT layer.
    12. Apply the inverse 4-CNOT layer.
    13. Measure helper sites 1, 2, 6, 7 in the X basis.
    14. Undo the transversal basis rotation on all seven data sites.

    Parameters
    ----------
    circuit
        Circuit being modified.
    patch_qubits
        Eleven Qiskit qubits ordered by the 11-site labels.
    bits
        Five classical bits: bit 0 = folded observable (site 7),
        bits 1-4 = helper sites 1, 2, 6, 7.
    cultivated_state
        ``"T"`` projects onto the +1 eigenspace of H_XY; ``"Ty"`` projects
        onto the +1 eigenspace of H_XZ.
    add_barriers
        Whether to insert barriers between conceptual stages.
    skip_cx_1_4
        If True, omit the CX 1->4 gate from the fold/unfold network.
    skip_cx
        If given, omit the specified (control, target) CX gate.
        Takes precedence over skip_cx_1_4.
    """
    patch_qubits = resolve_patch_qubits(
        patch_qubits,
        expected_count=NUM_QUBITS_PATCH_11_SITES,
    )

    if len(bits) != PATCH_11_SITE_CULTIVATION_MEASUREMENTS_PER_ROUND:
        raise ValueError(
            "bits must contain exactly " f"{PATCH_11_SITE_CULTIVATION_MEASUREMENTS_PER_ROUND} bits."
        )

    c = list(bits)

    # Step 2 helper preparation.
    for helper_site in PATCH_11_SITE_HELPER_SITES:
        reset_to_plus(circuit, patch_qubits[helper_site])

    if add_barriers:
        circuit.barrier(label="prepare helpers")

    data_qubits = [patch_qubits[data_site] for data_site in PATCH_11_SITE_DATA_SITES]
    _append_transversal_cultivation_basis_rotation(
        circuit,
        data_qubits=data_qubits,
        cultivated_state=cultivated_state,
        inverse=False,
    )

    if add_barriers:
        circuit.barrier(label=f"{cultivated_state} basis rotation")

    # Steps 3--6: fold.
    append_11_site_patch_fold_network(
        circuit,
        patch_qubits,
        add_barriers=add_barriers,
        skip_cx_1_4=skip_cx_1_4,
        skip_cx=skip_cx,
    )

    # Step 7: measure the folded observable on site 7.
    measure_in_x_basis(
        circuit,
        patch_qubits[PATCH_11_SITE_FOLDED_MEASUREMENT_SITE],
        c[0],
    )

    if add_barriers:
        circuit.barrier(label="measure folded observable")

    # Reprepare site 7 to make the time-reversed unfold explicit.
    reset_to_plus(circuit, patch_qubits[PATCH_11_SITE_FOLDED_MEASUREMENT_SITE])

    if add_barriers:
        circuit.barrier(label="reprepare folded site")

    # Steps 8--12: unfold.
    append_11_site_patch_unfold_network(
        circuit,
        patch_qubits,
        add_barriers=add_barriers,
        skip_cx_1_4=skip_cx_1_4,
        skip_cx=skip_cx,
    )

    # Step 13: measure the four helper / M qubits.
    for bit_index, helper_site in enumerate(PATCH_11_SITE_HELPER_SITES, start=1):
        measure_in_x_basis(
            circuit,
            patch_qubits[helper_site],
            c[bit_index],
        )

    if add_barriers:
        circuit.barrier(label="measure helpers")

    _append_transversal_cultivation_basis_rotation(
        circuit,
        data_qubits=data_qubits,
        cultivated_state=cultivated_state,
        inverse=True,
    )

    if add_barriers:
        circuit.barrier(label=f"undo {cultivated_state} basis rotation")


def append_14_site_patch_magic_state_projection_round(
    circuit: QuantumCircuit,
    *,
    patch_qubits: Sequence,
    bits: Sequence,
    cultivated_state: CultivatedStateName,
    add_barriers: bool = True,
) -> None:
    """Append one logical magic-state projection round on the 14-site patch.

    The round prepares H8-H12 in ``|+>``, rotates H_XY or H_XZ onto logical X,
    folds onto H10, measures and resets H10, unfolds, measures H8-H12 in X,
    then undoes the basis rotation. The six output bits are the folded H10 root
    followed by H8, H9, H10, H11, and H12. The second H10 result is a root
    check; the other four helper results are fixed-zero checks. H7 is unused.

    Cultivation postselects the folded root to zero. KT retains either root
    branch and requires the repeated H10 result to agree with it.

    Parameters
    ----------
    cultivated_state
        ``"T"`` projects onto the +1 eigenspace of H_XY; ``"Ty"`` projects
        onto the +1 eigenspace of H_XZ.
    """
    patch_qubits = resolve_patch_qubits(
        patch_qubits,
        expected_count=NUM_QUBITS_PATCH_14_SITES,
    )
    if len(bits) != PATCH_14_SITE_CULTIVATION_MEASUREMENTS_PER_ROUND:
        raise ValueError(
            "bits must contain exactly " f"{PATCH_14_SITE_CULTIVATION_MEASUREMENTS_PER_ROUND} bits."
        )
    bits = list(bits)

    for helper_site in PATCH_14_SITE_HELPER_SITES[1:]:
        reset_to_plus(circuit, patch_qubits[helper_site])
    if add_barriers:
        circuit.barrier(label="prepare helpers")

    data_qubits = [patch_qubits[data_site] for data_site in PATCH_14_SITE_DATA_SITES]
    _append_transversal_cultivation_basis_rotation(
        circuit,
        data_qubits=data_qubits,
        cultivated_state=cultivated_state,
        inverse=False,
    )
    if add_barriers:
        circuit.barrier(label=f"{cultivated_state} basis rotation")

    append_14_site_patch_fold_network(
        circuit,
        patch_qubits,
        add_barriers=add_barriers,
    )
    measure_in_x_basis(
        circuit,
        patch_qubits[PATCH_14_SITE_FOLDED_MEASUREMENT_SITE],
        bits[0],
    )
    if add_barriers:
        circuit.barrier(label="measure folded observable")

    reset_to_plus(circuit, patch_qubits[PATCH_14_SITE_FOLDED_MEASUREMENT_SITE])
    if add_barriers:
        circuit.barrier(label="restore folded site")

    append_14_site_patch_unfold_network(
        circuit,
        patch_qubits,
        add_barriers=add_barriers,
    )
    for bit_index, helper_site in enumerate(PATCH_14_SITE_HELPER_SITES[1:], start=1):
        measure_in_x_basis(circuit, patch_qubits[helper_site], bits[bit_index])
    if add_barriers:
        circuit.barrier(label="measure helpers")

    _append_transversal_cultivation_basis_rotation(
        circuit,
        data_qubits=data_qubits,
        cultivated_state=cultivated_state,
        inverse=True,
    )
    if add_barriers:
        circuit.barrier(label=f"undo {cultivated_state} basis rotation")


# =============================================================================
# 3. Injection-and-cultivation circuit construction
# =============================================================================


def append_11_site_patch_injection_and_cultivation(
    circuit: QuantumCircuit,
    *,
    add_qed_round: bool,
    skip_cx_1_4: bool,
    patch_qubits: Sequence,
    injected_state: InjectedStateName = "Ty",
    cultivated_state: CultivatedStateName = "Ty",
    encoder_bits: Sequence | None,
    cultivation_bits: Sequence[Sequence],
    cultivation_rounds: int = 1,
    qed_bits: Sequence | None = None,
    add_barriers: bool = True,
    skip_cx: tuple[int, int] | None = None,
) -> None:
    """Append Steane injection, optional QED, and zero or more cultivation rounds.

    When ``add_qed_round`` is true, the appender measures all three X and all
    three Z Steane stabilizers immediately after injection and before
    cultivation. Hirano and Rosenfeld call this a QEC cycle, but its six
    outcomes are used only for
    error detection and postselection; this helper applies no recovery.

    Parameters
    ----------
    circuit
        Circuit being modified.
    patch_qubits
        Eleven qubits indexed by the 11-site patch labels.
    injected_state
        Physical state injected on the center data site before encoding.
    cultivated_state
        Magic state selected by each projection round. ``"T"`` measures
        logical H_XY and ``"Ty"`` measures logical H_XZ.
    encoder_bits
        Four classical bits for encoder-flag postselection, or ``None`` to
        skip the encoder-flag measurements entirely.
    cultivation_bits
        Exactly ``cultivation_rounds`` 5-bit sequences, one per cultivation
        round.
    cultivation_rounds
        Number of selected magic-state cultivation rounds. Zero appends
        injection and optional QED without cultivation.
    qed_bits
        Six bits for the post-injection QED outcomes, ordered by
        ``post_injection_qed_stabilizer_labels``: the three X stabilizers
        followed by the three Z stabilizers. Required when
        ``add_qed_round`` is true and ignored otherwise.
    add_qed_round
        Whether to append the post-injection Steane X/Z stabilizer round.
    add_barriers
        Whether to insert barriers between stages.
    skip_cx_1_4
        If True, omit the CX 1->4 gate from the fold/unfold network.
    skip_cx
        If given, omit the specified (control, target) CX gate.
    """
    if encoder_bits is not None and len(encoder_bits) < 4:
        raise ValueError("encoder_bits must contain at least 4 bits.")
    if cultivation_rounds < 0:
        raise ValueError("cultivation_rounds must be non-negative.")
    if len(cultivation_bits) != cultivation_rounds:
        raise ValueError(
            "cultivation_bits must contain exactly one 5-bit " "sequence per cultivation round."
        )
    if add_qed_round and (qed_bits is None or len(qed_bits) != STEANE_QED_MEASUREMENTS_PER_ROUND):
        raise ValueError(
            "qed_bits must contain exactly "
            f"{STEANE_QED_MEASUREMENTS_PER_ROUND} bits when add_qed_round is true."
        )

    append_11_site_patch_rosenfeld_encoder(
        circuit,
        patch_qubits=patch_qubits,
        injected_state=injected_state,
        flag_bits=encoder_bits,
        add_barriers=add_barriers,
    )
    if add_barriers:
        circuit.barrier(label="state injection")

    if add_qed_round:
        assert qed_bits is not None
        append_11_site_patch_stabilizer_measurement(
            circuit,
            patch_qubits=patch_qubits,
            bits=qed_bits,
            basis="BOTH",
            add_barriers=add_barriers,
        )
        if add_barriers:
            circuit.barrier(label="post-injection QED")

    for round_index in range(cultivation_rounds):
        append_11_site_patch_magic_state_projection_round(
            circuit,
            patch_qubits=patch_qubits,
            bits=cultivation_bits[round_index],
            cultivated_state=cultivated_state,
            add_barriers=add_barriers,
            skip_cx_1_4=skip_cx_1_4,
            skip_cx=skip_cx,
        )


def append_14_site_patch_injection_and_cultivation(
    circuit: QuantumCircuit,
    *,
    add_qed_round: bool,
    patch_qubits: Sequence,
    encoder_bits: Sequence | None,
    cultivation_bits: Sequence[Sequence],
    encoder: Patch14SiteEncoderName = "abstract_routed",
    injected_state: InjectedStateName = "Ty",
    cultivated_state: CultivatedStateName = "Ty",
    cultivation_rounds: int = 1,
    qed_bits: Sequence | None = None,
    add_barriers: bool = True,
) -> None:
    """Append 14-site injection, optional QED, and cultivation rounds.

    ``encoder="abstract_routed"`` is the production default and accepts an
    arbitrary injected state. ``"abstract"`` selects the direct unrouted
    reference encoder, while ``"hirano"`` selects the specialized logical-Y
    encoder. The optional QED record contains the six Steane stabilizers
    followed by the terminal D2' copy-detour flag; all seven are postselection
    data and no recovery is applied.

    ``encoder_bits`` receives the encoder-flag postselection outcomes; pass
    ``None`` to skip those measurements entirely, in which case the classical
    record omits the flag bits and no encoder-flag postselection is available.

    ``cultivated_state="Ty"`` is the Qiskit default and measures logical
    H_XZ. Pass ``"T"`` to retain logical-H_XY cultivation.
    """
    encoder_flag_count = patch_14_site_encoder_flag_count(encoder)
    if encoder_bits is not None and len(encoder_bits) < encoder_flag_count:
        raise ValueError(f"encoder_bits must contain at least {encoder_flag_count} bits.")
    if cultivation_rounds < 0:
        raise ValueError("cultivation_rounds must be non-negative.")
    if len(cultivation_bits) != cultivation_rounds:
        raise ValueError(
            "cultivation_bits must contain exactly one "
            f"{PATCH_14_SITE_CULTIVATION_MEASUREMENTS_PER_ROUND}-bit sequence per "
            "cultivation round."
        )
    if add_qed_round and (
        qed_bits is None or len(qed_bits) != PATCH_14_SITE_QED_MEASUREMENTS_PER_ROUND
    ):
        raise ValueError(
            "qed_bits must contain exactly "
            f"{PATCH_14_SITE_QED_MEASUREMENTS_PER_ROUND} bits when add_qed_round is true."
        )

    encoder_flags = list(encoder_bits) if encoder_bits is not None and encoder_flag_count else None
    if encoder == "abstract":
        append_14_site_patch_abstract_encoder(
            circuit,
            patch_qubits=patch_qubits,
            injected_state=injected_state,
            add_barriers=add_barriers,
        )
    elif encoder == "abstract_routed":
        append_14_site_patch_abstract_routed_encoder(
            circuit,
            patch_qubits=patch_qubits,
            injected_state=injected_state,
            flag_bits=encoder_flags,
            add_barriers=add_barriers,
        )
    elif encoder == "hirano":
        append_14_site_patch_hirano_encoder(
            circuit,
            patch_qubits=patch_qubits,
            injected_state=injected_state,
            flag_bits=encoder_flags,
            add_barriers=add_barriers,
        )
    else:
        raise ValueError("encoder must be 'abstract', 'abstract_routed', or 'hirano'.")
    if add_barriers:
        circuit.barrier(label="state injection")

    if add_qed_round:
        assert qed_bits is not None
        append_14_site_patch_superdense_stabilizer_measurement(
            circuit,
            patch_qubits=patch_qubits,
            stabilizer_bits=qed_bits[:STEANE_QED_MEASUREMENTS_PER_ROUND],
            d2_copy_flag_bit=qed_bits[-1],
            basis="BOTH",
            add_barriers=add_barriers,
        )
        if add_barriers:
            circuit.barrier(label="post-injection QED")

    for round_bits in cultivation_bits:
        append_14_site_patch_magic_state_projection_round(
            circuit,
            patch_qubits=patch_qubits,
            bits=round_bits,
            cultivated_state=cultivated_state,
            add_barriers=add_barriers,
        )


def build_11_site_patch_injection_and_cultivation_circuit(
    injected_state: InjectedStateName = "Ty",
    *,
    add_qed_round: bool,
    skip_cx_1_4: bool,
    cultivated_state: CultivatedStateName = "Ty",
    cultivation_rounds: int = 1,
    add_barriers: bool = True,
    skip_cx: tuple[int, int] | None = None,
) -> QuantumCircuit:
    """This is the authoritative Qiskit register-allocation and appending example.

    It is the register-owning counterpart to
    :func:`append_steane_injection_and_cultivation`. It allocates one 11-qubit
    patch register, one four-bit encoder-flag register, an optional six-bit
    post-injection QED register, and one five-bit result register per
    cultivation round, then delegates all circuit construction to the appender.

    When selected, the QED round matches Hirano's full protocol and the
    state-preparation block in Rosenfeld Fig. 3. Its syndrome is postselection
    data; the circuit performs no error correction.

    Register layout
    ---------------
    ``q``
        Eleven qubits ordered by the cultivation patch site labels ``0..10``.
    ``enc``
        Four encoder-flag postselection bits.
    ``qed``
        Six post-injection Steane stabilizer outcomes when ``add_qed_round`` is
        true, ordered as the three X checks followed by the three Z checks.
    ``cult_0``, ``cult_1``, ...
        Five postselection bits for each cultivation round.

    Parameters
    ----------
    injected_state
        Physical state injected on the center data site before encoding.
    cultivated_state
        Magic state selected by each projection round. The Qiskit default is
        ``"Ty"``; pass ``"T"`` for the prior H_XY protocol.
    cultivation_rounds
        Number of selected magic-state cultivation rounds. Zero appends
        injection and optional QED without cultivation.
    add_qed_round
        Whether to measure all six Steane stabilizers after injection and
        before cultivation.
    add_barriers
        Whether to insert barriers between conceptual circuit stages.
    skip_cx_1_4
        If true, omit the CX 1->4 gate from every fold/unfold network.
    skip_cx
        If given, omit the specified ``(control, target)`` CX gate.

    Returns
    -------
    QuantumCircuit
        Register-owning circuit containing injection and all requested
        cultivation rounds.
    """
    if cultivation_rounds < 0:
        raise ValueError("cultivation_rounds must be non-negative.")

    # Quantum allocation: one 11-qubit qreg ordered by patch sites 0..10.
    qreg = QuantumRegister(NUM_QUBITS_PATCH_11_SITES, "q")

    # Classical allocation:
    # - one 4-bit creg for the state-injection encoder-flag measurements;
    # - one 6-bit creg for post-injection Steane QED, when enabled;
    # - one 5-bit creg for each cultivation round's root and four helper
    #   measurements.
    enc_flags_creg = ClassicalRegister(len(PATCH_11_SITE_HELPER_SITES), "enc_flags")
    qed_creg = (
        ClassicalRegister(STEANE_QED_MEASUREMENTS_PER_ROUND, "qed") if add_qed_round else None
    )
    cult_cregs = [
        ClassicalRegister(
            PATCH_11_SITE_CULTIVATION_MEASUREMENTS_PER_ROUND,
            f"cult_{round_index}",
        )
        for round_index in range(cultivation_rounds)
    ]
    cregs = [enc_flags_creg]
    if qed_creg is not None:
        cregs.append(qed_creg)
    cregs.extend(cult_cregs)
    circuit = QuantumCircuit(qreg, *cregs)

    append_11_site_patch_injection_and_cultivation(
        circuit,
        patch_qubits=list(qreg),
        injected_state=injected_state,
        cultivated_state=cultivated_state,
        encoder_bits=list(enc_flags_creg),
        cultivation_bits=[list(cult_creg) for cult_creg in cult_cregs],
        cultivation_rounds=cultivation_rounds,
        qed_bits=(list(qed_creg) if qed_creg is not None else None),
        add_qed_round=add_qed_round,
        add_barriers=add_barriers,
        skip_cx_1_4=skip_cx_1_4,
        skip_cx=skip_cx,
    )
    return circuit


def build_14_site_patch_injection_and_cultivation_circuit(
    injected_state: InjectedStateName = "Ty",
    *,
    add_qed_round: bool,
    encoder: Patch14SiteEncoderName = "abstract_routed",
    cultivated_state: CultivatedStateName = "Ty",
    cultivation_rounds: int = 1,
    add_barriers: bool = True,
) -> QuantumCircuit:
    """Build the register-owning 14-site injection-and-cultivation circuit.

    The optional QED register has seven bits: six Steane stabilizers followed
    by the inferred terminal D2' copy-detour flag.
    """
    if cultivation_rounds < 0:
        raise ValueError("cultivation_rounds must be non-negative.")

    qreg = QuantumRegister(NUM_QUBITS_PATCH_14_SITES, "q")
    encoder_flag_count = patch_14_site_encoder_flag_count(encoder)
    enc_flags_creg = (
        ClassicalRegister(encoder_flag_count, "enc_flags") if encoder_flag_count else None
    )
    qed_creg = (
        ClassicalRegister(PATCH_14_SITE_QED_MEASUREMENTS_PER_ROUND, "qed")
        if add_qed_round
        else None
    )
    cult_cregs = [
        ClassicalRegister(
            PATCH_14_SITE_CULTIVATION_MEASUREMENTS_PER_ROUND,
            f"cult_{round_index}",
        )
        for round_index in range(cultivation_rounds)
    ]
    cregs: list[ClassicalRegister] = []
    if enc_flags_creg is not None:
        cregs.append(enc_flags_creg)
    if qed_creg is not None:
        cregs.append(qed_creg)
    cregs.extend(cult_cregs)
    circuit = QuantumCircuit(qreg, *cregs)

    append_14_site_patch_injection_and_cultivation(
        circuit,
        patch_qubits=list(qreg),
        encoder=encoder,
        injected_state=injected_state,
        cultivated_state=cultivated_state,
        encoder_bits=(list(enc_flags_creg) if enc_flags_creg is not None else ()),
        cultivation_bits=[list(cult_creg) for cult_creg in cult_cregs],
        cultivation_rounds=cultivation_rounds,
        qed_bits=(list(qed_creg) if qed_creg is not None else None),
        add_qed_round=add_qed_round,
        add_barriers=add_barriers,
    )
    return circuit


# =============================================================================
# 4. Postselection helpers
# =============================================================================


def postselect_success_counts(
    counts: dict[str, int],
    postselection_bit_positions: Sequence[int] | None = None,
) -> dict[str, int]:
    """Filter counts to shots passing the selected error-detection checks.

    In Qiskit's convention, X-basis measurement outcome 0 means eigenvalue +1.
    Successful injection and cultivation requires zero-valued encoder-flag,
    post-injection QED, folded-logical, and cultivation-helper outcomes.

    Parameters
    ----------
    counts
        Qiskit counts dictionary from a cultivation circuit run.
    postselection_bit_positions
        Indices of the postselection bits within the bitstring, using Qiskit's
        convention where index 0 is the *rightmost* character. Only these
        positions are checked for the all-zero acceptance condition; any other
        bits, such as a subsequent logical measurement, are ignored.

        If ``None`` (default), the *entire* bitstring is required to be
        all-zero.  This is the correct behaviour when the circuit contains
        only cultivation measurements.

    Returns
    -------
    dict[str, int]
        Filtered dictionary containing only shots that passed the cultivation
        postselection check.
    """
    accepted: dict[str, int] = {}
    for bitstring, count in counts.items():
        bs = bitstring.replace(" ", "")
        if postselection_bit_positions is None:
            if set(bs) <= {"0"}:
                accepted[bitstring] = count
        else:
            if all(qiskit_bit_value(bitstring, i) == 0 for i in postselection_bit_positions):
                accepted[bitstring] = count
    return accepted
