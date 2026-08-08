from __future__ import annotations

from collections.abc import Sequence

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

from msc.steane_code.decoding.decoder import (
    X_STABILIZER_LABELS,
    Z_STABILIZER_LABELS,
)
from msc.steane_code.state_injection_qiskit import (
    NUM_PATCH_QUBITS,
    InjectedStateName,
    append_patch_stabilizer_measurement,
    append_patch_steane_state_injection_encoder_crumble_sequences,
)
from msc.utils.normalize import qiskit_bit_value, resolve_patch_qubits

# =============================================================================
# 11-site cultivation patch layout
# =============================================================================

SITE_LAYOUT_DOC = """
11-site labeling used in the cultivation patch
----------------------------------------------

The physical sites are labeled as

              0

          1       2

      3       4       5

          6       7

      8       9       10

The seven Steane/color-code data sites are

    DATA_SITES = (0, 3, 4, 5, 8, 9, 10)

and the four helper/measurement sites are

    HELPER_SITES = (1, 2, 6, 7).

The step-3-to-step-6 fold schedule from the figure is a 4-3-2-1
CNOT schedule:

Step 3:
    CX 2 -> 0
    CX 1 -> 4
    CX 6 -> 3
    CX 7 -> 10

Step 4:
    CX 6 -> 8
    CX 4 -> 2
    CX 7 -> 5

Step 5:
    CX 4 -> 6
    CX 7 -> 9

Step 6:
    CX 7 -> 4

The folded observable is then measured on site 7 in the X basis.

The unfold is the exact inverse of the above CNOT schedule.

Step 13 measures the four helper/measurement sites

    1, 2, 6, 7

in the X basis.

Finally, the inverse single-qubit rotation is applied to all data sites.
"""

DATA_SITES: tuple[int, ...] = (0, 3, 4, 5, 8, 9, 10)
HELPER_SITES: tuple[int, ...] = (1, 2, 6, 7)
FOLDED_MEASUREMENT_SITE: int = 7
CULTIVATION_MEASUREMENTS_PER_ROUND: int = 1 + len(HELPER_SITES)
POST_INJECTION_QED_STABILIZER_LABELS: tuple[str, ...] = X_STABILIZER_LABELS + Z_STABILIZER_LABELS
STEANE_QED_MEASUREMENTS_PER_ROUND: int = len(POST_INJECTION_QED_STABILIZER_LABELS)


# =============================================================================
# 1. Steane state-injection encoder (expressed in 11-site patch labels)
# =============================================================================

# The encoder and its patch-mapping constants (INJECTION_SITE, PLUS_SITES,
# _STEANE_TO_PATCH, append_steane_state_injection_encoder) live in
# state_injection_qiskit.py and are imported above.


# =============================================================================
# 2. Fold / unfold network
# =============================================================================

FOLD_LAYERS: tuple[tuple[tuple[int, int], ...], ...] = (
    # Step 3: four CNOTs.
    (
        (2, 0),
        (1, 4),
        (6, 3),
        (7, 10),
    ),
    # Step 4: three CNOTs.
    (
        (6, 8),
        (4, 2),
        (7, 5),
    ),
    # Step 5: two CNOTs.
    (
        (4, 6),
        (7, 9),
    ),
    # Step 6: one CNOT.
    ((7, 4),),
)


def measure_in_x_basis(circuit: QuantumCircuit, qubit, classical_bit) -> None:
    """Measure ``qubit`` in the X basis."""
    circuit.h(qubit)
    circuit.measure(qubit, classical_bit)


def reset_to_plus(circuit: QuantumCircuit, qubit) -> None:
    """Reset a qubit and prepare ``|+>``."""
    circuit.reset(qubit)
    circuit.h(qubit)


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


def filter_fold_layer(
    layer: Sequence[tuple[int, int]],
    *,
    skip_cx_1_4: bool = False,
    skip_cx: tuple[int, int] | None = None,
) -> list[tuple[int, int]]:
    """Optionally remove a specific CX gate from a fold layer.

    Parameters
    ----------
    skip_cx_1_4
        shorthand for ``skip_cx=(1, 4)``.
    skip_cx
        ``(control, target)`` pair to remove. Takes precedence over
        ``skip_cx_1_4`` if both are given.
    """
    gate_to_skip = skip_cx if skip_cx is not None else ((1, 4) if skip_cx_1_4 else None)
    if gate_to_skip is None:
        return list(layer)
    return [(c, t) for c, t in layer if (c, t) != gate_to_skip]


def append_fold_network(
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
    for layer_index, layer in enumerate(FOLD_LAYERS, start=3):
        filtered = filter_fold_layer(layer, skip_cx_1_4=skip_cx_1_4, skip_cx=skip_cx)
        append_cnot_layer_from_site_labels(circuit, patch_qubits, filtered)
        if add_barriers:
            circuit.barrier(label=f"fold step {layer_index}")


def append_unfold_network(
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
        reversed(FOLD_LAYERS),
        strict=False,
    ):
        filtered = filter_fold_layer(layer, skip_cx_1_4=skip_cx_1_4, skip_cx=skip_cx)
        append_cnot_layer_from_site_labels(
            circuit,
            patch_qubits,
            list(reversed(filtered)),
        )
        if add_barriers:
            circuit.barrier(label=f"unfold step {reverse_step_index}")


def append_h_xy_projection_round(
    circuit: QuantumCircuit,
    *,
    patch_qubits: Sequence,
    classical_bits: Sequence,
    add_barriers: bool = True,
    skip_cx_1_4: bool = False,
    skip_cx: tuple[int, int] | None = None,
) -> None:
    """Append one H_XY projection / cultivation round.

    See SITE_LAYOUT_DOC for the 11-site labeling.

    This routine implements the Google paper Fig.1 panel-C cultivation sequence:

    1. Prepare the four helper sites in |+>.
    2. Apply T to all seven data sites to rotate H_XY onto X.
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
    14. Apply T^dagger to all seven data sites.

    Parameters
    ----------
    circuit
        Circuit being modified.
    patch_qubits
        Eleven Qiskit qubits ordered by the 11-site labels.
    classical_bits
        Five classical bits: bit 0 = folded observable (site 7),
        bits 1-4 = helper sites 1, 2, 6, 7.
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
        expected_count=NUM_PATCH_QUBITS,
    )

    if len(classical_bits) != CULTIVATION_MEASUREMENTS_PER_ROUND:
        raise ValueError(
            "classical_bits must contain exactly " f"{CULTIVATION_MEASUREMENTS_PER_ROUND} bits."
        )

    c = list(classical_bits)

    # Step 2 helper preparation.
    for helper_site in HELPER_SITES:
        reset_to_plus(circuit, patch_qubits[helper_site])

    if add_barriers:
        circuit.barrier(label="prepare helpers")

    # Step 2 data rotations. Transversal T rotates H_XY onto the folded
    # X measurement axis; the fold CNOT network is unchanged.
    for data_site in DATA_SITES:
        circuit.t(patch_qubits[data_site])

    if add_barriers:
        circuit.barrier(label="T on data")

    # Steps 3--6: fold.
    append_fold_network(
        circuit,
        patch_qubits,
        add_barriers=add_barriers,
        skip_cx_1_4=skip_cx_1_4,
        skip_cx=skip_cx,
    )

    # Step 7: measure the folded observable on site 7.
    measure_in_x_basis(
        circuit,
        patch_qubits[FOLDED_MEASUREMENT_SITE],
        c[0],
    )

    if add_barriers:
        circuit.barrier(label="measure folded observable")

    # Reprepare site 7 to make the time-reversed unfold explicit.
    reset_to_plus(circuit, patch_qubits[FOLDED_MEASUREMENT_SITE])

    if add_barriers:
        circuit.barrier(label="reset folded site")

    # Steps 8--12: unfold.
    append_unfold_network(
        circuit,
        patch_qubits,
        add_barriers=add_barriers,
        skip_cx_1_4=skip_cx_1_4,
        skip_cx=skip_cx,
    )

    # Step 13: measure the four helper / M qubits.
    for bit_index, helper_site in enumerate(HELPER_SITES, start=1):
        measure_in_x_basis(
            circuit,
            patch_qubits[helper_site],
            c[bit_index],
        )

    if add_barriers:
        circuit.barrier(label="measure helpers")

    # Step 14: undo the single-qubit basis rotation on the data.
    for data_site in DATA_SITES:
        circuit.tdg(patch_qubits[data_site])

    if add_barriers:
        circuit.barrier(label="T dagger on data")


# =============================================================================
# 3. Injection-and-cultivation circuit construction
# =============================================================================


def append_steane_injection_and_cultivation(
    circuit: QuantumCircuit,
    *,
    add_qed_round: bool,
    skip_cx_1_4: bool,
    patch_qubits: Sequence,
    injected_state: InjectedStateName = "T",
    encoder_classical_bits: Sequence,
    cultivation_classical_bits: Sequence[Sequence],
    cultivation_rounds: int = 1,
    qed_classical_bits: Sequence | None = None,
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
    encoder_classical_bits
        Four classical bits for encoder helper postselection.
    cultivation_classical_bits
        Exactly ``cultivation_rounds`` 5-bit sequences, one per cultivation
        round.
    cultivation_rounds
        Number of H_XY cultivation rounds. Zero appends injection and optional
        QED without cultivation.
    qed_classical_bits
        Six bits for the post-injection QED outcomes, ordered by
        ``POST_INJECTION_QED_STABILIZER_LABELS``: the three X stabilizers
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
    if len(encoder_classical_bits) < 4:
        raise ValueError("encoder_classical_bits must contain at least 4 bits.")
    if cultivation_rounds < 0:
        raise ValueError("cultivation_rounds must be non-negative.")
    if len(cultivation_classical_bits) != cultivation_rounds:
        raise ValueError(
            "cultivation_classical_bits must contain exactly one 5-bit "
            "sequence per cultivation round."
        )
    if add_qed_round and (
        qed_classical_bits is None or len(qed_classical_bits) != STEANE_QED_MEASUREMENTS_PER_ROUND
    ):
        raise ValueError(
            "qed_classical_bits must contain exactly "
            f"{STEANE_QED_MEASUREMENTS_PER_ROUND} bits when add_qed_round is true."
        )

    append_patch_steane_state_injection_encoder_crumble_sequences(
        circuit,
        patch_qubits=patch_qubits,
        injected_state=injected_state,
        flag_classical_bits=encoder_classical_bits,
        add_barriers=add_barriers,
    )
    if add_barriers:
        circuit.barrier(label="state injection")

    if add_qed_round:
        assert qed_classical_bits is not None
        append_patch_stabilizer_measurement(
            circuit,
            patch_qubits=patch_qubits,
            classical_bits=qed_classical_bits,
            basis="both",
            add_barriers=add_barriers,
        )
        if add_barriers:
            circuit.barrier(label="post-injection QED")

    for round_index in range(cultivation_rounds):
        append_h_xy_projection_round(
            circuit,
            patch_qubits=patch_qubits,
            classical_bits=cultivation_classical_bits[round_index],
            add_barriers=add_barriers,
            skip_cx_1_4=skip_cx_1_4,
            skip_cx=skip_cx,
        )


def build_steane_injection_and_cultivation_circuit(
    injected_state: InjectedStateName = "T",
    *,
    add_qed_round: bool,
    skip_cx_1_4: bool,
    cultivation_rounds: int = 1,
    add_barriers: bool = True,
    skip_cx: tuple[int, int] | None = None,
) -> QuantumCircuit:
    """This is the authoritative Qiskit register-allocation and appending example.

    It is the register-owning counterpart to
    :func:`append_steane_injection_and_cultivation`. It allocates one 11-qubit
    patch register, one four-bit encoder-helper register, an optional six-bit
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
        Four encoder-helper postselection bits.
    ``qed``
        Six post-injection Steane stabilizer outcomes when ``add_qed_round`` is
        true, ordered as the three X checks followed by the three Z checks.
    ``cult_0``, ``cult_1``, ...
        Five postselection bits for each cultivation round.

    Parameters
    ----------
    injected_state
        Physical state injected on the center data site before encoding.
    cultivation_rounds
        Number of H_XY cultivation rounds. Zero appends injection and optional
        QED without cultivation.
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
    qreg = QuantumRegister(NUM_PATCH_QUBITS, "q")

    # Classical allocation:
    # - one 4-bit creg for the state-injection encoder helper measurements;
    # - one 6-bit creg for post-injection Steane QED, when enabled;
    # - one 5-bit creg for each cultivation round's root and four helper
    #   measurements.
    enc_creg = ClassicalRegister(len(HELPER_SITES), "enc")
    qed_creg = (
        ClassicalRegister(STEANE_QED_MEASUREMENTS_PER_ROUND, "qed") if add_qed_round else None
    )
    cult_cregs = [
        ClassicalRegister(
            CULTIVATION_MEASUREMENTS_PER_ROUND,
            f"cult_{round_index}",
        )
        for round_index in range(cultivation_rounds)
    ]
    cregs = [enc_creg]
    if qed_creg is not None:
        cregs.append(qed_creg)
    cregs.extend(cult_cregs)
    circuit = QuantumCircuit(qreg, *cregs)

    append_steane_injection_and_cultivation(
        circuit,
        patch_qubits=list(qreg),
        injected_state=injected_state,
        encoder_classical_bits=list(enc_creg),
        cultivation_classical_bits=[list(cult_creg) for cult_creg in cult_cregs],
        cultivation_rounds=cultivation_rounds,
        qed_classical_bits=(list(qed_creg) if qed_creg is not None else None),
        add_qed_round=add_qed_round,
        add_barriers=add_barriers,
        skip_cx_1_4=skip_cx_1_4,
        skip_cx=skip_cx,
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
    Successful injection and cultivation requires zero-valued encoder-helper,
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
