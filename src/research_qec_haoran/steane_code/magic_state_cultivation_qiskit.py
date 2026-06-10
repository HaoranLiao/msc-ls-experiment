from __future__ import annotations

from collections.abc import Sequence

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

from research_qec_haoran.steane_code.state_injection_qiskit import (
    InjectedStateName,
    append_patch_steane_state_injection_encoder,
)
from research_qec_haoran.utils.gates import apply_ty, apply_ty_dagger


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
    (
        (7, 4),
    ),
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
        circuit.cx(patch_qubits[control_site], patch_qubits[target_site])


def _filter_fold_layer(
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
        filtered = _filter_fold_layer(layer, skip_cx_1_4=skip_cx_1_4, skip_cx=skip_cx)
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
        filtered = _filter_fold_layer(layer, skip_cx_1_4=skip_cx_1_4, skip_cx=skip_cx)
        append_cnot_layer_from_site_labels(
            circuit,
            patch_qubits,
            list(reversed(filtered)),
        )
        if add_barriers:
            circuit.barrier(label=f"unfold step {reverse_step_index}")


def append_hadamard_axis_projection_round(
    circuit: QuantumCircuit,
    *,
    patch_qubits: Sequence,
    classical_bits: Sequence,
    add_barriers: bool = True,
    skip_cx_1_4: bool = False,
    skip_cx: tuple[int, int] | None = None,
) -> None:
    """Append one H_L projection / cultivation round.

    See SITE_LAYOUT_DOC for the 11-site labeling.

    This routine implements the Google paper Fig.1 panel-C cultivation sequence:

    1. Prepare the four helper sites in |+>.
    2. Apply T_y = RY(pi/4) to all seven data sites.
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
    14. Apply T_y^dagger = RY(-pi/4) to all seven data sites.

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
    if len(patch_qubits) != 11:
        raise ValueError("patch_qubits must contain exactly 11 qubits.")

    if len(classical_bits) != 5:
        raise ValueError("classical_bits must contain exactly 5 bits.")

    q = list(patch_qubits)
    c = list(classical_bits)

    # Step 2 helper preparation.
    for helper_site in HELPER_SITES:
        reset_to_plus(circuit, q[helper_site])

    if add_barriers:
        circuit.barrier(label="prepare helpers")

    # Step 2 data rotations. The figure labels these as T gates,
    # but in this cultivation context they are Ty = RY(pi/4).
    for data_site in DATA_SITES:
        apply_ty(circuit, q[data_site])

    if add_barriers:
        circuit.barrier(label="Ty on data")

    # Steps 3--6: fold.
    append_fold_network(circuit, q, add_barriers=add_barriers, skip_cx_1_4=skip_cx_1_4, skip_cx=skip_cx)

    # Step 7: measure the folded observable on site 7.
    measure_in_x_basis(circuit, q[FOLDED_MEASUREMENT_SITE], c[0])

    if add_barriers:
        circuit.barrier(label="measure folded observable")

    # Reprepare site 7 to make the time-reversed unfold explicit.
    reset_to_plus(circuit, q[FOLDED_MEASUREMENT_SITE])

    if add_barriers:
        circuit.barrier(label="reset folded site")

    # Steps 8--12: unfold.
    append_unfold_network(circuit, q, add_barriers=add_barriers, skip_cx_1_4=skip_cx_1_4, skip_cx=skip_cx)

    # Step 13: measure the four helper / M qubits.
    for bit_index, helper_site in enumerate(HELPER_SITES, start=1):
        measure_in_x_basis(circuit, q[helper_site], c[bit_index])

    if add_barriers:
        circuit.barrier(label="measure helpers")

    # Step 14: undo the single-qubit basis rotation on the data.
    for data_site in DATA_SITES:
        apply_ty_dagger(circuit, q[data_site])

    if add_barriers:
        circuit.barrier(label="Ty dagger on data")


# =============================================================================
# 3. Full circuit builder
# =============================================================================

def build_steane_injection_with_h_cultivation(
    *,
    add_barriers: bool = True,
    skip_cx_1_4: bool = False,
    skip_cx: tuple[int, int] | None = None,
) -> QuantumCircuit:
    """Build the full 11-site Steane injection + H_L projection circuit.

    See SITE_LAYOUT_DOC for the full 11-site layout diagram.
        DATA_SITES = (0, 3, 4, 5, 8, 9, 10)
        HELPER_SITES = (1, 2, 6, 7)

    The encoder prepares |H_L⟩ directly on the seven data sites:
    - Site 0 receives the physical |H⟩ state.
    - Sites 5, 9, 10 are prepared in |+⟩.
    - Sites 3, 4, 8 remain in |0⟩.

    The projection round measures the logical Hadamard-axis observable via
        H_L = T_y^dagger X_L T_y,  with  T_y = RY(pi/4).

    The fold schedule is
        step 3: CX 2->0, (1->4), 6->3, 7->10
        step 4: CX 6->8, 4->2, 7->5
        step 5: CX 4->6, 7->9
        step 6: CX 7->4
    followed by X-basis measurement of site 7, reset of site 7, exact inverse
    unfolding, X-basis measurement of helper sites 1, 2, 6, 7, and
    finally T_y^dagger on all data sites.

    Parameters
    ----------
    add_barriers
        Whether to insert barriers between conceptual stages.
    skip_cx_1_4
        If True, omit the CX 1->4 gate from the fold/unfold network.
    skip_cx
        If given, omit the specified (control, target) CX gate.
        Takes precedence over skip_cx_1_4.

    Returns
    -------
    QuantumCircuit
        An 11-qubit Qiskit circuit with 5 classical measurement bits.
    """
    qreg = QuantumRegister(11, "q")
    creg = ClassicalRegister(5, "c")
    circuit = QuantumCircuit(qreg, creg)

    append_patch_steane_state_injection_encoder(
        circuit,
        patch_qubits=list(qreg),  # type: ignore[arg-type]
        injected_state="H",
        add_barriers=add_barriers,
    )

    append_hadamard_axis_projection_round(
        circuit,
        patch_qubits=list(qreg),  # type: ignore[arg-type]
        classical_bits=list(creg),  # type: ignore[arg-type]
        add_barriers=add_barriers,
        skip_cx_1_4=skip_cx_1_4,
        skip_cx=skip_cx,
    )

    return circuit


# =============================================================================
# 4. Postselection helpers
# =============================================================================

def postselect_success_counts(counts: dict[str, int]) -> dict[str, int]:
    """Filter a Qiskit counts dictionary to successful cultivation shots.

    In Qiskit's convention, X-basis measurement outcome 0 means eigenvalue +1.
    The desired successful postselection condition is

        classical_bits[:] == 00000

    i.e. the folded logical measurement (bit 0, site 7) is +1 and all four
    helper/flag measurements (bits 1-4, sites 1, 2, 6, 7) are +1.

    Since the success condition is all-zero, Qiskit's endian convention for
    displaying classical registers does not matter here.

    Parameters
    ----------
    counts
        Qiskit counts dictionary from a cultivation circuit run.

    Returns
    -------
    dict[str, int]
        Filtered dictionary containing only successful (all-zero) bitstrings.
    """
    return {
        bitstring: count
        for bitstring, count in counts.items()
        if set(bitstring.replace(" ", "")) <= {"0"}
    }


if __name__ == "__main__":
    qc = build_steane_injection_with_h_cultivation(add_barriers=True)

    print(SITE_LAYOUT_DOC)
    print(qc.draw(output="text", fold=-1))
