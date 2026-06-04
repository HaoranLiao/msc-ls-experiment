from __future__ import annotations

from collections.abc import Sequence
from math import pi, sqrt
from typing import Literal

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister


InjectedStateName = Literal["H", "T", "0", "1", "+", "-", "Y+", "Y-"]


# =============================================================================
# 1. Steane state-injection encoder (expressed in 11-site patch labels)
# =============================================================================

# The encoder prepares |ψ_L⟩ on the seven data sites of the 11-site patch.
#
# Input preparation (all in 11-site labels):
#   site 0  = |ψ⟩  (injected state)
#   site 5  = |+⟩
#   site 9  = |+⟩
#   site 10 = |+⟩
#   sites 3, 4, 8 = |0⟩  (default)
#
# The CNOT list below is the Steane encoder expressed directly in the
# 11-site patch labels (derived from Hirano et al., arXiv:2510.24615,
# Figure 5 / 6(a), mapped onto the triangular patch geometry).

INJECTION_SITE: int = 0
PLUS_SITES: tuple[int, ...] = (5, 9, 10)

ENCODER_CNOTS: tuple[tuple[int, int], ...] = (
    (10, 3),
    (0, 10),
    (5, 0),
    (3, 4),
    (3, 8),
    (0, 9),
    (9, 8),
    (9, 5),
    (5, 4),
    (0, 3),
)


def prepare_single_qubit_state(
    circuit: QuantumCircuit,
    qubit,
    state: InjectedStateName | Sequence[complex],
) -> None:
    """Prepare the unencoded one-qubit input state.

    Parameters
    ----------
    circuit
        Circuit being modified.
    qubit
        Qubit on which the unencoded state is prepared.
    state
        Either a named state or a custom length-2 vector ``[alpha, beta]``.

    Notes
    -----
    The supported named states are:

    ``"H"``
        The +1 eigenstate of the Hadamard operator,

        ``cos(pi/8)|0> + sin(pi/8)|1>``.

        This is prepared as ``RY(pi/4)|0>``.

    ``"T"``
        The standard circuit-model T state,

        ``(|0> + exp(i*pi/4)|1>) / sqrt(2)``.

        This is prepared as ``T H |0>``.

    ``"Y+"``
        The +1 eigenstate of Pauli Y,

        ``(|0> + i|1>) / sqrt(2)``.

    ``"Y-"``
        The -1 eigenstate of Pauli Y,

        ``(|0> - i|1>) / sqrt(2)``.

    If a custom vector is given, it is normalized before initialization.
    """
    if isinstance(state, str):
        s = state.upper()

        if s == "0":
            return

        if s == "1":
            circuit.x(qubit)
            return

        if s == "+":
            circuit.h(qubit)
            return

        if s == "-":
            circuit.x(qubit)
            circuit.h(qubit)
            return

        if s == "Y+":
            circuit.h(qubit)
            circuit.s(qubit)
            return

        if s == "Y-":
            circuit.h(qubit)
            circuit.sdg(qubit)
            return

        if s == "T":
            circuit.h(qubit)
            circuit.t(qubit)
            return

        if s == "H":
            circuit.ry(pi / 4, qubit)
            return

        raise ValueError(f"Unknown injected state name: {state!r}")

    if len(state) != 2:
        raise ValueError("Custom state must be a length-2 vector [alpha, beta].")

    alpha = complex(state[0])
    beta = complex(state[1])
    norm = sqrt(abs(alpha) ** 2 + abs(beta) ** 2)

    if norm == 0:
        raise ValueError("Cannot initialize the zero vector.")

    circuit.initialize([alpha / norm, beta / norm], [qubit])


def append_steane_state_injection_encoder(
    circuit: QuantumCircuit,
    *,
    patch_qubits: Sequence,
    injected_state: InjectedStateName | Sequence[complex] = "H",
) -> None:
    """Append the Steane-code state-injection encoder on the 11-site patch.

    Parameters
    ----------
    circuit
        Circuit being modified.
    patch_qubits
        Eleven Qiskit qubits indexed by the 11-site patch labels.
    injected_state
        The physical one-qubit state to inject on site 0.

    Notes
    -----
    The encoder prepares |ψ_L⟩ on the seven data sites using the
    11-site patch labels directly:

    - Site 0 receives the injected state |ψ⟩.
    - Sites 5, 9, 10 are prepared in |+⟩.
    - Sites 3, 4, 8 remain in |0⟩.

    With ``injected_state="H"``, the output is ``|H_L⟩``.
    With ``injected_state="T"``, the output is ``|T_L⟩``.
    """
    if len(patch_qubits) != 11:
        raise ValueError("patch_qubits must contain exactly 11 qubits.")

    q = list(patch_qubits)

    # Injected state on site 0.
    prepare_single_qubit_state(circuit, q[INJECTION_SITE], injected_state)

    # |+⟩ preparation on sites 5, 9, 10.
    for site in PLUS_SITES:
        circuit.h(q[site])

    circuit.barrier(label="Steane injection inputs")

    # Encoder CNOTs (all in 11-site labels).
    for control, target in ENCODER_CNOTS:
        circuit.cx(q[control], q[target])

    circuit.barrier(label="encoded Steane state")


# =============================================================================
# 2. Corrected 11-site cultivation patch
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
    CX 4 -> 7

The folded observable is then measured on site 4 in the X basis.

The unfold is the exact inverse of the above CNOT schedule.

Step 13 measures the four helper/measurement sites

    1, 2, 6, 7

in the X basis.

Finally, the inverse single-qubit rotation is applied to all data sites.
"""


DATA_SITES: tuple[int, ...] = (0, 3, 4, 5, 8, 9, 10)
HELPER_SITES: tuple[int, ...] = (1, 2, 6, 7)
FOLDED_MEASUREMENT_SITE: int = 4



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
        (4, 7),
    ),
)


def apply_google_ty(circuit: QuantumCircuit, qubit) -> None:
    """Apply Google's cultivation ``T`` gate.

    Notes
    -----
    In the cultivation figure, the gate labeled ``T`` is not Qiskit's
    standard phase gate ``diag(1, exp(i*pi/4))``.

    It is

    ``T_y = exp(-i*pi*Y/8) = RY(pi/4)``.

    This is the rotation satisfying

    ``H = T_y^dagger X T_y``.

    Therefore measuring ``H_L`` can be implemented by

    1. applying ``T_y`` to each data qubit,
    2. measuring the mapped logical ``X`` observable,
    3. applying ``T_y^dagger`` to each data qubit.
    """
    circuit.ry(pi / 4, qubit)


def apply_google_ty_dagger(circuit: QuantumCircuit, qubit) -> None:
    """Apply ``T_y^dagger = RY(-pi/4)``."""
    circuit.ry(-pi / 4, qubit)


def measure_x_basis(circuit: QuantumCircuit, qubit, classical_bit) -> None:
    """Measure ``qubit`` in the X basis.

    Parameters
    ----------
    circuit
        Circuit being modified.
    qubit
        Qubit to measure.
    classical_bit
        Classical bit receiving the measurement outcome.

    Notes
    -----
    Qiskit's computational measurement convention is used after applying
    a Hadamard. Thus outcome ``0`` means the measured X eigenvalue is ``+1``.
    """
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
        Legacy shorthand for ``skip_cx=(1, 4)``.
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

    Parameters
    ----------
    circuit
        Circuit being modified.
    patch_qubits
        Qubits indexed by the 11-site labels.
    add_barriers
        Whether to insert barriers after each fold layer.
    skip_cx_1_4
        If ``True``, omit the ``CX 1 -> 4`` gate from the fold (and
        correspondingly from the unfold when using
        :func:`append_unfold_network` with the same flag).

    Notes
    -----
    This implements panel-C steps 3--6:

    step 3: four CNOTs,
    step 4: three CNOTs,
    step 5: two CNOTs,
    step 6: one CNOT.
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

    Parameters
    ----------
    circuit
        Circuit being modified.
    patch_qubits
        Qubits indexed by the 11-site labels.
    add_barriers
        Whether to insert barriers after each inverse layer.
    skip_cx_1_4
        If ``True``, omit the ``CX 1 -> 4`` gate (matching the forward fold).

    Notes
    -----
    This implements the time-reversed part of the figure, panel-C
    steps 8--12. Since CNOT is self-inverse, the inverse network is the
    same CNOTs in exactly reverse order.
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
    """Append one Google-style ``H_L`` projection / cultivation round.

    Parameters
    ----------
    circuit
        Circuit being modified.
    patch_qubits
        Eleven Qiskit qubits ordered by the 11-site labels described below.
    classical_bits
        Five classical bits.

        ``classical_bits[0]``
            X-basis measurement of the folded logical observable at site 4.

        ``classical_bits[1]``
            X-basis measurement of helper site 1.

        ``classical_bits[2]``
            X-basis measurement of helper site 2.

        ``classical_bits[3]``
            X-basis measurement of helper site 6.

        ``classical_bits[4]``
            X-basis measurement of helper site 7.

    add_barriers
        Whether to insert barriers between conceptual stages.

    11-site labeling
    ----------------
    The patch is labeled as

    ::

                  0

              1       2

          3       4       5

              6       7

          8       9       10

    The data sites are

    ::

        DATA_SITES = (0, 3, 4, 5, 8, 9, 10)

    The helper/measurement sites are

    ::

        HELPER_SITES = (1, 2, 6, 7)

    Circuit structure
    -----------------
    This routine implements the corrected panel-C cultivation sequence:

    1. Prepare the four helper sites in ``|+>``.
    2. Apply ``T_y = RY(pi/4)`` to all seven data sites.
    3. Apply the 4-CNOT fold layer.
    4. Apply the 3-CNOT fold layer.
    5. Apply the 2-CNOT fold layer.
    6. Apply the 1-CNOT fold layer.
    7. Measure site 4 in the X basis.
    8. Reset site 4 to ``|+>`` for the time-reversed unfold.
    9. Apply the inverse 1-CNOT layer.
    10. Apply the inverse 2-CNOT layer.
    11. Apply the inverse 3-CNOT layer.
    12. Apply the inverse 4-CNOT layer.
    13. Measure helper sites ``1, 2, 6, 7`` in the X basis.
    14. Apply ``T_y^dagger = RY(-pi/4)`` to all seven data sites.

    Postselection
    -------------
    In Qiskit's convention, after X-basis measurement, bit value ``0``
    corresponds to the ``+1`` eigenvalue. Therefore the desired successful
    postselection condition is

    ::

        classical_bits[:] == 00000

    i.e. the folded logical measurement is ``+1`` and all four helper/flag
    measurements are ``+1``.
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
        apply_google_ty(circuit, q[data_site])

    if add_barriers:
        circuit.barrier(label="Ty on data")

    # Steps 3--6: fold.
    append_fold_network(circuit, q, add_barriers=add_barriers, skip_cx_1_4=skip_cx_1_4, skip_cx=skip_cx)

    # Step 7: measure the folded observable on site 4.
    measure_x_basis(circuit, q[FOLDED_MEASUREMENT_SITE], c[0])

    if add_barriers:
        circuit.barrier(label="measure folded observable")

    # Reprepare site 4 to make the time-reversed unfold explicit.
    reset_to_plus(circuit, q[FOLDED_MEASUREMENT_SITE])

    if add_barriers:
        circuit.barrier(label="reset folded site")

    # Steps 8--12: unfold.
    append_unfold_network(circuit, q, add_barriers=add_barriers, skip_cx_1_4=skip_cx_1_4, skip_cx=skip_cx)

    # Step 13: measure the four helper / M qubits.
    for bit_index, helper_site in enumerate(HELPER_SITES, start=1):
        measure_x_basis(circuit, q[helper_site], c[bit_index])

    if add_barriers:
        circuit.barrier(label="measure helpers")

    # Step 14: undo the single-qubit basis rotation on the data.
    for data_site in DATA_SITES:
        apply_google_ty_dagger(circuit, q[data_site])

    if add_barriers:
        circuit.barrier(label="Ty dagger on data")


# =============================================================================
# 3. Full circuit builder
# =============================================================================

def build_steane_injection_with_h_cultivation(
    *,
    injected_state: InjectedStateName | Sequence[complex] = "H",
    add_barriers: bool = True,
    skip_cx_1_4: bool = False,
    skip_cx: tuple[int, int] | None = None,
) -> QuantumCircuit:
    """Build the full 11-site Steane injection + ``H_L`` projection circuit.

    Parameters
    ----------
    injected_state
        Physical single-qubit state to inject before encoding. The intended
        cultivation use case is ``"H"``, which prepares a noisy logical
        ``|H_L>`` and then projects toward the ``+1`` eigenspace of ``H_L``.

        Other choices such as ``"T"`` are allowed for testing the encoder,
        but the cultivation/projection stage is specifically an ``H_L``
        projection circuit.

    add_barriers
        Whether to insert barriers between conceptual stages.

    Returns
    -------
    QuantumCircuit
        An 11-qubit Qiskit circuit with 5 classical measurement bits.

    11-site patch layout
    --------------------
    The circuit uses the following site labels:

    ::

                  0

              1       2

          3       4       5

              6       7

          8       9       10

    The seven data sites are

    ::

        DATA_SITES = (0, 3, 4, 5, 8, 9, 10)

    The four helper / measurement sites are

    ::

        HELPER_SITES = (1, 2, 6, 7)

    State injection
    ---------------
    The encoder prepares |ψ_L⟩ directly on the seven data sites:

    - Site 0 receives the injected state |ψ⟩.
    - Sites 5, 9, 10 are prepared in |+⟩.
    - Sites 3, 4, 8 remain in |0⟩.

    For ``injected_state="H"``, this produces ``|H_L⟩``.

    Cultivation / projection
    ------------------------
    The projection round measures the logical Hadamard-axis observable via

    ::

        H_L = T_y^dagger X_L T_y,

    with

    ::

        T_y = exp(-i*pi*Y/8) = RY(pi/4).

    The fold schedule is

    ::

        step 3:
            CX 2 -> 0
            CX 1 -> 4
            CX 6 -> 3
            CX 7 -> 10

        step 4:
            CX 6 -> 8
            CX 4 -> 2
            CX 7 -> 5

        step 5:
            CX 4 -> 6
            CX 7 -> 9

        step 6:
            CX 4 -> 7

    followed by X-basis measurement of site 4, reset of site 4, exact inverse
    unfolding, X-basis measurement of helper sites ``1, 2, 6, 7``, and
    finally ``T_y^dagger`` on all data sites.

    Postselection
    -------------
    Keep shots with all five classical bits equal to zero:

    ::

        c = 00000

    assuming the Qiskit convention where X-basis measurement outcome ``0``
    means eigenvalue ``+1``.
    """
    qreg = QuantumRegister(11, "q")
    creg = ClassicalRegister(5, "c")
    circuit = QuantumCircuit(qreg, creg)

    q = list(qreg)
    c = list(creg)

    append_steane_state_injection_encoder(
        circuit,
        patch_qubits=q,
        injected_state=injected_state,
    )

    append_hadamard_axis_projection_round(
        circuit,
        patch_qubits=q,
        classical_bits=c,
        add_barriers=add_barriers,
        skip_cx_1_4=skip_cx_1_4,
        skip_cx=skip_cx,
    )

    return circuit


# =============================================================================
# 4. Postselection helpers
# =============================================================================

def is_successful_cultivation_bitstring(bitstring: str) -> bool:
    """Return whether a Qiskit count key passes the cultivation postselection.

    Parameters
    ----------
    bitstring
        A Qiskit counts key, for example ``"00000"``.

    Returns
    -------
    bool
        ``True`` if all measured postselection bits are zero.

    Notes
    -----
    Since the success condition is all-zero, Qiskit's endian convention for
    displaying classical registers does not matter here.
    """
    compact = bitstring.replace(" ", "")
    return len(compact) == 5 and set(compact) <= {"0"}


def postselect_success_counts(counts: dict[str, int]) -> dict[str, int]:
    """Filter a Qiskit counts dictionary to successful cultivation shots."""
    return {
        bitstring: count
        for bitstring, count in counts.items()
        if is_successful_cultivation_bitstring(bitstring)
    }


# =============================================================================
# 5. Example usage
# =============================================================================

if __name__ == "__main__":
    qc = build_steane_injection_with_h_cultivation(
        injected_state="H",
        add_barriers=True,
    )

    print(SITE_LAYOUT_DOC)
    print(qc.draw(output="text", fold=-1))
