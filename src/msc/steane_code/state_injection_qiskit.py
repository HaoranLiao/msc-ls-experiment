from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from qiskit import QuantumCircuit
from qiskit.circuit import Qubit

from msc.steane_code.state_injection_common import (
    _STEANE_TO_11_SITE_PATCH,
    _STEANE_TO_14_SITE_PATCH,
    ABSTRACT_ENCODER_CNOT_LAYERS,
    NUM_QUBITS_PATCH_11_SITES,
    NUM_QUBITS_PATCH_14_SITES,
    PATCH_11_SITE_DEFAULT_LOGICAL_X,
    PATCH_11_SITE_DEFAULT_LOGICAL_Z,
    PATCH_11_SITE_HELPER_SITES,
    PATCH_11_SITE_ROSENFELD_ENCODER_INJECTION_SITE,
    PATCH_11_SITE_ROSENFELD_ENCODER_LAYERS,
    PATCH_11_SITE_STABILIZER_ANCILLA_SITES,
    PATCH_14_SITE_ABSTRACT_ROUTED_ENCODER_FLAG_SITES,
    PATCH_14_SITE_ABSTRACT_ROUTED_ENCODER_INJECTION_SITE,
    PATCH_14_SITE_ABSTRACT_ROUTED_ENCODER_LAYERS,
    PATCH_14_SITE_DEFAULT_LOGICAL_X,
    PATCH_14_SITE_DEFAULT_LOGICAL_Z,
    PATCH_14_SITE_HELPER_SITES,
    PATCH_14_SITE_HIRANO_ENCODER_LAYERS,
    PatchStabilizerBasis,
    select_11_site_patch_stabilizer_measurement_layers,
    select_14_site_patch_superdense_measurement_layers,
)
from msc.utils.circuit_qiskit import (
    append_logical_measurement_on_indexed_qubits,
    apply_ty,
)
from msc.utils.normalize import resolve_patch_qubits

InjectedStateName = Literal["0", "1", "+", "-", "i", "-i", "Y+", "Y-", "S", "Sdg", "T", "Tdg", "Ty"]


def prepare_physical_state(
    circuit: QuantumCircuit,
    qubit: int | Qubit,
    state: InjectedStateName,
) -> None:
    """Prepare the single-qubit state to be encoded.

    Parameters
    ----------
    circuit
        Circuit being built.
    qubit
        Qubit on which the unencoded input state is prepared.
    state
        Named state to prepare.
    """
    if state == "0":
        return

    if state == "1":
        circuit.x(qubit)
        return

    if state == "+":
        circuit.h(qubit)
        return

    if state == "-":
        circuit.x(qubit)
        circuit.h(qubit)
        return

    if state in {"i", "Y+", "S"}:
        # +1 eigenstate of Pauli Y
        # |Y+> = S|+> = (|0> + i|1>) / sqrt(2)
        circuit.h(qubit)
        circuit.s(qubit)
        return

    if state in {"-i", "Y-", "Sdg"}:
        # -1 eigenstate of Pauli Y
        # |Y-> = Sdg|+> = (|0> - i|1>) / sqrt(2)
        circuit.h(qubit)
        circuit.sdg(qubit)
        return

    if state == "T":
        # +1 eigenstate of (X+Y)/sqrt(2)
        # |T> = T|+> = (|0> + exp(i*pi/4)|1>) / sqrt(2)
        circuit.h(qubit)
        circuit.t(qubit)
        return

    if state == "Tdg":
        # +1 eigenstate of (X-Y)/sqrt(2)
        # |Tdg> = Tdg|+> = (|0> + exp(-i*pi/4)|1>) / sqrt(2)
        circuit.h(qubit)
        circuit.tdg(qubit)
        return

    if state == "Ty":
        # +1 eigenstate of H_XZ = (X+Z)/sqrt(2).
        # |Ty> = cos(pi/8)|0> + sin(pi/8)|1> = T_y|0>.
        apply_ty(circuit, qubit)
        return

    raise ValueError(f"Unknown state name: {state!r}")


def prepare_injection_input(
    circuit: QuantumCircuit,
    qubit,
    *,
    injected_state: InjectedStateName | None,
    injection_azimuth_angle: float | None,
    injection_polar_angle: float | None,
) -> None:
    """Own the injection-input contract shared by every arbitrary-state encoder.

    The three arguments are mutually exclusive and all ``None`` injects ``"T"``.
    Each angle sweeps the plane holding one cultivation axis, reaching that axis
    at ``angle=π/4``: ``Rz(angle)|+>`` sweeps XY through ``|T>`` (+1 eigenstate
    of ``H_XY``), and ``Ry(angle)|0>`` sweeps XZ through ``|Ty>`` (+1 eigenstate
    of ``H_XZ``). The polar sweep is the injected angle ``θ`` of Rosenfeld et al.
    Figs. 2 and 3.
    """
    supplied = (injected_state, injection_azimuth_angle, injection_polar_angle)
    if sum(argument is not None for argument in supplied) > 1:
        raise ValueError(
            "injected_state, injection_azimuth_angle, and injection_polar_angle are mutually "
            "exclusive; supply at most one."
        )

    if injection_azimuth_angle is not None:
        circuit.h(qubit)
        circuit.rz(injection_azimuth_angle, qubit)
    elif injection_polar_angle is not None:
        circuit.ry(injection_polar_angle, qubit)
    else:
        prepare_physical_state(
            circuit, qubit, injected_state if injected_state is not None else "T"
        )


def append_abstract_encoder(
    circuit: QuantumCircuit,
    *,
    data_qubits: Sequence,
    injected_state: InjectedStateName | None = None,
    injection_azimuth_angle: float | None = None,
    injection_polar_angle: float | None = None,
    add_barriers: bool = True,
) -> None:
    """Append a Steane-code state-injection encoder to an existing circuit.

    Implements the map
    |psi> |+> |0> |0> |+> |0> |+> -> |psi_L>
    on the seven supplied data qubits (ordered 0-6). The caller owns register
    creation and passes the target qubits via ``data_qubits``.

    Parameters
    ----------
    circuit
        Circuit being modified.
    data_qubits
        Seven qubits that will hold the encoded Steane-code state (ordered 0-6).
        The injected physical state is prepared on ``data_qubits[0]``.
    injected_state, injection_azimuth_angle, injection_polar_angle
        Mutually exclusive injection inputs prepared on qubit 0; see
        :func:`prepare_injection_input`. The output is the corresponding encoded
        state in the Steane code.
    add_barriers
        Whether to insert a barrier between preparation and encoding stages.
    """
    if len(data_qubits) < 7:
        raise ValueError("data_qubits must contain at least 7 qubits.")

    dq = list(data_qubits)

    prepare_injection_input(
        circuit,
        dq[0],
        injected_state=injected_state,
        injection_azimuth_angle=injection_azimuth_angle,
        injection_polar_angle=injection_polar_angle,
    )

    circuit.h(dq[1])
    circuit.h(dq[4])
    circuit.h(dq[6])

    if add_barriers:
        circuit.barrier(dq)

    for layer in ABSTRACT_ENCODER_CNOT_LAYERS:
        for control_qubit, target_qubit in layer:
            circuit.cx(dq[control_qubit], dq[target_qubit])


# =============================================================================
# 11-site patch (abstract and Rosenfeld encoders)
# =============================================================================
#
# The patch geometry, abstract-to-patch qubit mapping, stabilizer/logical
# supports, Rosenfeld et al. encoder schedule, and stabilizer-selection validator
# are framework-neutral and imported from ``state_injection_common``.


# ---------------------------------------------------------------------------
# Encoder on the 11-site patch
# ---------------------------------------------------------------------------


def append_11_site_patch_abstract_encoder(
    circuit: QuantumCircuit,
    *,
    patch_qubits: Sequence,
    injected_state: InjectedStateName | None = None,
    injection_azimuth_angle: float | None = None,
    injection_polar_angle: float | None = None,
    add_barriers: bool = True,
) -> None:
    """Append the Steane-code state-injection encoder on the 11-site patch.

    Composes the 7-qubit encoder from append_abstract_encoder onto the 7 data
    sites of the 11-site patch via _STEANE_TO_11_SITE_PATCH, which maps abstract
    qubits 0-6 to patch sites (0, 3, 5, 4, 8, 9, 10).

    Parameters
    ----------
    circuit
        Circuit being modified.
    patch_qubits
        Eleven Qiskit qubits indexed by the 11-site patch labels.
    injected_state, injection_azimuth_angle, injection_polar_angle
        Mutually exclusive injection inputs prepared on site 0; see
        :func:`prepare_injection_input`.
    add_barriers
        Whether to insert a barrier after the encoder.
    """
    patch_qubits = resolve_patch_qubits(
        patch_qubits,
        expected_count=NUM_QUBITS_PATCH_11_SITES,
    )

    # Map abstract encoder qubits 0–6 to the corresponding patch data sites.
    data_qubits = [patch_qubits[_STEANE_TO_11_SITE_PATCH[i]] for i in range(7)]
    append_abstract_encoder(
        circuit,
        data_qubits=data_qubits,
        injected_state=injected_state,
        injection_azimuth_angle=injection_azimuth_angle,
        injection_polar_angle=injection_polar_angle,
        add_barriers=add_barriers,
    )
    if add_barriers:
        circuit.barrier(label="state injection")


def append_11_site_patch_rosenfeld_encoder(
    circuit: QuantumCircuit,
    *,
    patch_qubits: Sequence,
    injected_state: InjectedStateName | None = None,
    injection_azimuth_angle: float | None = None,
    injection_polar_angle: float | None = None,
    flag_bits: Sequence | None = None,
    add_barriers: bool = True,
) -> QuantumCircuit:
    """Append the Rosenfeld et al. nearest-neighbor Steane encoder on the 11-site patch.

    This replays Rosenfeld et al.'s state-injection schedule after
    mapping every site into this repo's 11-site patch labels.  Unlike
    ``append_11_site_patch_abstract_encoder``, this schedule uses all four
    helper sites for nearest-neighbor routing.

    After the encoder layers the four helper sites should be in |0⟩. When
    ``flag_bits`` is provided the helpers are measured into those
    bits so the caller can postselect on all-zero outcomes.

    Parameters
    ----------
    circuit
        Circuit being modified.
    patch_qubits
        Eleven Qiskit qubits indexed by the 11-site patch labels.
    injected_state, injection_azimuth_angle, injection_polar_angle
        Mutually exclusive injection inputs prepared on patch site 4, the center
        data site corresponding to Hirano label ``0``; see
        :func:`prepare_injection_input`. The angles drive kickback-tomography
        scans.
    flag_bits
        Four classical bits for the helper-site measurements. When ``None``
        (default), the helper sites are not measured.
    add_barriers
        Whether to keep the Rosenfeld et al. tick-layer barriers.

    Returns
    -------
    QuantumCircuit
        The same circuit object, after appending the encoder.
    """
    patch_qubits = resolve_patch_qubits(
        patch_qubits,
        expected_count=NUM_QUBITS_PATCH_11_SITES,
    )
    if flag_bits is not None and len(flag_bits) < len(PATCH_11_SITE_HELPER_SITES):
        raise ValueError(f"flag_bits must contain at least {len(PATCH_11_SITE_HELPER_SITES)} bits.")

    prepare_injection_input(
        circuit,
        patch_qubits[PATCH_11_SITE_ROSENFELD_ENCODER_INJECTION_SITE],
        injected_state=injected_state,
        injection_azimuth_angle=injection_azimuth_angle,
        injection_polar_angle=injection_polar_angle,
    )
    if add_barriers:
        circuit.barrier(label="rosenfeld injection input")

    for gate_name, operands in PATCH_11_SITE_ROSENFELD_ENCODER_LAYERS:
        if gate_name == "H":
            for site in operands:
                circuit.h(patch_qubits[site])
        elif gate_name == "CZ":
            for site_a, site_b in operands:
                circuit.cz(patch_qubits[site_a], patch_qubits[site_b])
        else:  # pragma: no cover - defensive if the static schedule is edited.
            raise ValueError(f"Unsupported Rosenfeld encoder gate: {gate_name!r}")
        if add_barriers:
            circuit.barrier()
    if add_barriers:
        circuit.barrier(label="rosenfeld state injection")

    if flag_bits is not None:
        flag_bits = list(flag_bits)
        for i, site in enumerate(PATCH_11_SITE_HELPER_SITES):
            circuit.measure(patch_qubits[site], flag_bits[i])

    return circuit


# ---------------------------------------------------------------------------
# Stabilizer measurement on the 11-site patch
# ---------------------------------------------------------------------------


def append_11_site_patch_stabilizer_measurement(
    circuit: QuantumCircuit,
    *,
    patch_qubits: Sequence,
    bits: Sequence,
    basis: PatchStabilizerBasis = "BOTH",
    stabilizer_indices: Sequence[int] | None = None,
    add_barriers: bool = True,
) -> None:
    """Append non-destructive stabilizer measurements on the 11-site patch.

    Uses the patch helper sites (2, 6, 7) as ancillas to measure the selected
    X-type and/or Z-type stabilizers without disturbing the encoded state. The
    ancilla sites are reset before each measurement basis. Face 0345 routes data
    3 through helper 1, a bridge prepared opposite to the syndrome basis and
    restored (not measured); the other faces run in parallel in four CX layers.
    ``"BOTH"`` measures X then Z.

    Parameters
    ----------
    circuit
        Circuit being modified.
    patch_qubits
        Eleven qubits indexed by the 11-site patch labels.
    bits
        One bit per selected stabilizer for a single basis, or two bits per
        selected stabilizer for ``"BOTH"``. In ``"BOTH"`` mode, the first
        group holds X-stabilizer outcomes and the second holds Z-stabilizer
        outcomes.
    basis
        Stabilizer basis to measure. The default ``"BOTH"`` preserves the
        original full-syndrome behavior.
    stabilizer_indices
        Ordered subset of the three patch stabilizers to measure. ``None``
        measures all three in canonical order. This selection changes neither
        the stabilizer supports nor their associated helper ancillas.
    add_barriers
        Whether to insert a barrier between X and Z stabilizer rounds.
    """
    patch_qubits = resolve_patch_qubits(
        patch_qubits,
        expected_count=NUM_QUBITS_PATCH_11_SITES,
    )
    selected_indices, phases = select_11_site_patch_stabilizer_measurement_layers(
        basis, stabilizer_indices
    )

    required_bits = len(selected_indices) * (2 if basis == "BOTH" else 1)
    if len(bits) < required_bits:
        raise ValueError(f"bits must contain at least {required_bits} bits.")

    selected_ancilla_sites = tuple(
        PATCH_11_SITE_STABILIZER_ANCILLA_SITES[index] for index in selected_indices
    )
    for phase_index, (measurement_basis, layers) in enumerate(phases):
        if phase_index and add_barriers:
            circuit.barrier()
        for site in selected_ancilla_sites:
            circuit.reset(patch_qubits[site])
        if 0 in selected_indices:
            circuit.reset(patch_qubits[1])
        if measurement_basis == "X":
            for site in selected_ancilla_sites:
                circuit.h(patch_qubits[site])
        elif measurement_basis == "Z":
            if 0 in selected_indices:
                circuit.h(patch_qubits[1])
        else:  # pragma: no cover - shared schedule validates.
            raise ValueError(f"Unsupported measurement basis: {measurement_basis!r}.")
        for layer in layers:
            for control, target in layer:
                circuit.cx(patch_qubits[control], patch_qubits[target])
        if measurement_basis == "X":
            for site in selected_ancilla_sites:
                circuit.h(patch_qubits[site])
        for output_index, site in enumerate(selected_ancilla_sites):
            circuit.measure(
                patch_qubits[site], bits[phase_index * len(selected_indices) + output_index]
            )


# ---------------------------------------------------------------------------
# Logical measurement on the 11-site patch
# ---------------------------------------------------------------------------


def append_11_site_patch_logical_measurement(
    circuit: QuantumCircuit,
    basis: str = "Z",
    *,
    patch_qubits: Sequence,
    support: tuple[int, ...] | None = None,
    ancilla_qubit=None,
    bits: Sequence,
    destructive: bool = True,
) -> None:
    """Append a logical measurement on the 11-site patch."""
    patch_qubits = resolve_patch_qubits(
        patch_qubits,
        expected_count=NUM_QUBITS_PATCH_11_SITES,
    )
    append_logical_measurement_on_indexed_qubits(
        circuit,
        basis,
        indexed_qubits=patch_qubits,
        default_logical_x=PATCH_11_SITE_DEFAULT_LOGICAL_X,
        default_logical_z=PATCH_11_SITE_DEFAULT_LOGICAL_Z,
        support=support,
        ancilla_qubit=ancilla_qubit,
        bits=bits,
        destructive=destructive,
    )


# =============================================================================
# 14-site patch (abstract, abstract-routed, and Hirano encoders)
# =============================================================================


def append_14_site_patch_abstract_encoder(
    circuit: QuantumCircuit,
    *,
    patch_qubits: Sequence,
    injected_state: InjectedStateName | None = None,
    injection_azimuth_angle: float | None = None,
    injection_polar_angle: float | None = None,
    add_barriers: bool = True,
) -> QuantumCircuit:
    """Append the abstract Steane encoder directly on the 14-site data sites.

    This is the fixed-geometry, unrouted reference encoder. Unlike Hirano's
    specialized phase-injection schedule, the abstract Steane isometry can
    encode an arbitrary injected state: all named inputs supported by
    :func:`append_abstract_encoder`, or any swept equatorial or meridional state.
    It maps the abstract Steane triangle onto
    Hirano data labels ``(6, 4, 2, 0, 1, 5, 3)`` and never uses the six helper
    sites or D2'. In particular, D1 is data site 1 at ``(1, 5)`` from the start;
    no post-injection relocation is needed::

              x=0    x=1    x=2    x=3    x=4    x=5    x=6
       y=0                   D6/A0
       y=1                          H9            D2'
       y=2                   D4/A1         H10           D2/A2
       y=3           H7            D0/A3          H12
       y=4                  H8            H11
       y=5           D1/A4         D5/A5          D3/A6

    ``A<i>`` denotes abstract encoder qubit ``i``. The injected physical state
    is therefore prepared on D6/A0. The abstract CXs are not routed through the
    patch connectivity.

    Parameters
    ----------
    circuit
        Circuit being modified.
    patch_qubits
        Fourteen Qiskit qubits indexed by the fixed patch labels.
    injected_state, injection_azimuth_angle, injection_polar_angle
        Mutually exclusive injection inputs prepared on D6/A0, with no logical-Y
        restriction; see :func:`prepare_injection_input`.
    add_barriers
        Whether to retain the abstract encoder's preparation barrier and append
        a labeled completion barrier.

    Returns
    -------
    QuantumCircuit
        The same circuit object, after appending the encoder.
    """
    patch_qubits = resolve_patch_qubits(
        patch_qubits,
        expected_count=NUM_QUBITS_PATCH_14_SITES,
    )
    data_qubits = [patch_qubits[site] for site in _STEANE_TO_14_SITE_PATCH]
    append_abstract_encoder(
        circuit,
        data_qubits=data_qubits,
        injected_state=injected_state,
        injection_azimuth_angle=injection_azimuth_angle,
        injection_polar_angle=injection_polar_angle,
        add_barriers=add_barriers,
    )
    if add_barriers:
        circuit.barrier(label="14-site abstract state injection")
    return circuit


def append_14_site_patch_abstract_routed_encoder(
    circuit: QuantumCircuit,
    *,
    patch_qubits: Sequence,
    injected_state: InjectedStateName | None = None,
    injection_azimuth_angle: float | None = None,
    injection_polar_angle: float | None = None,
    flag_bits: Sequence | None = None,
    add_barriers: bool = True,
) -> QuantumCircuit:
    """Append the routed arbitrary-state encoder found for IBM Miami.

    This is a hardware-local realization of the same isometry as
    :func:`append_14_site_patch_abstract_encoder`. It uses the fixed 14-site
    patch, prepares the arbitrary input on D3, and returns the encoded state on
    D0–D6 with every helper and D2' disentangled in ``|0>``. H8–H12 provide
    routing workspace; H7 and D2' are idle. Every CX is diagonal-local in the
    fixed rotated patch geometry::

              x=0    x=1    x=2    x=3    x=4    x=5    x=6
       y=0                   D6
       y=1                          H9            D2'
       y=2                   D4            H10           D2
       y=3           H7            D0            H12
       y=4                  H8            H11
       y=5           D1            D5            D3*

    ``*`` marks the arbitrary-input site. The explicit H/CX schedule and its
    IBM Miami native translation both have two-qubit depth 14 while involving
    only 12 patch sites. H8–H12 return to ``|0>`` and are optionally measured
    in Z as five routing flags.

    Parameters
    ----------
    injected_state, injection_azimuth_angle, injection_polar_angle
        Mutually exclusive injection inputs prepared on D3; see
        :func:`prepare_injection_input`.
    flag_bits
        Five classical bits for optional Z-basis measurements of H8–H12.
        ``None`` skips the flag measurements.
    """
    patch_qubits = resolve_patch_qubits(
        patch_qubits,
        expected_count=NUM_QUBITS_PATCH_14_SITES,
    )
    if flag_bits is not None and len(flag_bits) < len(
        PATCH_14_SITE_ABSTRACT_ROUTED_ENCODER_FLAG_SITES
    ):
        raise ValueError(
            "flag_bits must contain at least "
            f"{len(PATCH_14_SITE_ABSTRACT_ROUTED_ENCODER_FLAG_SITES)} bits."
        )

    prepare_injection_input(
        circuit,
        patch_qubits[PATCH_14_SITE_ABSTRACT_ROUTED_ENCODER_INJECTION_SITE],
        injected_state=injected_state,
        injection_azimuth_angle=injection_azimuth_angle,
        injection_polar_angle=injection_polar_angle,
    )

    for tick_layer in PATCH_14_SITE_ABSTRACT_ROUTED_ENCODER_LAYERS:
        for gate_name, operands in tick_layer:
            if gate_name == "H":
                for site in operands:
                    circuit.h(patch_qubits[site])
            elif gate_name == "CX":
                for control_site, target_site in operands:
                    circuit.cx(patch_qubits[control_site], patch_qubits[target_site])
            else:  # pragma: no cover - defensive if the shared schedule is edited.
                raise ValueError(f"Unsupported abstract-routed gate: {gate_name!r}.")
        if add_barriers:
            circuit.barrier(*patch_qubits)

    if flag_bits is not None:
        flag_bits = list(flag_bits)
        for index, site in enumerate(PATCH_14_SITE_ABSTRACT_ROUTED_ENCODER_FLAG_SITES):
            circuit.measure(patch_qubits[site], flag_bits[index])
    return circuit


def append_14_site_patch_hirano_encoder(
    circuit: QuantumCircuit,
    *,
    patch_qubits: Sequence,
    injected_state: InjectedStateName = "S",
    flag_bits: Sequence | None = None,
    add_barriers: bool = True,
) -> QuantumCircuit:
    """Append Qiskit's parallel of Hirano's 14-site encoder.

    The shared tick schedule is the specialized Hirano reference circuit. It
    retains Hirano's injection-stage coordinates, with D1 at ``(0, 4)`` next to
    H7, so every encoder CX is local in that geometry. A logical-Y eigenstate is
    injected by one phase operation on D4 at the schedule's injection tick. Use
    :func:`append_14_site_patch_abstract_encoder` for the direct fixed-geometry
    mapping or :func:`append_14_site_patch_abstract_routed_encoder` for its
    hardware-local arbitrary-state realization.

    Hirano's injection-stage layout is::

              x=0    x=1    x=2    x=3    x=4    x=5    x=6
       y=0                   D6
       y=1                          H9            D2'
       y=2                   D4            H10           D2
       y=3           H7            D0            H12
       y=4    D1            H8            H11
       y=5                          D5            D3

    Parameters
    ----------
    circuit
        Circuit being modified.
    patch_qubits
        Fourteen Qiskit qubits indexed by the patch labels.
    injected_state
        Logical-Y eigenstate to encode. ``"S"``/``"i"``/``"Y+"`` inject
        ``S``; ``"Sdg"``/``"-i"``/``"Y-"`` inject ``Sdg``. Every other state
        raises ``ValueError``. Use an abstract-routed 14-site encoder to encode an
        arbitrary state.
    flag_bits
        Six bits for optional X-basis helper measurements. ``None`` skips them.
    add_barriers
        Whether to keep barriers between Hirano schedule ticks.

    Returns
    -------
    QuantumCircuit
        The same circuit object, after appending the encoder.
    """
    patch_qubits = resolve_patch_qubits(
        patch_qubits,
        expected_count=NUM_QUBITS_PATCH_14_SITES,
    )
    if flag_bits is not None and len(flag_bits) < len(PATCH_14_SITE_HELPER_SITES):
        raise ValueError(f"flag_bits must contain at least {len(PATCH_14_SITE_HELPER_SITES)} bits.")

    if injected_state in ("S", "i", "Y+"):
        gate_on_injection_site = "S"
    elif injected_state in ("Sdg", "-i", "Y-"):
        gate_on_injection_site = "Sdg"
    else:
        raise ValueError(
            "The 14-site Hirano encoder cultivates a logical-Y eigenstate, so it "
            "can inject only the logical-Y eigenstate family: 'S'/'i'/'Y+' "
            "(logical Y=+1) or 'Sdg'/'-i'/'Y-' (logical Y=-1); got "
            f"{injected_state!r}."
        )

    for tick_layer in PATCH_14_SITE_HIRANO_ENCODER_LAYERS:
        for gate_name, operands in tick_layer:
            if gate_name in ("RESET_X", "RESET_Z"):
                for site in operands:
                    circuit.reset(patch_qubits[site])
                    if gate_name == "RESET_X":
                        circuit.h(patch_qubits[site])
            elif gate_name == "CX":
                for control_site, target_site in operands:
                    circuit.cx(patch_qubits[control_site], patch_qubits[target_site])
            elif gate_name == "INJECT":
                injection_qubit = patch_qubits[operands[0]]
                if gate_on_injection_site == "S":
                    circuit.s(injection_qubit)
                elif gate_on_injection_site == "Sdg":
                    circuit.sdg(injection_qubit)
                else:  # pragma: no cover - assigned from validated input above.
                    raise ValueError(f"Unsupported injection gate: {gate_on_injection_site!r}.")
            else:  # pragma: no cover - defensive if the shared schedule is edited.
                raise ValueError(f"Unsupported Hirano encoder gate: {gate_name!r}.")
        if add_barriers:
            circuit.barrier()

    if flag_bits is not None:
        flag_bits = list(flag_bits)
        for index, site in enumerate(PATCH_14_SITE_HELPER_SITES):
            circuit.h(patch_qubits[site])
            circuit.measure(patch_qubits[site], flag_bits[index])
    return circuit


# ----------------------------------------
# Stabilizer measurement
# ----------------------------------------


def append_14_site_patch_superdense_stabilizer_measurement(
    circuit: QuantumCircuit,
    *,
    patch_qubits: Sequence,
    stabilizer_bits: Sequence,
    bell_partner_flag_bits: Sequence | None = None,
    d2_copy_flag_bit=None,
    basis: PatchStabilizerBasis = "BOTH",
    stabilizer_indices: Sequence[int] | None = None,
    add_barriers: bool = True,
) -> None:
    """Append the shared collision-free superdense round on the 14-site patch.

    Selected faces retain the caller's ``stabilizer_indices`` order. In
    ``"BOTH"`` mode, each face's X result precedes its Z result, so the default
    six-readout order is ``[X0145, Z0145, X0235, Z0235, X0246, Z0246]``. D1
    remains at ``(1, 5)`` and couples to H8 in both directions.

    Single-basis checks always measure the other Bell ancilla in its native basis
    and record it in ``bell_partner_flag_bits``. Its ideal
    outcome is zero and is a fault flag, not another requested stabilizer
    result. Retaining this flag protects graphlike distance three.

    Selecting face 0246 always measures the restored D2' into
    ``d2_copy_flag_bit``: Z for a Z-only extraction and X otherwise.
    Its known-zero outcome flags the D2-copy detour and is not another
    stabilizer result. ``stabilizer_bits`` contains only the requested
    face-parity results; the two flag arguments contain only flag destinations.
    """
    patch_qubits = resolve_patch_qubits(
        patch_qubits,
        expected_count=NUM_QUBITS_PATCH_14_SITES,
    )
    measurement_layers, readouts = select_14_site_patch_superdense_measurement_layers(
        basis,
        stabilizer_indices,
    )
    required_stabilizer_bits = sum(readout_role == "STABILIZER" for _, _, readout_role in readouts)
    if len(stabilizer_bits) < required_stabilizer_bits:
        raise ValueError(
            "stabilizer_bits must contain at least " f"{required_stabilizer_bits} bits."
        )
    stabilizer_bits = list(stabilizer_bits)
    required_bell_partner_bits = sum(
        readout_role == "BELL_PARTNER_FLAG" for _, _, readout_role in readouts
    )
    if required_bell_partner_bits:
        if bell_partner_flag_bits is None:
            raise ValueError("bell_partner_flag_bits are required for a single-basis measurement.")
        bell_partner_flag_bits = list(bell_partner_flag_bits)
    elif bell_partner_flag_bits is not None:
        raise ValueError("bell_partner_flag_bits require a single-basis measurement.")
    else:
        bell_partner_flag_bits = []
    measure_d2_copy_flag = any(readout_role == "D2_COPY_FLAG" for _, _, readout_role in readouts)
    if measure_d2_copy_flag and d2_copy_flag_bit is None:
        raise ValueError("d2_copy_flag_bit is required for the D2-copy flag.")
    if not measure_d2_copy_flag and d2_copy_flag_bit is not None:
        raise ValueError("d2_copy_flag_bit requires a measurement of face 0246.")
    if len(bell_partner_flag_bits) < required_bell_partner_bits:
        raise ValueError(
            "bell_partner_flag_bits must contain at least " f"{required_bell_partner_bits} bits."
        )
    measurement_index = 0
    bell_partner_flag_index = 0

    for tick_layer in measurement_layers:
        for gate_name, operands in tick_layer:
            if gate_name in ("RESET_X", "RESET_Z"):
                for site in operands:
                    circuit.reset(patch_qubits[site])
                    if gate_name == "RESET_X":
                        circuit.h(patch_qubits[site])
            elif gate_name == "CX":
                for control_site, target_site in operands:
                    circuit.cx(patch_qubits[control_site], patch_qubits[target_site])
        if add_barriers:
            circuit.barrier()

    for readout_basis, site, readout_role in readouts:
        if readout_basis == "X":
            circuit.h(patch_qubits[site])
        if readout_role == "STABILIZER":
            bit = stabilizer_bits[measurement_index]
            measurement_index += 1
        elif readout_role == "BELL_PARTNER_FLAG":
            bit = bell_partner_flag_bits[bell_partner_flag_index]
            bell_partner_flag_index += 1
        elif readout_role == "D2_COPY_FLAG":
            bit = d2_copy_flag_bit
        else:  # pragma: no cover - roles are assigned by the shared selector.
            raise ValueError(f"Unsupported superdense readout role: {readout_role!r}.")
        circuit.measure(patch_qubits[site], bit)
    if add_barriers:
        circuit.barrier()


# ----------------------------------------
# Logical measurement
# ----------------------------------------


def append_14_site_patch_logical_measurement(
    circuit: QuantumCircuit,
    basis: str = "Y",
    *,
    patch_qubits: Sequence,
    support: tuple[int, ...] | None = None,
    ancilla_qubit=None,
    bits: Sequence,
    destructive: bool = True,
) -> None:
    """Append a logical Pauli measurement on the fixed 14-site patch."""
    patch_qubits = resolve_patch_qubits(
        patch_qubits,
        expected_count=NUM_QUBITS_PATCH_14_SITES,
    )
    append_logical_measurement_on_indexed_qubits(
        circuit,
        basis,
        indexed_qubits=patch_qubits,
        default_logical_x=PATCH_14_SITE_DEFAULT_LOGICAL_X,
        default_logical_z=PATCH_14_SITE_DEFAULT_LOGICAL_Z,
        support=support,
        ancilla_qubit=ancilla_qubit,
        bits=bits,
        destructive=destructive,
    )
