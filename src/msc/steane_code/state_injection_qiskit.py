from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from qiskit import QuantumCircuit
from qiskit.circuit import Qubit

from msc.steane_code.state_injection_common import (
    _STEANE_TO_11_SITE_PATCH,
    DEFAULT_LOGICAL_X,
    DEFAULT_LOGICAL_Z,
    NUM_QUBITS_PATCH_11_SITES,
    PATCH_11_SITE_DEFAULT_LOGICAL_X,
    PATCH_11_SITE_DEFAULT_LOGICAL_Z,
    PATCH_11_SITE_HELPER_SITES,
    PATCH_11_SITE_ROSENFELD_ENCODER_INJECTION_SITE,
    PATCH_11_SITE_ROSENFELD_ENCODER_LAYERS,
    PATCH_11_SITE_STABILIZER_ANCILLA_SITES,
    PATCH_11_SITE_X_STABILIZERS,
    PATCH_11_SITE_Z_STABILIZERS,
    STEANE_ENCODER_CNOTS,
    X_STABILIZERS,
    Z_STABILIZERS,
    PatchStabilizerBasis,
    resolve_11_site_patch_stabilizer_indices,
)
from msc.utils.circuit_qiskit import (
    append_logical_measurement_on_indexed_qubits,
    append_parity_measurement,
    apply_ty,
)
from msc.utils.normalize import resolve_patch_qubits

InjectedStateName = Literal["0", "1", "+", "-", "i", "-i", "Y+", "Y-", "S", "T", "Tdg", "Ty"]


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

    if state in {"-i", "Y-"}:
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


def append_abstract_encoder(
    circuit: QuantumCircuit,
    *,
    data_qubits: Sequence,
    injected_state: InjectedStateName | None = None,
    injection_azimuth_angle: float | None = None,
    add_barriers: bool = True,
) -> None:
    """Append a Steane-code state-injection encoder to an existing circuit.

    Implements the map
    |psi> |+> |+> |+> |0> |0> |0> -> |psi_L>
    on the seven supplied data qubits (ordered 0-6). The caller owns register
    creation and passes the target qubits via ``data_qubits``.

    Parameters
    ----------
    circuit
        Circuit being modified.
    data_qubits
        Seven qubits that will hold the encoded Steane-code state (ordered 0-6).
        The injected physical state is prepared on ``data_qubits[0]``.
    injected_state
        The unencoded single-qubit state to inject on qubit 0. The output is
        the corresponding encoded state in the Steane code. This may be
        ``None`` because it is mutually exclusive with
        ``injection_azimuth_angle``.
    injection_azimuth_angle
        When set, inject ``Rz(injection_azimuth_angle)|+⟩ = (|0⟩ + e^{i·angle}|1⟩)/√2``
        on qubit 0 instead of a named ``injected_state``.  This sweeps the
        equatorial (XY-plane) states relevant to T-state cultivation:
        ``angle=π/4`` reproduces ``|T⟩ = T|+⟩``, the +1 eigenstate of
        ``H_XY = (X + Y)/√2``. This may be ``None`` because it is mutually
        exclusive with ``injected_state``. At most one of these arguments may
        be non-``None``; when both are ``None``, ``injected_state="T"`` is used.
    add_barriers
        Whether to insert a barrier between preparation and encoding stages.
    """
    if len(data_qubits) < 7:
        raise ValueError("data_qubits must contain at least 7 qubits.")
    if injected_state is not None and injection_azimuth_angle is not None:
        raise ValueError(
            "injected_state and injection_azimuth_angle are mutually exclusive; supply at most one."
        )

    dq = list(data_qubits)

    if injection_azimuth_angle is not None:
        # Prepare Rz(angle)|+> on q0.
        circuit.h(dq[0])
        circuit.rz(injection_azimuth_angle, dq[0])
    else:
        # Prepare the unencoded state on q0.
        state: InjectedStateName = injected_state if injected_state is not None else "T"
        prepare_physical_state(circuit, dq[0], state)

    # Prepare the three Steane codeword-superposition variables u, v, w.
    circuit.h(dq[1])
    circuit.h(dq[4])
    circuit.h(dq[6])

    if add_barriers:
        circuit.barrier(dq)

    # CNOT encoder.
    for control_qubit, target_qubit in STEANE_ENCODER_CNOTS:
        circuit.cx(dq[control_qubit], dq[target_qubit])


def append_stabilizer_measurement(
    circuit: QuantumCircuit,
    *,
    data_qubits: Sequence,
    ancilla_qubits: Sequence,
    classical_bits: Sequence,
    add_barriers: bool = True,
) -> None:
    """Append non-destructive stabilizer measurements to an existing circuit.

    Measures all 6 Steane-code stabilizers (3 X-type, 3 Z-type) using
    dedicated ancilla qubits.

    Parameters
    ----------
    circuit
        Circuit being modified.
    data_qubits
        Seven qubits holding the encoded Steane-code state (ordered 0–6).
    ancilla_qubits
        Six ancilla qubits: indices 0–2 for X-stabilizers, 3–5 for Z-stabilizers.
    classical_bits
        Six classical bits: bits 0–2 hold X-stabilizer outcomes,
        bits 3–5 hold Z-stabilizer outcomes.
    add_barriers
        Whether to insert a barrier between X and Z stabilizer rounds.
    """
    if len(data_qubits) < 7:
        raise ValueError("data_qubits must contain at least 7 qubits.")
    if len(ancilla_qubits) < 6:
        raise ValueError("ancilla_qubits must contain at least 6 qubits.")
    if len(classical_bits) < 6:
        raise ValueError("classical_bits must contain at least 6 bits.")

    dq = list(data_qubits)
    aq = list(ancilla_qubits)
    cb = list(classical_bits)

    # X-stabilizer measurements.
    for i, support in enumerate(X_STABILIZERS):
        append_parity_measurement(
            circuit,
            "X",
            data_qubits=[dq[abstract_q] for abstract_q in support],
            ancilla_qubit=aq[i],
            classical_bit=cb[i],
        )

    if add_barriers:
        circuit.barrier()

    # Z-stabilizer measurements.
    for i, support in enumerate(Z_STABILIZERS):
        append_parity_measurement(
            circuit,
            "Z",
            data_qubits=[dq[abstract_q] for abstract_q in support],
            ancilla_qubit=aq[3 + i],
            classical_bit=cb[3 + i],
        )


def append_logical_measurement(
    circuit: QuantumCircuit,
    basis: str = "Z",
    *,
    data_qubits: Sequence,
    support: tuple[int, ...] | None = None,
    ancilla_qubit=None,
    classical_bits: Sequence,
    destructive: bool = True,
) -> None:
    """Append a logical measurement to an existing circuit.

    Parameters
    ----------
    circuit
        Circuit being modified.
    basis
        Which logical Pauli to measure: ``"X"``, ``"Y"``, or ``"Z"``.
    data_qubits
        Seven qubits holding the encoded Steane-code state (ordered 0–6).
    support
        Abstract qubit indices (0–6) for the logical representative. Defaults
        to ``DEFAULT_LOGICAL_Z`` for Z, or ``DEFAULT_LOGICAL_X`` for X and Y.
        For simplicity, a Y support must be a common X_L and Z_L representative.
    ancilla_qubit
        For non-destructive measurement, the ancilla qubit to use. Required
        when ``destructive=False``.
    classical_bits
        Classical bits for measurement results. For destructive: one per
        support qubit. For non-destructive: at least one bit.
    destructive
        If True (default), directly measure the data qubits in the
        appropriate basis. If False, use the ancilla for indirect measurement.
    """
    if len(data_qubits) < 7:
        raise ValueError("data_qubits must contain at least 7 qubits.")
    append_logical_measurement_on_indexed_qubits(
        circuit,
        basis,
        indexed_qubits=data_qubits,
        default_logical_x=DEFAULT_LOGICAL_X,
        default_logical_z=DEFAULT_LOGICAL_Z,
        support=support,
        ancilla_qubit=ancilla_qubit,
        classical_bits=classical_bits,
        destructive=destructive,
    )


# =============================================================================
# 11-site cultivation patch
# =============================================================================
#
# The patch geometry, abstract-to-patch qubit mapping, stabilizer/logical
# supports, Rosenfeld et al. encoder schedule, and stabilizer-selection validator
# are framework-neutral and imported from ``state_injection_common``.


def _append_11_site_patch_rosenfeld_encoder_layers(
    circuit: QuantumCircuit,
    *,
    patch_qubits: Sequence,
    add_barriers: bool,
) -> None:
    """Append the Rosenfeld et al. CZ/H encoder layers to the circuit.

    These layers do not include preparation of the arbitrary physical state on
    the injection site; that must be done separately before calling this helper.
    """
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


# ---------------------------------------------------------------------------
# Encoder on the 11-site patch
# ---------------------------------------------------------------------------


def append_11_site_patch_abstract_encoder(
    circuit: QuantumCircuit,
    *,
    patch_qubits: Sequence,
    injected_state: InjectedStateName | None = None,
    injection_azimuth_angle: float | None = None,
    add_barriers: bool = True,
) -> None:
    """Append the Steane-code state-injection encoder on the 11-site patch.

    Composes the 7-qubit encoder from append_abstract_encoder
    onto the 7 data sites of the 11-site patch via _STEANE_TO_11_SITE_PATCH.

    Parameters
    ----------
    circuit
        Circuit being modified.
    patch_qubits
        Eleven Qiskit qubits indexed by the 11-site patch labels.
    injected_state
        The physical one-qubit state to inject on site 0. This may be ``None``
        because it is mutually exclusive with ``injection_azimuth_angle``.
    injection_azimuth_angle
        When set, inject ``Rz(injection_azimuth_angle)|+⟩`` on the injection
        site instead of a named state. This may be ``None`` because it is
        mutually exclusive with ``injected_state``. At most one argument may
        be non-``None``; when both are ``None``, ``injected_state="T"`` is used.
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
    flag_classical_bits: Sequence | None = None,
    add_barriers: bool = True,
) -> QuantumCircuit:
    """Append the Rosenfeld et al. nearest-neighbor Steane encoder on the 11-site patch.

    This replays Rosenfeld et al.'s state-injection schedule after
    mapping every site into this repo's 11-site patch labels.  Unlike
    ``append_11_site_patch_abstract_encoder``, this schedule uses all four
    helper sites for nearest-neighbor routing.

    After the encoder layers the four helper sites should be in |0⟩. When
    ``flag_classical_bits`` is provided the helpers are measured into those
    bits so the caller can postselect on all-zero outcomes.

    Parameters
    ----------
    circuit
        Circuit being modified.
    patch_qubits
        Eleven Qiskit qubits indexed by the 11-site patch labels.
    injected_state
        The physical one-qubit state to inject on patch site 4, the center data
        site corresponding to Hirano label ``0``. This may be ``None`` because
        it is mutually exclusive with ``injection_azimuth_angle``.
    injection_azimuth_angle
        When set, inject ``Rz(injection_azimuth_angle)|+⟩ = (|0⟩ + e^{i·angle}|1⟩)/√2``
        on the injection site instead of a named ``injected_state``.  This
        sweeps the equatorial (XY-plane) states relevant to T-state cultivation:
        ``angle=π/4`` reproduces ``|T⟩ = T|+⟩``, the +1 eigenstate of
        ``H_XY = (X + Y)/√2``. Used for kickback-tomography angle scans. This
        may be ``None`` because it is mutually exclusive with
        ``injected_state``. At most one argument may be non-``None``; when both
        are ``None``, ``injected_state="T"`` is used.
    flag_classical_bits
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
    if injected_state is not None and injection_azimuth_angle is not None:
        raise ValueError(
            "injected_state and injection_azimuth_angle are mutually exclusive; supply at most one."
        )
    if flag_classical_bits is not None and len(flag_classical_bits) < len(
        PATCH_11_SITE_HELPER_SITES
    ):
        raise ValueError(
            f"flag_classical_bits must contain at least {len(PATCH_11_SITE_HELPER_SITES)} bits."
        )

    if injection_azimuth_angle is not None:
        circuit.h(patch_qubits[PATCH_11_SITE_ROSENFELD_ENCODER_INJECTION_SITE])
        circuit.rz(
            injection_azimuth_angle,
            patch_qubits[PATCH_11_SITE_ROSENFELD_ENCODER_INJECTION_SITE],
        )
    else:
        state: InjectedStateName = injected_state if injected_state is not None else "T"
        prepare_physical_state(
            circuit,
            patch_qubits[PATCH_11_SITE_ROSENFELD_ENCODER_INJECTION_SITE],
            state,
        )
    if add_barriers:
        circuit.barrier(label="rosenfeld injection input")

    _append_11_site_patch_rosenfeld_encoder_layers(
        circuit,
        patch_qubits=patch_qubits,
        add_barriers=add_barriers,
    )
    if add_barriers:
        circuit.barrier(label="rosenfeld state injection")

    if flag_classical_bits is not None:
        flag_classical_bits = list(flag_classical_bits)
        for i, site in enumerate(PATCH_11_SITE_HELPER_SITES):
            circuit.measure(patch_qubits[site], flag_classical_bits[i])

    return circuit


# ---------------------------------------------------------------------------
# Stabilizer measurement on the 11-site patch
# ---------------------------------------------------------------------------


def append_11_site_patch_stabilizer_measurement(
    circuit: QuantumCircuit,
    *,
    patch_qubits: Sequence,
    classical_bits: Sequence,
    basis: PatchStabilizerBasis = "BOTH",
    stabilizer_indices: Sequence[int] | None = None,
    add_barriers: bool = True,
) -> None:
    """Append non-destructive stabilizer measurements on the 11-site patch.

    Uses the patch helper sites (2, 6, 7) as ancillas to measure the selected
    X-type and/or Z-type stabilizers without disturbing the encoded state. The
    helper sites are reset before each measurement round.

    Parameters
    ----------
    circuit
        Circuit being modified.
    patch_qubits
        Eleven qubits indexed by the 11-site patch labels.
    classical_bits
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
    if basis not in {"X", "Z", "BOTH"}:
        raise ValueError("basis must be 'X', 'Z', or 'BOTH'.")

    selected_indices = resolve_11_site_patch_stabilizer_indices(stabilizer_indices)

    required_bits = len(selected_indices) * (2 if basis == "BOTH" else 1)
    if len(classical_bits) < required_bits:
        raise ValueError(f"classical_bits must contain at least {required_bits} bits.")

    cb = list(classical_bits)
    anc_sites = PATCH_11_SITE_STABILIZER_ANCILLA_SITES
    selected_ancilla_sites = tuple(anc_sites[index] for index in selected_indices)

    if basis in {"X", "BOTH"}:
        # X-stabilizer measurements.
        for site in selected_ancilla_sites:
            circuit.reset(patch_qubits[site])
        for output_index, stabilizer_index in enumerate(selected_indices):
            ancilla_site = anc_sites[stabilizer_index]
            support = PATCH_11_SITE_X_STABILIZERS[stabilizer_index]
            append_parity_measurement(
                circuit,
                "X",
                data_qubits=[patch_qubits[site] for site in support],
                ancilla_qubit=patch_qubits[ancilla_site],
                classical_bit=cb[output_index],
            )

    if basis == "BOTH" and add_barriers:
        circuit.barrier()

    if basis in {"Z", "BOTH"}:
        # Z-stabilizer measurements.
        for site in selected_ancilla_sites:
            circuit.reset(patch_qubits[site])
        classical_offset = len(selected_indices) if basis == "BOTH" else 0
        for output_index, stabilizer_index in enumerate(selected_indices):
            ancilla_site = anc_sites[stabilizer_index]
            support = PATCH_11_SITE_Z_STABILIZERS[stabilizer_index]
            append_parity_measurement(
                circuit,
                "Z",
                data_qubits=[patch_qubits[site] for site in support],
                ancilla_qubit=patch_qubits[ancilla_site],
                classical_bit=cb[classical_offset + output_index],
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
    classical_bits: Sequence,
    destructive: bool = True,
) -> None:
    """Append a logical measurement on the 11-site patch.

    Parameters
    ----------
    circuit
        Circuit being modified.
    basis
        Which logical Pauli to measure: ``"X"``, ``"Y"``, or ``"Z"``.
    patch_qubits
        Eleven qubits indexed by the 11-site patch labels.
    support
        Patch-site indices for the logical representative. Defaults to
        ``PATCH_11_SITE_DEFAULT_LOGICAL_Z`` for Z, or
        ``PATCH_11_SITE_DEFAULT_LOGICAL_X`` for X and Y.
        For simplicity, a Y support must be a common X_L and Z_L representative.
    ancilla_qubit
        For non-destructive measurement, the ancilla qubit to use. Required
        when ``destructive=False``.
    classical_bits
        Classical bits for measurement results. For destructive: one per
        support qubit. For non-destructive: at least one bit.
    destructive
        If True (default), directly measure the data qubits in the
        appropriate basis. If False, use the ancilla for indirect measurement.
    """
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
        classical_bits=classical_bits,
        destructive=destructive,
    )
