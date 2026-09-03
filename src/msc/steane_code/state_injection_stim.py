"""Clifford Stim counterparts of the Steane state-injection helpers.

Stim can encode any supported Clifford input through the abstract 7-qubit,
11-site, and 14-site encoders. The non-Clifford ``T`` injection used on hardware
is represented by its Clifford proxy ``S|+> = |i>``. Stim analysis reports the
native ``S``-state logical error rate without converting it to a ``T``-state
estimate.

The public helpers intentionally have the same names as their Qiskit
counterparts. The module name identifies the framework at import time. Shared
encoder schedules, patch labels, logical supports, and validation helpers are
imported from :mod:`state_injection_common` so the two implementations have one
source of truth.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import stim

from msc.steane_code.state_injection_common import (
    _STEANE_TO_11_SITE_PATCH,
    _STEANE_TO_14_SITE_PATCH,
    ABSTRACT_ENCODER_CNOT_LAYERS,
    NUM_QUBITS_PATCH_11_SITES,
    NUM_QUBITS_PATCH_14_SITES,
    PATCH_11_SITE_COORDS,
    PATCH_11_SITE_DEFAULT_LOGICAL_X,
    PATCH_11_SITE_DEFAULT_LOGICAL_Z,
    PATCH_11_SITE_HELPER_SITES,
    PATCH_11_SITE_ROSENFELD_ENCODER_INJECTION_SITE,
    PATCH_11_SITE_ROSENFELD_ENCODER_LAYERS,
    PATCH_11_SITE_STABILIZER_ANCILLA_SITES,
    PATCH_11_SITE_X_STABILIZERS,
    PATCH_11_SITE_Z_STABILIZERS,
    PATCH_14_SITE_ABSTRACT_ROUTED_ENCODER_FLAG_SITES,
    PATCH_14_SITE_ABSTRACT_ROUTED_ENCODER_INJECTION_SITE,
    PATCH_14_SITE_ABSTRACT_ROUTED_ENCODER_LAYERS,
    PATCH_14_SITE_COORDS,
    PATCH_14_SITE_DEFAULT_LOGICAL_X,
    PATCH_14_SITE_DEFAULT_LOGICAL_Z,
    PATCH_14_SITE_HELPER_SITES,
    PATCH_14_SITE_HIRANO_ENCODER_COORDS,
    PATCH_14_SITE_HIRANO_ENCODER_LAYERS,
    PatchStabilizerBasis,
    resolve_patch_stabilizer_indices,
    select_14_site_patch_superdense_measurement_layers,
)
from msc.utils import circuit_stim
from msc.utils.circuit_stim import (
    append_logical_measurement_on_indexed_qubits,
    append_parity_measurement,
)
from msc.utils.normalize import resolve_patch_qubits

# Clifford single-qubit states injectable in Stim.  This is the Clifford subset
# of the Qiskit ``InjectedStateName``; the non-Clifford ``"T"``/``"Tdg"``/``"Ty"``
# magic states cannot be simulated by Stim.  ``"S"`` is the Clifford proxy used
# in place of ``"T"``.
StimInjectedStateName = Literal["0", "1", "+", "-", "i", "-i", "Y+", "Y-", "S", "Sdg"]

DEFAULT_STIM_INJECTED_STATE: StimInjectedStateName = "S"
DEFAULT_STIM_11_SITE_PATCH_QUBITS: tuple[int, ...] = tuple(range(NUM_QUBITS_PATCH_11_SITES))
DEFAULT_STIM_14_SITE_PATCH_QUBITS: tuple[int, ...] = tuple(range(NUM_QUBITS_PATCH_14_SITES))


def prepare_physical_state(
    circuit: stim.Circuit,
    qubit: int,
    state: StimInjectedStateName,
) -> None:
    """Prepare a Clifford single-qubit ``state`` on ``qubit`` (assumed ``|0>``).

    Mirrors ``prepare_physical_state`` restricted to Clifford states. ``"S"``
    (equivalently ``"i"``/``"Y+"``) prepares ``S|+> = |i>``, the Clifford proxy
    for the ``T`` magic state.
    """
    if state == "0":
        return
    if state == "1":
        circuit.append("X", [qubit])
        return
    if state == "+":
        circuit.append("H", [qubit])
        return
    if state == "-":
        circuit.append("X", [qubit])
        circuit.append("H", [qubit])
        return
    if state in ("i", "Y+", "S"):
        # |i> = S|+> = +1 eigenstate of Pauli Y (Clifford proxy for |T>).
        circuit.append("H", [qubit])
        circuit.append("S", [qubit])
        return
    if state in ("-i", "Y-", "Sdg"):
        # |-i> = S_dag|+> = -1 eigenstate of Pauli Y.
        circuit.append("H", [qubit])
        circuit.append("S_DAG", [qubit])
        return
    raise ValueError(f"Unknown Clifford state name: {state!r}")


def append_abstract_encoder(
    circuit: stim.Circuit,
    *,
    data_qubits: Sequence[int],
    injected_state: StimInjectedStateName = DEFAULT_STIM_INJECTED_STATE,
    add_ticks: bool = True,
) -> stim.Circuit:
    """Append the abstract Steane encoder on seven caller-owned data qubits.

    This is the Clifford parallel of Qiskit's :func:`append_abstract_encoder`.
    It implements
    ``|psi> |+> |0> |0> |+> |0> |+> -> |psi_L>`` for any single-qubit
    Clifford state supported by Stim and inserts explicit tick boundaries for
    the seven collision-free CX layers.
    """
    if len(data_qubits) < 7:
        raise ValueError("data_qubits must contain at least 7 qubits.")
    data_qubits = list(data_qubits)

    prepare_physical_state(circuit, data_qubits[0], injected_state)
    circuit.append("H", [data_qubits[site] for site in (1, 4, 6)])
    if add_ticks:
        circuit.append("TICK", [])

    for layer in ABSTRACT_ENCODER_CNOT_LAYERS:
        targets: list[int] = []
        for control, target in layer:
            targets.extend((data_qubits[control], data_qubits[target]))
        circuit.append("CX", targets)
        if add_ticks:
            circuit.append("TICK", [])
    return circuit


# =============================================================================
# 11-site patch (abstract and Rosenfeld encoders)
# =============================================================================


def append_11_site_patch_abstract_encoder(
    circuit: stim.Circuit,
    *,
    patch_qubits: Sequence[int] = DEFAULT_STIM_11_SITE_PATCH_QUBITS,
    injected_state: StimInjectedStateName = DEFAULT_STIM_INJECTED_STATE,
    add_ticks: bool = True,
) -> stim.Circuit:
    """Append the abstract Steane encoder on the 11-site patch data sites.

    This is the Clifford parallel of Qiskit's
    :func:`append_11_site_patch_abstract_encoder`. Abstract qubits 0–6 map to
    patch sites ``(0, 3, 5, 4, 8, 9, 10)``; the four helper sites remain idle.
    """
    patch_qubits = resolve_patch_qubits(
        patch_qubits,
        expected_count=NUM_QUBITS_PATCH_11_SITES,
    )
    for qubit, coords in zip(patch_qubits, PATCH_11_SITE_COORDS, strict=True):
        circuit.append("QUBIT_COORDS", [qubit], coords)
    return append_abstract_encoder(
        circuit,
        data_qubits=[patch_qubits[site] for site in _STEANE_TO_11_SITE_PATCH],
        injected_state=injected_state,
        add_ticks=add_ticks,
    )


def append_11_site_patch_rosenfeld_encoder(
    circuit: stim.Circuit,
    *,
    patch_qubits: Sequence[int] = DEFAULT_STIM_11_SITE_PATCH_QUBITS,
    injected_state: StimInjectedStateName = DEFAULT_STIM_INJECTED_STATE,
    measure_flags: bool = True,
    add_ticks: bool = True,
) -> stim.Circuit:
    """Append the Clifford (Stim) Rosenfeld et al. Steane injection encoder.

    Mirrors the Qiskit ``append_11_site_patch_rosenfeld_encoder``.
    The injection site is prepared in ``injected_state`` and the shared
    Rosenfeld et al. ``H``/``CZ`` layers then entangle the 11-site patch into
    the encoded state. Measured helper flags are annotated as known-zero
    detectors.

    Parameters
    ----------
    circuit
        Stim circuit being appended to.
    patch_qubits
        Eleven Stim qubit indices ordered by the 11-site patch labels. Defaults
        to ``range(11)``.
    injected_state
        Clifford state to inject on patch site 4 (the center data site).
        Defaults to ``"S"``, the proxy for the ``T`` magic state.
    measure_flags
        Whether to measure helper sites 1, 2, 6, and 7 into the Stim
        measurement record after encoding, as known-zero postselection flags.
        Defaults to ``True`` since a cultivation run always postselects on them.
    add_ticks
        Whether to insert ``TICK`` layer separators (mirrors the Qiskit
        ``add_barriers`` argument).

    Returns
    -------
    stim.Circuit
        The same circuit object, after appending the encoder.
    """
    patch_qubits = resolve_patch_qubits(
        patch_qubits,
        expected_count=NUM_QUBITS_PATCH_11_SITES,
    )
    for qubit, coords in zip(patch_qubits, PATCH_11_SITE_COORDS, strict=True):
        circuit.append("QUBIT_COORDS", [qubit], coords)

    prepare_physical_state(
        circuit,
        patch_qubits[PATCH_11_SITE_ROSENFELD_ENCODER_INJECTION_SITE],
        injected_state,
    )
    if add_ticks:
        circuit.append("TICK", [])

    for gate_name, operands in PATCH_11_SITE_ROSENFELD_ENCODER_LAYERS:
        if gate_name == "H":
            circuit.append("H", [patch_qubits[site] for site in operands])
        elif gate_name == "CZ":
            targets: list[int] = []
            for site_a, site_b in operands:
                targets.extend((patch_qubits[site_a], patch_qubits[site_b]))
            circuit.append("CZ", targets)
        else:  # pragma: no cover - defensive if the shared schedule is edited.
            raise ValueError(f"Unsupported Rosenfeld encoder gate: {gate_name!r}")

        if add_ticks:
            circuit.append("TICK", [])

    if measure_flags:
        first_flag_position = circuit.num_measurements
        circuit.append(
            "M",
            [patch_qubits[site] for site in PATCH_11_SITE_HELPER_SITES],
        )
        for position in range(first_flag_position, circuit.num_measurements):
            circuit_stim.append_detector(
                circuit,
                measurement_positions=(position,),
            )

    return circuit


def append_11_site_patch_stabilizer_measurement(
    circuit: stim.Circuit,
    *,
    patch_qubits: Sequence[int] = DEFAULT_STIM_11_SITE_PATCH_QUBITS,
    basis: PatchStabilizerBasis = "BOTH",
    stabilizer_indices: Sequence[int] | None = None,
    add_ticks: bool = True,
) -> None:
    """Append non-destructive Stim stabilizer measurements on the 11-site patch.

    Mirrors the Qiskit ``append_11_site_patch_stabilizer_measurement``: the helper sites (2, 6, 7)
    act as ancillas measuring the X- and/or Z-type stabilizers. Measurements are
    appended to the record in order X0, X1, X2 then Z0, Z1, Z2 (three per basis).
    Every measured code stabilizer is annotated against its known +1 reference.

    Parameters
    ----------
    circuit
        Stim circuit being appended to.
    patch_qubits
        Eleven Stim qubit indices ordered by the 11-site patch labels. Defaults
        to ``range(11)``.
    basis
        Stabilizer basis to measure (``"X"``, ``"Z"``, or ``"BOTH"``).
    stabilizer_indices
        Ordered subset of the three canonical stabilizers to measure. ``None``
        measures all three.
    add_ticks
        Whether to insert a ``TICK`` between X and Z rounds in ``"BOTH"``
        mode.
    """
    if basis not in ("X", "Z", "BOTH"):
        raise ValueError("basis must be 'X', 'Z', or 'BOTH'.")

    patch_qubits = resolve_patch_qubits(
        patch_qubits,
        expected_count=NUM_QUBITS_PATCH_11_SITES,
    )
    selected_indices = resolve_patch_stabilizer_indices(stabilizer_indices)
    first_stabilizer_position = circuit.num_measurements
    anc = PATCH_11_SITE_STABILIZER_ANCILLA_SITES

    if basis in ("X", "BOTH"):
        for stabilizer_index in selected_indices:
            ancilla_site = anc[stabilizer_index]
            support = PATCH_11_SITE_X_STABILIZERS[stabilizer_index]
            append_parity_measurement(
                circuit,
                "X",
                data_qubits=[patch_qubits[site] for site in support],
                ancilla_qubit=patch_qubits[ancilla_site],
            )
        if basis == "BOTH" and add_ticks:
            circuit.append("TICK", [])

    if basis in ("Z", "BOTH"):
        for stabilizer_index in selected_indices:
            ancilla_site = anc[stabilizer_index]
            support = PATCH_11_SITE_Z_STABILIZERS[stabilizer_index]
            append_parity_measurement(
                circuit,
                "Z",
                data_qubits=[patch_qubits[site] for site in support],
                ancilla_qubit=patch_qubits[ancilla_site],
            )
    for position in range(first_stabilizer_position, circuit.num_measurements):
        circuit_stim.append_detector(
            circuit,
            measurement_positions=(position,),
        )


def append_11_site_patch_logical_measurement(
    circuit: stim.Circuit,
    basis: str = "Z",
    *,
    patch_qubits: Sequence[int] = DEFAULT_STIM_11_SITE_PATCH_QUBITS,
    support: tuple[int, ...] | None = None,
    ancilla_qubit: int | None = None,
    destructive: bool = True,
    observable_index: int | None = None,
) -> None:
    """Append a logical Pauli measurement on the 11-site patch.

    Measurement results are appended to Stim's measurement record. A
    destructive measurement appends one result per support qubit; a
    non-destructive measurement appends one parity result from ``ancilla_qubit``.
    As in the Qiskit implementation, a corrected weight-three logical-Y result
    uses ``0`` for the logical ``+1`` eigenvalue.
    Set ``observable_index`` to append this measurement parity to a Stim
    logical observable; omit it when the result is protocol data or a Pauli-
    frame input rather than a terminal observable.
    """
    patch_qubits = resolve_patch_qubits(
        patch_qubits,
        expected_count=NUM_QUBITS_PATCH_11_SITES,
    )
    first_measurement_position = circuit.num_measurements
    append_logical_measurement_on_indexed_qubits(
        circuit,
        basis,
        indexed_qubits=patch_qubits,
        default_logical_x=PATCH_11_SITE_DEFAULT_LOGICAL_X,
        default_logical_z=PATCH_11_SITE_DEFAULT_LOGICAL_Z,
        support=support,
        ancilla_qubit=ancilla_qubit,
        destructive=destructive,
    )
    if observable_index is not None:
        circuit_stim.append_observable_include(
            circuit,
            measurement_positions=tuple(
                range(first_measurement_position, circuit.num_measurements)
            ),
            observable_index=observable_index,
        )


# =============================================================================
# 14-site patch (abstract, abstract-routed, and Hirano encoders)
# =============================================================================


def append_14_site_patch_abstract_encoder(
    circuit: stim.Circuit,
    *,
    patch_qubits: Sequence[int] = DEFAULT_STIM_14_SITE_PATCH_QUBITS,
    injected_state: StimInjectedStateName = DEFAULT_STIM_INJECTED_STATE,
    add_ticks: bool = True,
) -> stim.Circuit:
    """Append the abstract Steane encoder directly on the 14-site data sites.

    This is the fixed-geometry, unrouted reference encoder. Unlike Hirano's
    specialized phase-injection schedule, the abstract Steane isometry is not
    restricted to logical-Y eigenstates: it encodes any Clifford single-qubit
    input supported by Stim. It maps the abstract Steane triangle onto Hirano
    data labels ``(6, 4, 2, 0, 1, 5, 3)`` and never uses the six helper sites or
    D2'. In particular, D1 is data site 1 at ``(1, 5)`` from the start; no
    post-injection relocation is needed::

              x=0    x=1    x=2    x=3    x=4    x=5    x=6
       y=0                   D6/A0
       y=1                          H9            D2'
       y=2                   D4/A1         H10           D2/A2
       y=3           H7            D0/A3          H12
       y=4                  H8            H11
       y=5           D1/A4         D5/A5          D3/A6

    ``A<i>`` denotes abstract encoder qubit ``i``. The injected physical state
    is therefore prepared on D6/A0; abstract qubits A1, A4, and A6 are prepared
    in ``|+>`` before the seven-layer CX encoder. The abstract CXs are not routed
    through the patch connectivity.
    """
    patch_qubits = resolve_patch_qubits(
        patch_qubits,
        expected_count=NUM_QUBITS_PATCH_14_SITES,
    )
    for qubit, coords in zip(patch_qubits, PATCH_14_SITE_COORDS, strict=True):
        circuit.append("QUBIT_COORDS", [qubit], coords)
    return append_abstract_encoder(
        circuit,
        data_qubits=[patch_qubits[site] for site in _STEANE_TO_14_SITE_PATCH],
        injected_state=injected_state,
        add_ticks=add_ticks,
    )


def append_14_site_patch_abstract_routed_encoder(
    circuit: stim.Circuit,
    *,
    patch_qubits: Sequence[int] = DEFAULT_STIM_14_SITE_PATCH_QUBITS,
    injected_state: StimInjectedStateName = DEFAULT_STIM_INJECTED_STATE,
    measure_flags: bool = True,
    add_ticks: bool = True,
) -> stim.Circuit:
    """Append the Clifford parallel of the routed IBM Miami encoder.

    This is the same arbitrary-input Steane isometry as
    :func:`append_14_site_patch_abstract_encoder`, restricted only by Stim to
    Clifford input states. It starts the input on D3 and returns the encoded
    state to D0–D6, while every helper and D2' finishes disentangled in
    ``|0>``. H8–H12 provide routing workspace; H7 and D2' are idle::

              x=0    x=1    x=2    x=3    x=4    x=5    x=6
       y=0                   D6
       y=1                          H9            D2'
       y=2                   D4            H10           D2
       y=3           H7            D0            H12
       y=4                  H8            H11
       y=5           D1            D5            D3*

    ``*`` marks the arbitrary-input site. Every CX is diagonal-local in the
    fixed rotated patch geometry, and the two-qubit depth is 14. H8–H12 return
    to ``|0>`` and can be measured in Z as known-zero routing flags.

    Parameters
    ----------
    measure_flags
        Whether to measure H8–H12 in Z and append a known-zero detector for
        each routing flag.
    """
    patch_qubits = resolve_patch_qubits(
        patch_qubits,
        expected_count=NUM_QUBITS_PATCH_14_SITES,
    )
    for qubit, coords in zip(patch_qubits, PATCH_14_SITE_COORDS, strict=True):
        circuit.append("QUBIT_COORDS", [qubit], coords)
    prepare_physical_state(
        circuit,
        patch_qubits[PATCH_14_SITE_ABSTRACT_ROUTED_ENCODER_INJECTION_SITE],
        injected_state,
    )
    for tick_layer in PATCH_14_SITE_ABSTRACT_ROUTED_ENCODER_LAYERS:
        for gate_name, operands in tick_layer:
            if gate_name == "H":
                circuit.append("H", [patch_qubits[site] for site in operands])
            elif gate_name == "CX":
                targets: list[int] = []
                for control_site, target_site in operands:
                    targets.extend((patch_qubits[control_site], patch_qubits[target_site]))
                circuit.append("CX", targets)
            else:  # pragma: no cover - defensive if the shared schedule is edited.
                raise ValueError(f"Unsupported abstract-routed gate: {gate_name!r}.")
        if add_ticks:
            circuit.append("TICK", [])

    if measure_flags:
        first_flag_position = circuit.num_measurements
        circuit.append(
            "M",
            [patch_qubits[site] for site in PATCH_14_SITE_ABSTRACT_ROUTED_ENCODER_FLAG_SITES],
        )
        for position in range(first_flag_position, circuit.num_measurements):
            circuit_stim.append_detector(
                circuit,
                measurement_positions=(position,),
            )
    return circuit


def append_14_site_patch_hirano_encoder(
    circuit: stim.Circuit,
    *,
    patch_qubits: Sequence[int] = DEFAULT_STIM_14_SITE_PATCH_QUBITS,
    injected_state: StimInjectedStateName = DEFAULT_STIM_INJECTED_STATE,
    measure_flags: bool = True,
    add_ticks: bool = True,
) -> stim.Circuit:
    """Append the Clifford (Stim) Hirano et al. 14-site Steane injection encoder.

    Ports ``injection_generator`` from Hirano's lattice-surgery cultivation
    (arXiv:2510.24615) onto the rotated 14-site patch.  The encoder entangles
    the 13 core sites (0–12); D2' (site 13) is idle here and is prepared fresh
    by the superdense round. It retains Hirano's injection-stage coordinates,
    with D1 at ``(0, 4)`` next to H7; every encoder CX is therefore local in
    this specialized geometry. Use
    :func:`append_14_site_patch_abstract_encoder` for the direct fixed-geometry
    reference or :func:`append_14_site_patch_abstract_routed_encoder` for
    its hardware-local arbitrary-state realization. The magic state is
    injected mid-encoding by a single phase gate (``S`` or ``S_DAG``) on data
    site 4 (Steane qubit 4).
    Because that lone phase gate is the only tunable
    operation, this circuit can encode **only a logical-Y eigenstate** — it
    cannot prepare ``|0>``/``|1>``/``|+>``/``|->`` or any other non-Y state.  The
    seven data sites finish in a Steane logical-Y eigenstate and the six helper
    sites return disentangled to |+>.

    The 14-site layout (data sites ``D<i>`` hold Steane qubit ``i``, helper
    sites ``H<j>``; ``D4`` is the injection site; ``D2'`` is site 13; local
    coordinates with ``y`` increasing downward, ``x`` 0–6, ``y`` 0–5)::

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
        Stim circuit being appended to.
    patch_qubits
        Fourteen Stim qubit indices ordered by the 14-site patch labels.
        Defaults to ``range(14)``.
    injected_state
        Single-qubit state to inject mid-encoding.  **Only the logical-Y
        eigenstate family is accepted**, because the encoder cultivates a
        logical-Y eigenstate via one phase gate at the injection tick:
        ``"S"``/``"i"``/``"Y+"`` inject ``S`` (logical ``Y = +1``, the default)
        and ``"Sdg"``/``"-i"``/``"Y-"`` inject ``S_DAG`` (logical ``Y = -1``).
        Non-Y Clifford states — ``"0"``/``"1"``/``"+"``/``"-"`` and any other
        name — cannot be encoded by this circuit and raise ``ValueError``.
        Treating the injected ``S`` as the ``T`` magic-state proxy is the
        caller's choice.
    measure_flags
        Whether to measure the six helper sites (which return disentangled to
        |+>) in the X basis after encoding, as encoding flags. Like the Rosenfeld
        11-site encoder's four |0>-returning Z-basis flags, each is annotated as
        a known-zero detector so a clean encoding produces no detection events.
    add_ticks
        Whether to insert ``TICK`` layer separators between encoder ticks.

    Returns
    -------
    stim.Circuit
        The same circuit object, after appending patch coordinates and encoder
        layers.
    """
    patch_qubits = resolve_patch_qubits(
        patch_qubits,
        expected_count=NUM_QUBITS_PATCH_14_SITES,
    )
    if injected_state in ("S", "i", "Y+"):
        gate_on_injection_site = "S"
    elif injected_state in ("Sdg", "-i", "Y-"):
        gate_on_injection_site = "S_DAG"
    else:
        raise ValueError(
            "The 14-site Hirano encoder cultivates a logical-Y eigenstate, so it "
            "can inject only the logical-Y eigenstate family: 'S'/'i'/'Y+' "
            "(logical Y=+1) or 'Sdg'/'-i'/'Y-' (logical Y=-1). Non-Y Clifford "
            f"states such as '0'/'1'/'+'/'-' cannot be injected; got "
            f"{injected_state!r}."
        )

    for qubit, coords in zip(
        patch_qubits,
        PATCH_14_SITE_HIRANO_ENCODER_COORDS,
        strict=True,
    ):
        circuit.append("QUBIT_COORDS", [qubit], coords)
    for tick_layer in PATCH_14_SITE_HIRANO_ENCODER_LAYERS:
        for gate_name, operands in tick_layer:
            if gate_name in ("RESET_X", "RESET_Z"):
                stim_reset = "RX" if gate_name == "RESET_X" else "RZ"
                circuit.append(stim_reset, [patch_qubits[site] for site in operands])
            elif gate_name == "CX":
                targets: list[int] = []
                for control_site, target_site in operands:
                    targets.extend((patch_qubits[control_site], patch_qubits[target_site]))
                circuit.append("CX", targets)
            elif gate_name == "INJECT":
                circuit.append(
                    gate_on_injection_site,
                    [patch_qubits[operands[0]]],
                )
            else:  # pragma: no cover - defensive if the shared schedule is edited.
                raise ValueError(f"Unsupported Hirano encoder gate: {gate_name!r}.")
        if add_ticks:
            circuit.append("TICK", [])

    if measure_flags:
        # The six helper sites return disentangled to |+>, so an X-basis readout
        # is a known-zero flag; a fault reaching a helper flips its flag.
        first_flag_position = circuit.num_measurements
        circuit.append(
            "MX",
            [patch_qubits[site] for site in PATCH_14_SITE_HELPER_SITES],
        )
        for position in range(first_flag_position, circuit.num_measurements):
            circuit_stim.append_detector(
                circuit,
                measurement_positions=(position,),
            )

    return circuit


def append_14_site_patch_superdense_stabilizer_measurement(
    circuit: stim.Circuit,
    *,
    patch_qubits: Sequence[int] = DEFAULT_STIM_14_SITE_PATCH_QUBITS,
    basis: PatchStabilizerBasis = "BOTH",
    stabilizer_indices: Sequence[int] | None = None,
    add_ticks: bool = True,
) -> None:
    """Append selected 14-site Steane checks using superdense Bell ancillas.

    Each face uses one Bell pair to extract its X and Z parity simultaneously.
    The default record order is
    ``[X0145, Z0145, X0235, Z0235, X0246, Z0246]``. Each result receives a
    detector against its known ``+1`` value.

    A single-basis extraction omits the opposite-basis data couplings but still
    reads the partner ancilla in its native basis. This known-zero result is a
    Bell-correlation fault flag, not the opposite-basis stabilizer parity.

    Face 0246 temporarily encodes D2 into D2' (site 13) to reach its distant
    ancilla. After the final uncompute, D2' should be ``|0>`` for a Z-only
    extraction or ``|+>`` otherwise, so its measurement is a restoration flag.
    The ``Z0246`` or ``X0246`` parity is reported separately by the Bell
    ancilla. If both bases are extracted, D2' is reset directly from the Z
    repetition encoding into the X encoding, so only the final X flag is read.

    Parameters
    ----------
    circuit
        Stim circuit being appended to.
    patch_qubits
        Fourteen Stim qubit indices ordered by the 14-site patch labels.
    basis
        Stabilizer basis to measure (``"X"``, ``"Z"``, or ``"BOTH"``).
    stabilizer_indices
        Ordered subset of the three canonical stabilizers to measure. ``None``
        measures all three.
    add_ticks
        Whether to insert ``TICK`` layer separators between round layers.
    """
    patch_qubits = resolve_patch_qubits(
        patch_qubits,
        expected_count=NUM_QUBITS_PATCH_14_SITES,
    )

    measurement_layers, readouts = select_14_site_patch_superdense_measurement_layers(
        basis,
        stabilizer_indices,
    )
    for tick_layer in measurement_layers:
        for gate_name, operands in tick_layer:
            if gate_name in ("RESET_X", "RESET_Z"):
                stim_reset = "RX" if gate_name == "RESET_X" else "RZ"
                circuit.append(stim_reset, [patch_qubits[site] for site in operands])
            elif gate_name == "CX":
                targets: list[int] = []
                for control_site, target_site in operands:
                    targets.extend((patch_qubits[control_site], patch_qubits[target_site]))
                circuit.append("CX", targets)
        if add_ticks:
            circuit.append("TICK", [])

    # Stabilizers and both flag roles are known-zero detectors in this Stim path.
    for readout_basis, site, _readout_role in readouts:
        stim_gate = "MX" if readout_basis == "X" else "M"
        circuit.append(stim_gate, [patch_qubits[site]])
        circuit_stim.append_detector(
            circuit,
            measurement_positions=(circuit.num_measurements - 1,),
        )
    if add_ticks:
        circuit.append("TICK", [])


def append_14_site_patch_logical_measurement(
    circuit: stim.Circuit,
    basis: str = "Y",
    *,
    patch_qubits: Sequence[int] = DEFAULT_STIM_14_SITE_PATCH_QUBITS,
    support: tuple[int, ...] | None = None,
    ancilla_qubit: int | None = None,
    destructive: bool = True,
    observable_index: int | None = None,
) -> None:
    """Append a logical Pauli measurement on the 14-site patch.

    Mirrors ``append_11_site_patch_logical_measurement``. The helper supports
    X, Y, or Z readout; its Y default matches the current 14-site cultivation
    protocol, not a restriction of the mapped abstract encoder. Set
    ``observable_index`` to append this measurement parity to a Stim logical
    observable; omit it when the result is protocol data or a Pauli-frame input
    rather than a terminal observable.
    """
    patch_qubits = resolve_patch_qubits(
        patch_qubits,
        expected_count=NUM_QUBITS_PATCH_14_SITES,
    )
    first_measurement_position = circuit.num_measurements
    append_logical_measurement_on_indexed_qubits(
        circuit,
        basis,
        indexed_qubits=patch_qubits,
        default_logical_x=PATCH_14_SITE_DEFAULT_LOGICAL_X,
        default_logical_z=PATCH_14_SITE_DEFAULT_LOGICAL_Z,
        support=support,
        ancilla_qubit=ancilla_qubit,
        destructive=destructive,
    )
    if observable_index is not None:
        circuit_stim.append_observable_include(
            circuit,
            measurement_positions=tuple(
                range(first_measurement_position, circuit.num_measurements)
            ),
            observable_index=observable_index,
        )
