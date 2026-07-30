"""Circuit-level utility helpers (gate decompositions, depth metrics, device access)."""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import Any, TypeVar

from qiskit import QuantumCircuit
from qiskit.circuit.classical import expr
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler import TransformationPass

from magic_state_cultivation.utils.normalize import normalize_pauli_char

_SupportT = TypeVar("_SupportT", bound=Hashable)

# ---------------------------------------------------------------------------
# Circuit transformations
# ---------------------------------------------------------------------------


class RemoveResetOperations(TransformationPass):
    """Remove every reset operation from a completed circuit."""

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        dag.remove_all_ops_named("reset")
        return dag


# ---------------------------------------------------------------------------
# Gate decompositions
# ---------------------------------------------------------------------------


def apply_ty(circuit: QuantumCircuit, qubit) -> None:
    """Apply ``T_y = RY(pi/4) = exp(-i*pi*Y/8)``.

    This rotation prepares the retained ``|Ty>`` encoder input.

    The decomposition uses only named Clifford+T gates (no parameterized
    ``rz`` or ``ry``), making it compatible with Aer's ``extended_stabilizer``
    simulation method.
    """
    # RY(pi/4) = Sdg · H · T · H · S  (up to global phase)
    circuit.sdg(qubit)
    circuit.h(qubit)
    circuit.t(qubit)
    circuit.h(qubit)
    circuit.s(qubit)


def apply_ty_dagger(circuit: QuantumCircuit, qubit) -> None:
    """Apply ``T_y^dagger = RY(-pi/4)``."""
    # RY(-pi/4) = Sdg · H · Tdg · H · S  (up to global phase)
    circuit.sdg(qubit)
    circuit.h(qubit)
    circuit.tdg(qubit)
    circuit.h(qubit)
    circuit.s(qubit)


# ---------------------------------------------------------------------------
# Parity measurements
# ---------------------------------------------------------------------------


def logical_y_parity_is_inverted(x_z_overlap: Sequence[object]) -> bool:
    """Return whether measuring the physical Y string reverses logical Y.

    Let ``S_X`` and ``S_Z`` be the supports of X-only and Z-only logical
    representatives for the same encoded qubit, and let
    ``m = len(S_X & S_Z)``. Each shared site contributes ``XZ = -ZX``, while
    operators on different sites commute, so

        X_L Z_L = (-1)**m Z_L X_L.

    Logical X and Z for the same encoded qubit must anticommute; therefore
    ``m`` must be odd for these X-only and Z-only representatives. General
    mixed-Pauli representatives may have even total support overlap—the
    invariant requirement is an odd number of sites whose local Paulis
    anticommute.

    The default rotated-surface-code representatives have ``m = 1``;
    the common weight-three Steane representatives have ``m = 3``.
    """
    if len(x_z_overlap) % 2 == 0:
        raise ValueError(
            "X-only logical-X and Z-only logical-Z representatives for the same "
            "encoded qubit must have odd support overlap so they anticommute."
        )
    return len(x_z_overlap) % 4 == 3


def append_classical_bit_flip(circuit: QuantumCircuit, classical_bit) -> None:
    """Flip a measured classical bit during circuit execution."""
    clbit = circuit.clbits[classical_bit] if isinstance(classical_bit, int) else classical_bit
    circuit.store(clbit, expr.bit_not(clbit))


def append_scheduled_parity_measurements(
    circuit: QuantumCircuit,
    *,
    bases: Sequence[str],
    data_qubit_schedules: Sequence[Sequence[object | None]],
    ancilla_qubits: Sequence,
    classical_bits: Sequence,
) -> None:
    """Append scheduled non-destructive Pauli-string measurements.

    Parameters
    ----------
    circuit
        Circuit to modify.
    bases
        Case-insensitive physical Pauli basis for each schedule row. ``"Z"``
        uses data-controlled CX gates, ``"X"`` uses ancilla-controlled CX
        gates, and ``"Y"`` uses ancilla-controlled CY gates. X- and Y-basis
        ancillas receive Hadamard gates before and after their interactions.
        ``"I"`` measures the corresponding untouched ancilla.
    data_qubit_schedules
        A schedule matrix with one row per parity measurement and one column
        per interaction time step. Each row corresponds to one ancilla. For
        example, ``((q0, q1), (q2, q3))`` schedules ``q0`` and ``q2`` in the
        first step, followed by ``q1`` and ``q3`` in the second. Use ``None``
        when a measurement is idle during a step. All rows must have equal
        length. Row order is paired with ``ancilla_qubits`` and
        ``classical_bits``.
    ancilla_qubits
        Measurement ancillas. The first ``len(data_qubit_schedules)`` entries
        are used and additional entries are untouched. Every schedule row must
        have a distinct ancilla, and every used ancilla must start in ``|0>``.
    classical_bits
        Destination bits for the ancilla measurements, in schedule order.
        Additional entries are untouched.

    Notes
    -----
    This helper does not reset ancillas or insert barriers. The caller owns
    reset policy and any operator-specific outcome-sign interpretation.
    Measurements within each time step act on disjoint data qubits, so a
    schedule with four populated interaction layers has two-qubit depth four.
    """
    schedules = [tuple(schedule) for schedule in data_qubit_schedules]
    bases = list(bases)
    ancillas = list(ancilla_qubits)
    measurement_bits = list(classical_bits)
    num_measurements = len(schedules)

    if len(bases) < num_measurements:
        raise ValueError(f"bases must have at least {num_measurements} entries, got {len(bases)}.")
    if len(ancillas) < num_measurements:
        raise ValueError(
            "ancilla_qubits must have at least " f"{num_measurements} entries, got {len(ancillas)}."
        )
    if len(measurement_bits) < num_measurements:
        raise ValueError(
            "classical_bits must have at least "
            f"{num_measurements} entries, got {len(measurement_bits)}."
        )

    schedule_widths = {len(schedule) for schedule in schedules}
    if len(schedule_widths) > 1:
        raise ValueError("all data-qubit schedules must have the same number of steps.")

    bases = [normalize_pauli_char(basis) for basis in bases[:num_measurements]]
    used_ancillas = ancillas[:num_measurements]
    used_bits = measurement_bits[:num_measurements]
    if len(used_ancillas) != len(set(used_ancillas)):
        raise ValueError("each parity-measurement schedule row requires a unique ancilla.")

    schedule_layers = list(zip(*schedules, strict=True))
    for data_qubits_at_step in schedule_layers:
        active_data_qubits = [
            data_qubit for data_qubit in data_qubits_at_step if data_qubit is not None
        ]
        if len(active_data_qubits) != len(set(active_data_qubits)):
            raise ValueError("a data qubit cannot participate twice in one parallel step.")

    for basis, ancilla in zip(bases, used_ancillas, strict=True):
        if basis in {"X", "Y"}:
            circuit.h(ancilla)

    for data_qubits_at_step in schedule_layers:
        for data_qubit, basis, ancilla in zip(
            data_qubits_at_step,
            bases,
            used_ancillas,
            strict=True,
        ):
            if data_qubit is None:
                continue
            if basis == "Z":
                circuit.cx(data_qubit, ancilla)
            elif basis == "X":
                circuit.cx(ancilla, data_qubit)
            elif basis == "Y":
                circuit.cy(ancilla, data_qubit)

    for basis, ancilla in zip(bases, used_ancillas, strict=True):
        if basis in {"X", "Y"}:
            circuit.h(ancilla)

    for ancilla, classical_bit in zip(used_ancillas, used_bits, strict=True):
        circuit.measure(ancilla, classical_bit)


def append_parity_measurement(
    circuit: QuantumCircuit,
    basis: str,
    *,
    data_qubits: Sequence,
    ancilla_qubit,
    classical_bit,
) -> None:
    """Append a single non-destructive Pauli-string parity measurement.

    Parameters
    ----------
    circuit
        Circuit to modify.
    basis
        Case-insensitive physical Pauli basis. ``"Z"`` appends
        ``CX(data, ancilla)`` for each data qubit. ``"X"`` surrounds
        ``CX(ancilla, data)`` interactions with Hadamard gates on the ancilla.
        ``"Y"`` similarly surrounds ``CY(ancilla, data)`` interactions with
        Hadamards. ``"I"`` measures the untouched ancilla.
    data_qubits
        Ordered support of the physical Pauli string. The order is preserved
        when appending the two-qubit interactions.
    ancilla_qubit
        Measurement ancilla, assumed to start in ``|0>``.
    classical_bit
        Destination bit for the ancilla measurement.

    Notes
    -----
    This helper does not reset the ancilla or insert barriers. It delegates to
    :func:`append_scheduled_parity_measurements` with one schedule row, one
    ancilla, and one classical bit. For data qubits ``(q0, q1, q2, q3)``, the
    complete schedule matrix is ``((q0, q1, q2, q3),)``. The outer tuple has
    exactly one row because this API measures exactly one parity; the four
    entries in that row are successive interaction steps. The caller is
    responsible for any required reset and operator-specific outcome-sign
    interpretation.
    """
    append_scheduled_parity_measurements(
        circuit,
        bases=(basis,),
        data_qubit_schedules=(tuple(data_qubits),),
        ancilla_qubits=(ancilla_qubit,),
        classical_bits=(classical_bit,),
    )


# ---------------------------------------------------------------------------
# Logical measurements
# ---------------------------------------------------------------------------


def append_pauli_string_measurement(
    circuit: QuantumCircuit,
    *,
    paulis: Sequence[str],
    data_qubits: Sequence,
    ancilla_qubit,
    classical_bit,
) -> None:
    """Append a non-destructive measurement of a physical Pauli string.

    ``paulis`` and ``data_qubits`` give the Pauli acting on each physical
    qubit. Homogeneous strings reuse :func:`append_parity_measurement`. Mixed
    strings prepare the ancilla in ``|+>``, apply the corresponding controlled
    Pauli from the ancilla to each data qubit, return the ancilla to the Z
    basis, and measure it.

    The ancilla is assumed to start in ``|0>``. This helper does not reset the
    ancilla, insert barriers, or interpret an operator-specific outcome sign.
    """
    selected_paulis = tuple(normalize_pauli_char(pauli) for pauli in paulis)
    selected_data_qubits = tuple(data_qubits)
    if len(selected_paulis) != len(selected_data_qubits):
        raise ValueError("paulis and data_qubits must have the same length.")
    if len(selected_data_qubits) != len(set(selected_data_qubits)):
        raise ValueError("a Pauli string can act on each data qubit at most once.")

    if selected_paulis and len(set(selected_paulis)) == 1:
        append_parity_measurement(
            circuit,
            selected_paulis[0],
            data_qubits=selected_data_qubits,
            ancilla_qubit=ancilla_qubit,
            classical_bit=classical_bit,
        )
        return

    circuit.h(ancilla_qubit)
    for pauli, data_qubit in zip(selected_paulis, selected_data_qubits, strict=True):
        if pauli == "X":
            circuit.cx(ancilla_qubit, data_qubit)
        elif pauli == "Y":
            circuit.cy(ancilla_qubit, data_qubit)
        elif pauli == "Z":
            circuit.cz(ancilla_qubit, data_qubit)
    circuit.h(ancilla_qubit)
    circuit.measure(ancilla_qubit, classical_bit)


def resolve_logical_measurement_terms(
    basis: str,
    *,
    logical_x_support: tuple[_SupportT, ...],
    logical_z_support: tuple[_SupportT, ...],
) -> tuple[tuple[tuple[_SupportT, str], ...], bool]:
    """Resolve a logical Pauli into ordered support-Pauli terms and its sign."""
    if basis == "X":
        return tuple((support_item, "X") for support_item in logical_x_support), False
    if basis == "Z":
        return tuple((support_item, "Z") for support_item in logical_z_support), False

    logical_x_set = set(logical_x_support)
    logical_z_set = set(logical_z_support)
    if len(logical_x_set) != len(logical_x_support):
        raise ValueError("logical_x_support must not contain duplicate indices.")
    if len(logical_z_set) != len(logical_z_support):
        raise ValueError("logical_z_support must not contain duplicate indices.")

    x_z_overlap = tuple(
        support_item
        for support_item in logical_x_support
        if support_item in logical_z_set
    )
    invert_parity = logical_y_parity_is_inverted(x_z_overlap)
    terms = tuple(
        (support_item, "Y" if support_item in logical_z_set else "X")
        for support_item in logical_x_support
    ) + tuple(
        (support_item, "Z")
        for support_item in logical_z_support
        if support_item not in logical_x_set
    )
    return terms, invert_parity


def append_logical_measurement_on_indexed_qubits(
    circuit: QuantumCircuit,
    basis: str,
    *,
    indexed_qubits: Sequence,
    default_logical_x: tuple[int, ...],
    default_logical_z: tuple[int, ...],
    support: tuple[int, ...] | None,
    ancilla_qubit,
    classical_bits: Sequence,
    destructive: bool,
) -> None:
    """Append a logical Pauli measurement to an existing circuit.

    Parameters
    ----------
    circuit
        Circuit being modified.
    basis
        Logical measurement basis: ``"X"``, ``"Y"``, or ``"Z"``.
    indexed_qubits
        Physical qubits addressed by the integer indices in ``support``.
    default_logical_x, default_logical_z
        Default logical-X and logical-Z supports used when ``support`` is
        omitted. Logical Y measures ``i X_L Z_L`` using both supports.
    support
        Physical support expressed as indices into ``indexed_qubits``. For X
        or Z this overrides the corresponding default support. For Y it
        denotes a common X/Z support, as used by the self-dual Steane code.
    ancilla_qubit
        Ancilla used for a non-destructive parity measurement.
    classical_bits
        Destination bits: one per support qubit when destructive, otherwise one.
    destructive
        Directly measure the support when true; otherwise measure its parity.

    A logical-Y measurement uses Y on the intersection of the selected
    logical-X and logical-Z representatives, X on the X-only sites, and Z on
    the Z-only sites. Outcomes are sign-corrected so bit parity zero denotes
    the logical ``+1`` eigenvalue.
    """
    basis = normalize_pauli_char(basis)

    qubits = list(indexed_qubits)
    measurement_bits = list(classical_bits)
    if support is not None:
        logical_x_support = support if basis in {"X", "Y"} else default_logical_x
        logical_z_support = support if basis in {"Z", "Y"} else default_logical_z
    else:
        logical_x_support = default_logical_x
        logical_z_support = default_logical_z
    measurement_terms, invert_parity = resolve_logical_measurement_terms(
        basis,
        logical_x_support=logical_x_support,
        logical_z_support=logical_z_support,
    )

    if destructive:
        if len(measurement_bits) < len(measurement_terms):
            raise ValueError(
                "classical_bits must have at least "
                f"{len(measurement_terms)} entries for destructive measurement."
            )

        for index, physical_basis in measurement_terms:
            if physical_basis == "X":
                circuit.h(qubits[index])
            elif physical_basis == "Y":
                circuit.sdg(qubits[index])
                circuit.h(qubits[index])

        for output_index, (index, _) in enumerate(measurement_terms):
            circuit.measure(qubits[index], measurement_bits[output_index])
    else:
        if ancilla_qubit is None:
            raise ValueError("ancilla_qubit is required for non-destructive measurement.")
        if not measurement_bits:
            raise ValueError(
                "classical_bits must have at least 1 entry for " "non-destructive measurement."
            )

        append_pauli_string_measurement(
            circuit,
            paulis=tuple(physical_basis for _, physical_basis in measurement_terms),
            data_qubits=[qubits[index] for index, _ in measurement_terms],
            ancilla_qubit=ancilla_qubit,
            classical_bit=measurement_bits[0],
        )

    if invert_parity:
        append_classical_bit_flip(circuit, measurement_bits[0])


# ---------------------------------------------------------------------------
# Depth metrics
# ---------------------------------------------------------------------------


def two_qubit_depth(circuit: QuantumCircuit) -> int:
    """Return the two-qubit gate depth of a circuit."""
    return circuit.depth(
        filter_function=lambda inst: inst.operation.num_qubits == 2,
    )


# ---------------------------------------------------------------------------
# IBM device access
# ---------------------------------------------------------------------------

IBM_RUNTIME_ACCOUNT = "QR"
IBM_RUNTIME_BACKEND = "ibm_miami"
_IBM_RUNTIME_ACCOUNT_KWARGS = {
    "channel",
    "token",
    "url",
    "instance",
    "proxies",
    "verify",
    "private_endpoint",
    "region",
    "plans_preference",
    "tags",
}


def ibm_runtime_service(account_name: str = IBM_RUNTIME_ACCOUNT) -> Any:
    """Return a Qiskit Runtime service using a saved account when available.

    Loading the saved account explicitly avoids differences between test guards
    and test bodies: both paths use the same credential metadata from
    ``QiskitRuntimeService.saved_accounts()`` before falling back to the named
    account lookup.
    """
    from qiskit_ibm_runtime import QiskitRuntimeService

    saved_account = QiskitRuntimeService.saved_accounts().get(account_name)
    if saved_account is None:
        return QiskitRuntimeService(name=account_name)

    service_kwargs = {
        key: value
        for key, value in saved_account.items()
        if key in _IBM_RUNTIME_ACCOUNT_KWARGS and value is not None
    }
    return QiskitRuntimeService(**service_kwargs)


def ibm_backend(
    account_name: str = IBM_RUNTIME_ACCOUNT,
    backend_name: str = IBM_RUNTIME_BACKEND,
) -> Any:
    """Return a backend from the configured IBM Runtime saved account."""
    return ibm_runtime_service(account_name).backend(backend_name)


def ibm_backend_available(
    account_name: str = IBM_RUNTIME_ACCOUNT,
    backend_name: str = IBM_RUNTIME_BACKEND,
) -> bool:
    """Return True if the named IBM Runtime account and backend are reachable."""
    try:
        ibm_backend(account_name, backend_name)
        return True
    except Exception:
        return False
