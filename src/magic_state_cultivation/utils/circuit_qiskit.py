"""Circuit-level utility helpers (gate decompositions, depth metrics, device access)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from qiskit import QuantumCircuit
from qiskit.circuit.classical import expr

from magic_state_cultivation.utils.normalize import normalize_pauli_char

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
# Parity measurements & Logical measurements
# ---------------------------------------------------------------------------


def logical_y_parity_is_inverted(support: tuple[int, ...]) -> bool:
    """Return whether measuring the physical Y string reverses logical Y.

    For an odd support of weight w,

        prod(Y_j) = i**(w - 1) Y_L,  where Y_L = i X_L Z_L.

    Thus a weight-3 representative obeys ``prod(Y_j) = -Y_L``.
    """
    if len(support) % 2 == 0:
        raise ValueError("Y logical support must have odd weight.")
    return len(support) % 4 == 3


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

    Raises
    ------
    ValueError
        If a basis is not a supported Pauli label, fewer bases, ancillas, or
        classical bits are supplied than schedules, schedule lengths differ,
        one ancilla is assigned to more than one schedule row, or one data
        qubit appears in more than one measurement during the same time step.

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
        raise ValueError(
            f"bases must have at least {num_measurements} entries, got {len(bases)}."
        )
    if len(ancillas) < num_measurements:
        raise ValueError(
            "ancilla_qubits must have at least "
            f"{num_measurements} entries, got {len(ancillas)}."
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

    Raises
    ------
    ValueError
        If ``basis`` is not a supported Pauli label.

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
        Default supports used when ``support`` is omitted.
    support
        Physical support expressed as indices into ``indexed_qubits``.
    ancilla_qubit
        Ancilla used for a non-destructive parity measurement.
    classical_bits
        Destination bits: one per support qubit when destructive, otherwise one.
    destructive
        Directly measure the support when true; otherwise measure its parity.

    Logical-Y outcomes are sign-corrected so bit parity zero denotes the
    logical ``+1`` eigenvalue.
    """
    basis = normalize_pauli_char(basis)
    if basis == "Y" and set(default_logical_x) != set(default_logical_z):
        raise ValueError("logical Y requires common logical X and Z support.")

    qubits = list(indexed_qubits)
    measurement_bits = list(classical_bits)
    selected_support = support
    if selected_support is None:
        selected_support = default_logical_z if basis == "Z" else default_logical_x

    invert_y_parity = basis == "Y" and logical_y_parity_is_inverted(selected_support)

    if destructive:
        if len(measurement_bits) < len(selected_support):
            raise ValueError(
                "classical_bits must have at least "
                f"{len(selected_support)} entries for destructive measurement."
            )

        if basis == "X":
            for index in selected_support:
                circuit.h(qubits[index])
        elif basis == "Y":
            for index in selected_support:
                circuit.sdg(qubits[index])
                circuit.h(qubits[index])

        for output_index, index in enumerate(selected_support):
            circuit.measure(qubits[index], measurement_bits[output_index])
    else:
        if ancilla_qubit is None:
            raise ValueError("ancilla_qubit is required for non-destructive measurement.")
        if not measurement_bits:
            raise ValueError(
                "classical_bits must have at least 1 entry for " "non-destructive measurement."
            )

        append_parity_measurement(
            circuit,
            basis,
            data_qubits=[qubits[index] for index in selected_support],
            ancilla_qubit=ancilla_qubit,
            classical_bit=measurement_bits[0],
        )

    if invert_y_parity:
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
