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


def append_parity_measurement(
    circuit: QuantumCircuit,
    basis: str,
    *,
    data_qubits: Sequence,
    ancilla_qubit,
    classical_bit,
) -> None:
    """Append a non-destructive Pauli-string parity measurement.

    The ancilla is assumed to start in ``|0>``. The caller owns any required
    reset and is responsible for interpreting operator-specific sign
    conventions after the physical parity is measured.
    """
    normalized_basis = normalize_pauli_char(basis)

    if normalized_basis == "Z":
        for data_qubit in data_qubits:
            circuit.cx(data_qubit, ancilla_qubit)
    elif normalized_basis in {"X", "Y"}:
        circuit.h(ancilla_qubit)
        for data_qubit in data_qubits:
            if normalized_basis == "X":
                circuit.cx(ancilla_qubit, data_qubit)
            else:
                circuit.cy(ancilla_qubit, data_qubit)
        circuit.h(ancilla_qubit)

    circuit.measure(ancilla_qubit, classical_bit)


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
    normalized_basis = normalize_pauli_char(basis)
    if normalized_basis == "Y" and set(default_logical_x) != set(default_logical_z):
        raise ValueError("logical Y requires common logical X and Z support.")

    qubits = list(indexed_qubits)
    measurement_bits = list(classical_bits)
    selected_support = support
    if selected_support is None:
        selected_support = default_logical_z if normalized_basis == "Z" else default_logical_x

    invert_y_parity = normalized_basis == "Y" and logical_y_parity_is_inverted(selected_support)

    if destructive:
        if len(measurement_bits) < len(selected_support):
            raise ValueError(
                "classical_bits must have at least "
                f"{len(selected_support)} entries for destructive measurement."
            )

        if normalized_basis == "X":
            for index in selected_support:
                circuit.h(qubits[index])
        elif normalized_basis == "Y":
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
            normalized_basis,
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
