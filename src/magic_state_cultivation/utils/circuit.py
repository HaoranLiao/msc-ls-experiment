"""Circuit-level utility helpers (gate decompositions, depth metrics, device access)."""

from __future__ import annotations

from typing import Any

from qiskit import QuantumCircuit

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
