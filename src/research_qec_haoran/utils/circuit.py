"""Circuit-level utility helpers (gate decompositions, depth metrics, device access)."""

from __future__ import annotations

from qiskit import QuantumCircuit


# ---------------------------------------------------------------------------
# Gate decompositions
# ---------------------------------------------------------------------------

def apply_ty(circuit: QuantumCircuit, qubit) -> None:
    """Apply ``T_y = RY(pi/4) = exp(-i*pi*Y/8)``.

    This is the rotation satisfying ``H = T_y^dagger X T_y``, used in
    magic-state cultivation to map the Hadamard-axis observable onto the X basis.

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

def ibm_backend_available(
    account_name: str = "main-flex-us",
    backend_name: str = "ibm_miami",
) -> bool:
    """Return True if the named IBM Runtime account and backend are reachable."""
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService

        service = QiskitRuntimeService(name=account_name)
        service.backend(backend_name)
        return True
    except Exception:
        return False

