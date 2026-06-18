"""Deterministic fault-injection helpers for the 11-site Steane patch (Qiskit).

Provides a single reusable function that appends physical Pauli gates to
specified data sites on an 11-site cultivation patch circuit.
"""

from __future__ import annotations

from collections.abc import Sequence

from qiskit import QuantumCircuit

from research_qec_haoran.steane_code.state_injection_qiskit import PATCH_DATA_SITES


def append_patch_pauli_faults(
    circuit: QuantumCircuit,
    faults: Sequence[tuple[str, int]],
    patch_qubits: Sequence | None = None,
) -> None:
    """Append deterministic Pauli faults to data sites on the 11-site patch.

    Parameters
    ----------
    circuit
        Circuit being modified.
    faults
        Sequence of ``(pauli, site)`` pairs where *pauli* is ``"X"``,
        ``"Y"``, or ``"Z"`` and *site* is an 11-site patch label
        (must be one of ``PATCH_DATA_SITES``).
    patch_qubits
        Optional mapping from site label (int) to actual Qiskit qubit.
        If ``None``, site labels are used as raw qubit indices.

    Raises
    ------
    ValueError
        If a Pauli string is unrecognized or a site is not a data site.
    """
    for pauli, site in faults:
        if site not in PATCH_DATA_SITES:
            raise ValueError(
                f"site {site} is not a data site; valid data sites are {PATCH_DATA_SITES}"
            )
        qubit = site if patch_qubits is None else patch_qubits[site]
        p = pauli.upper()
        if p == "X":
            circuit.x(qubit)
        elif p == "Y":
            circuit.y(qubit)
        elif p == "Z":
            circuit.z(qubit)
        else:
            raise ValueError(f"unsupported Pauli {pauli!r}; expected 'X', 'Y', or 'Z'")

