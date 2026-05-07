"""Steane-code magic-state cultivation projection circuit in Qiskit.

This module follows the distance-3 color-code/Steane-code construction in
Rosenfeld et al., arXiv:2512.13908, Fig. 1 and Fig. S11.

The cultivation projection circuit measures the logical Hadamard axis by
applying paper ``T``, folding the logical ``X`` component into root
measurements, reversing the fold with paper ``T-dagger``, and finally measuring
flag qubits.  Accepted shots have all ``root`` and ``flag`` bits equal to ``0``.

For the state-injection circuit that prepares the encoded ``|T_L>`` candidate,
see ``state_injection_qiskit.py`` in the same package.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.quantum_info import DensityMatrix, Statevector

from research_qec_haoran.steane_code.state_injection_qiskit import (
    INJECTION_DETECTOR_QUBITS,
    PAPER_T_ANGLE,
    STEANE_DATA_QUBITS,
    STEANE_INJECTION_QUBIT,
    STEANE_X_STABILIZERS,
    STEANE_Z_STABILIZERS,
    _append_injection_step,
    _append_native_ops,
    _hadamard_all_matrix,
    _identity_matrix,
    _logical_pauli_matrix,
    _measure,
    _reset,
    _snapshot,
    build_noisy_t_state_injection_circuit,
    build_t_state_injection_circuit,
    build_t_state_injection_stage_circuits,
    encoded_t_statevector,
    encoded_theta_statevector,
    logical_bloch_components,
    noisy_encoded_t_density_matrix,
)

# Re-export injection symbols for backward compatibility
__all__ = [
    "CULTIVATION_FLAG_QUBITS",
    "CULTIVATION_ROOT_QUBITS",
    "INJECTION_DETECTOR_QUBITS",
    "PAPER_T_ANGLE",
    "STEANE_DATA_QUBITS",
    "STEANE_INJECTION_QUBIT",
    "STEANE_X_STABILIZERS",
    "STEANE_Z_STABILIZERS",
    "ProjectionResult",
    "build_cultivation_projection_circuit",
    "build_cultivation_stage_circuits",
    "build_noisy_t_state_injection_circuit",
    "build_noisy_t_state_then_cultivation_circuit",
    "build_t_state_injection_circuit",
    "build_t_state_injection_stage_circuits",
    "build_t_state_then_cultivation_circuit",
    "encoded_t_statevector",
    "encoded_theta_statevector",
    "logical_bloch_components",
    "noisy_encoded_t_density_matrix",
    "project_onto_magic_axis",
]

CULTIVATION_ROOT_QUBITS: tuple[int, ...] = (6, 9)
CULTIVATION_FLAG_QUBITS: tuple[int, ...] = (2, 4, 6, 9)

_NativeOp = tuple[str, tuple[int, ...], tuple[float, ...]]

_CULTIVATION_FORWARD_FOLD_OPS: tuple[_NativeOp, ...] = (
    ("h", (0,), ()),
    ("h", (1,), ()),
    ("cx", (1, 2), ()),
    ("cx", (0, 2), ()),
    ("h", (3,), ()),
    ("h", (4,), ()),
    ("cx", (4, 5), ()),
    ("h", (6,), ()),
    ("cx", (6, 7), ()),
    ("h", (7,), ()),
    ("h", (8,), ()),
    ("cx", (8, 9), ()),
    ("h", (9,), ()),
    ("cx", (5, 9), ()),
    ("h", (5,), ()),
    ("cx", (2, 5), ()),
    ("cx", (6, 10), ()),
    ("h", (6,), ()),
    ("cx", (3, 6), ()),
    ("cx", (5, 6), ()),
    ("h", (10,), ()),
    ("cx", (9, 11), ()),
    ("h", (9,), ()),
    ("h", (11,), ()),
    ("cx", (9, 11), ()),
    ("h", (9,), ()),
    ("h", (11,), ()),
)

_CULTIVATION_REVERSE_FOLD_AND_T_DAGGER_OPS: tuple[_NativeOp, ...] = (
    ("cx", (5, 6), ()),
    ("cx", (2, 5), ()),
    ("cx", (0, 2), ()),
    ("h", (0,), ()),
    ("ry", (0,), (-PAPER_T_ANGLE,)),
    ("cx", (1, 2), ()),
    ("h", (1,), ()),
    ("ry", (1,), (-PAPER_T_ANGLE,)),
    ("cx", (3, 6), ()),
    ("h", (3,), ()),
    ("ry", (3,), (-PAPER_T_ANGLE,)),
    ("h", (5,), ()),
    ("h", (6,), ()),
    ("h", (7,), ()),
    ("h", (9,), ()),
    ("h", (10,), ()),
    ("cx", (6, 10), ()),
    ("cx", (6, 7), ()),
    ("h", (6,), ()),
    ("ry", (7,), (-PAPER_T_ANGLE,)),
    ("ry", (10,), (-PAPER_T_ANGLE,)),
    ("h", (11,), ()),
    ("cx", (9, 11), ()),
    ("h", (9,), ()),
    ("h", (11,), ()),
    ("cx", (9, 11), ()),
    ("cx", (5, 9), ()),
    ("cx", (4, 5), ()),
    ("h", (4,), ()),
    ("ry", (5,), (-PAPER_T_ANGLE,)),
    ("h", (9,), ()),
    ("cx", (8, 9), ()),
    ("h", (8,), ()),
    ("ry", (8,), (-PAPER_T_ANGLE,)),
)


@dataclass(frozen=True)
class ProjectionResult:
    """Normalized postselected state and its Born acceptance probability."""

    state: Statevector | DensityMatrix
    acceptance_probability: float


def build_cultivation_projection_circuit() -> QuantumCircuit:
    """Build the standalone Fig. 1/Fig. S11 cultivation projection circuit.

    The input is assumed to already be encoded on ``STEANE_DATA_QUBITS``.
    The circuit measures the logical Hadamard axis by applying paper ``T``,
    folding the logical ``X`` component into root measurements, reversing the
    fold with paper ``T-dagger``, and finally measuring flag qubits.  Accepted shots
    have all ``root`` and ``flag`` bits equal to ``0``.
    """

    q = QuantumRegister(12, "q")
    root = ClassicalRegister(len(CULTIVATION_ROOT_QUBITS), "root")
    flag = ClassicalRegister(len(CULTIVATION_FLAG_QUBITS), "flag")
    circuit = QuantumCircuit(q, root, flag, name="steane_t_cultivation_projection")
    _append_cultivation_step(circuit, q, root, flag)
    return circuit


def build_cultivation_stage_circuits() -> tuple[tuple[str, QuantumCircuit], ...]:
    """Return cumulative ideal cultivation-projection construction stages."""

    q = QuantumRegister(12, "q")
    root = ClassicalRegister(len(CULTIVATION_ROOT_QUBITS), "root")
    flag = ClassicalRegister(len(CULTIVATION_FLAG_QUBITS), "flag")
    circuit = QuantumCircuit(q, root, flag, name="steane_t_cultivation_stages")
    snapshots: list[tuple[str, QuantumCircuit]] = []
    _append_cultivation_step(circuit, q, root, flag, snapshots=snapshots)
    return tuple(snapshots)


def build_noisy_t_state_then_cultivation_circuit(
    *,
    theta: float = PAPER_T_ANGLE,
    injection_error_probability: float = 0.0,
) -> QuantumCircuit:
    """Build state injection followed immediately by cultivation projection.

    This is the requested stop point: a noisy Steane-code ``|T>`` preparation,
    then the Fig. 1/S11 projection checks, with no tomography or grafting.
    Postselect accepted shots by requiring all bits in ``inj``, ``root``, and
    ``flag`` to be ``0``.
    """

    q = QuantumRegister(12, "q")
    inj = ClassicalRegister(len(INJECTION_DETECTOR_QUBITS), "inj")
    root = ClassicalRegister(len(CULTIVATION_ROOT_QUBITS), "root")
    flag = ClassicalRegister(len(CULTIVATION_FLAG_QUBITS), "flag")
    circuit = QuantumCircuit(q, inj, root, flag, name="steane_noisy_t_then_cultivation")
    _append_injection_step(
        circuit,
        q,
        inj,
        theta=theta,
        injection_error_probability=injection_error_probability,
    )
    circuit.barrier()
    _append_cultivation_step(circuit, q, root, flag)
    return circuit


def build_t_state_then_cultivation_circuit(*, theta: float = PAPER_T_ANGLE) -> QuantumCircuit:
    """Build ideal state injection followed by ideal cultivation projection."""

    return build_noisy_t_state_then_cultivation_circuit(
        theta=theta,
        injection_error_probability=0.0,
    )


def project_onto_magic_axis(
    state: Statevector | DensityMatrix,
    *,
    outcome: Literal["+", "-"] = "+",
    zero_tolerance: float = 1e-15,
) -> ProjectionResult:
    """Project a seven-qubit Steane state onto a logical-Hadamard eigenspace.

    ``outcome="+"`` is the accepted cultivation branch that projects toward
    ``|T_L>``.  The returned state is normalized; the accompanying probability
    is the unnormalized trace/norm before postselection.
    """

    if outcome not in {"+", "-"}:
        raise ValueError("outcome must be '+' or '-'")
    sign = 1 if outcome == "+" else -1
    projector = (_identity_matrix(7) + sign * _hadamard_all_matrix(7)) / 2

    if isinstance(state, Statevector):
        vector = projector @ state.data
        probability = float(np.vdot(vector, vector).real)
        if probability < zero_tolerance:
            raise ValueError("postselected branch has zero probability")
        return ProjectionResult(Statevector(vector / math.sqrt(probability)), probability)

    if isinstance(state, DensityMatrix):
        matrix = projector @ state.data @ projector
        probability = float(np.trace(matrix).real)
        if probability < zero_tolerance:
            raise ValueError("postselected branch has zero probability")
        return ProjectionResult(DensityMatrix(matrix / probability), probability)

    raise TypeError("state must be a qiskit.quantum_info.Statevector or DensityMatrix")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _paper_t(
    circuit: QuantumCircuit,
    q: QuantumRegister,
    qubits: tuple[int, ...],
    *,
    dagger: bool = False,
) -> None:
    angle = -PAPER_T_ANGLE if dagger else PAPER_T_ANGLE
    for index in qubits:
        circuit.ry(angle, q[index])


def _append_cultivation_step(
    circuit: QuantumCircuit,
    q: QuantumRegister,
    root: ClassicalRegister,
    flag: ClassicalRegister,
    snapshots: list[tuple[str, QuantumCircuit]] | None = None,
) -> None:
    _reset(circuit, q, (2, 4, 6, 9))
    _paper_t(circuit, q, STEANE_DATA_QUBITS)
    circuit.barrier()
    _snapshot("1_apply_transversal_paper_t", circuit, snapshots)

    _append_native_ops(circuit, q, _CULTIVATION_FORWARD_FOLD_OPS)
    _snapshot("2_forward_fold_to_roots", circuit, snapshots)
    _measure(circuit, q, root, CULTIVATION_ROOT_QUBITS)
    _snapshot("3_root_measurement", circuit, snapshots)

    _reset(circuit, q, (6, 9))
    _append_native_ops(circuit, q, _CULTIVATION_REVERSE_FOLD_AND_T_DAGGER_OPS)
    circuit.barrier()
    _snapshot("4_time_reversed_fold_and_paper_t_dagger", circuit, snapshots)
    _measure(circuit, q, flag, CULTIVATION_FLAG_QUBITS)
    _snapshot("5_flag_measurements", circuit, snapshots)




