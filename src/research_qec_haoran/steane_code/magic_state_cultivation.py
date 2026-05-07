"""Steane-code magic-state injection and cultivation circuits in Qiskit.

This module follows the distance-3 color-code/Steane-code construction in
Rosenfeld et al., arXiv:2512.13908, Fig. 1 and Fig. S11.

Conventions
-----------
The paper's magic-state axis uses

    T_hat = exp(-i pi Y / 8) = RY(pi / 4)

and therefore differs from Qiskit's usual phase ``TGate``.  All functions in
this module use the paper convention and call it ``paper T`` in labels and
comments.

The gate schedules in ``build_noisy_t_state_injection_circuit`` and
``build_cultivation_projection_circuit`` are Qiskit translations of the
standalone explicit circuits linked from the paper's supplemental Sec. VII.
The linked circuits are written as controlled-phase layers.  This module emits
an equivalent native Qiskit circuit with local basis-change simplifications
applied directly.  In particular, local ``H-CZ-H`` patterns are emitted as
native ``CX`` gates when that reduces the gate count.  The circuits stop after
the cultivation projection and its root/flag measurements; they do not append
tomography, kickback tomography, QEC idling, or grafting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.quantum_info import DensityMatrix, Kraus, Statevector

PAPER_T_ANGLE = math.pi / 4

# Physical qubit indices used by the explicit cultivation circuit.  The seven
# data qubits are the Steane-code/color-code block.  The other qubits are
# helper/flag/root-measurement qubits used during injection and cultivation.
STEANE_DATA_QUBITS: tuple[int, ...] = (0, 1, 3, 5, 7, 8, 10)
STEANE_INJECTION_QUBIT = 5
INJECTION_DETECTOR_QUBITS: tuple[int, ...] = (2, 4, 9, 6)
CULTIVATION_ROOT_QUBITS: tuple[int, ...] = (6, 9)
CULTIVATION_FLAG_QUBITS: tuple[int, ...] = (2, 4, 6, 9)

# The three Fig. S11 color-code plaquettes over canonical positions 0..6 inside
# the seven-qubit block.  Map these positions to physical circuit qubits with
# STEANE_DATA_QUBITS.  Logical X and Z are both transversal.
STEANE_X_STABILIZERS: tuple[tuple[int, ...], ...] = (
    (1, 5, 6, 3),
    (1, 3, 2, 0),
    (3, 6, 4, 2),
)
STEANE_Z_STABILIZERS = STEANE_X_STABILIZERS

_ALL_DATA_MASK = (1 << 7) - 1
_PAULI_I = np.array([[1, 0], [0, 1]], dtype=complex)
_PAULI_X = np.array([[0, 1], [1, 0]], dtype=complex)
_PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_PAULI_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_HADAMARD = np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2)

_NativeOp = tuple[str, tuple[int, ...], tuple[float, ...]]

# These op lists are the Fig. S11 controlled-phase layer schedules after local
# basis-change simplification into Qiskit-native H/CX/RY operations.  They are
# kept explicit so construction and display use the same minimal native gates,
# instead of relying on a post-hoc transpiler wrapper.
_INJECTION_ENCODING_OPS: tuple[_NativeOp, ...] = (
    ("h", (0,), ()),
    ("h", (1,), ()),
    ("h", (2,), ()),
    ("cx", (1, 4), ()),
    ("h", (4,), ()),
    ("h", (5,), ()),
    ("cx", (2, 5), ()),
    ("h", (2,), ()),
    ("cx", (2, 3), ()),
    ("h", (2,), ()),
    ("cx", (1, 2), ()),
    ("h", (5,), ()),
    ("cx", (4, 5), ()),
    ("h", (4,), ()),
    ("h", (5,), ()),
    ("cx", (3, 6), ()),
    ("h", (3,), ()),
    ("cx", (2, 3), ()),
    ("h", (2,), ()),
    ("cx", (0, 2), ()),
    ("h", (2,), ()),
    ("cx", (1, 2), ()),
    ("h", (2,), ()),
    ("h", (6,), ()),
    ("cx", (5, 6), ()),
    ("h", (5,), ()),
    ("cx", (2, 5), ()),
    ("h", (5,), ()),
    ("h", (8,), ()),
    ("cx", (8, 9), ()),
    ("cx", (4, 8), ()),
    ("cx", (1, 4), ()),
    ("h", (1,), ()),
    ("h", (4,), ()),
    ("h", (8,), ()),
    ("h", (9,), ()),
    ("cx", (6, 10), ()),
    ("cx", (3, 6), ()),
    ("h", (3,), ()),
    ("cx", (2, 3), ()),
    ("cx", (0, 2), ()),
    ("h", (6,), ()),
    ("cx", (6, 7), ()),
    ("h", (6,), ()),
    ("cx", (5, 6), ()),
    ("h", (5,), ()),
    ("cx", (4, 5), ()),
    ("h", (4,), ()),
    ("h", (7,), ()),
    ("cx", (6, 7), ()),
    ("h", (6,), ()),
    ("h", (7,), ()),
    ("h", (10,), ()),
    ("cx", (9, 10), ()),
    ("cx", (8, 9), ()),
)

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


def build_noisy_t_state_injection_circuit(
    *,
    theta: float = PAPER_T_ANGLE,
    injection_error_probability: float = 0.0,
) -> QuantumCircuit:
    r"""Build the paper's Steane-code state-injection circuit.

    The injected single-qubit state is
    ``|theta> = exp(-i theta Y / 2)|0>`` on physical qubit 5.  The default
    ``theta = pi/4`` is the paper's ``|T>`` axis state.  Optional injection
    noise is a one-qubit depolarizing channel on the non-fault-tolerant
    injection qubit before the state is expanded into the Steane block.

    The output logical state lives on ``STEANE_DATA_QUBITS`` after
    postselecting the four ``inj`` detector bits to ``0``.
    """

    q = QuantumRegister(11, "q")
    inj = ClassicalRegister(len(INJECTION_DETECTOR_QUBITS), "inj")
    circuit = QuantumCircuit(q, inj, name="steane_noisy_t_injection")
    _append_injection_step(
        circuit,
        q,
        inj,
        theta=theta,
        injection_error_probability=injection_error_probability,
    )
    return circuit


def build_t_state_injection_circuit(*, theta: float = PAPER_T_ANGLE) -> QuantumCircuit:
    r"""Build the ideal paper-convention Steane-code ``|T_L>`` injection circuit."""

    return build_noisy_t_state_injection_circuit(
        theta=theta,
        injection_error_probability=0.0,
    )


def build_t_state_injection_stage_circuits(
    *,
    theta: float = PAPER_T_ANGLE,
) -> tuple[tuple[str, QuantumCircuit], ...]:
    """Return cumulative ideal state-injection construction stages."""

    q = QuantumRegister(11, "q")
    inj = ClassicalRegister(len(INJECTION_DETECTOR_QUBITS), "inj")
    circuit = QuantumCircuit(q, inj, name="steane_t_injection_stages")
    snapshots: list[tuple[str, QuantumCircuit]] = []
    _append_injection_step(
        circuit,
        q,
        inj,
        theta=theta,
        injection_error_probability=0.0,
        snapshots=snapshots,
    )
    return tuple(snapshots)


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


def encoded_theta_statevector(theta: float) -> Statevector:
    r"""Return ``cos(theta/2)|0_L> + sin(theta/2)|1_L>`` for the Steane code."""

    zero = _encoded_zero_vector()
    one = _encoded_one_vector()
    vector = math.cos(theta / 2) * zero + math.sin(theta / 2) * one
    return Statevector(vector)


def encoded_t_statevector() -> Statevector:
    """Return the paper-convention encoded Steane ``|T_L>`` state."""

    return encoded_theta_statevector(PAPER_T_ANGLE)


def noisy_encoded_t_density_matrix(logical_error_probability: float) -> DensityMatrix:
    """Return a simple logical depolarizing-noise model for encoded ``|T_L>``.

    This state-level helper is useful for analytic checks of the projection.
    It is separate from the circuit builder, whose ``injection_error_probability``
    places physical depolarizing noise on the non-fault-tolerant injection
    qubit before encoding.
    """

    _validate_probability(logical_error_probability, "logical_error_probability")
    pure = np.outer(encoded_t_statevector().data, encoded_t_statevector().data.conj())
    if logical_error_probability == 0:
        return DensityMatrix(pure)

    noisy = (1 - logical_error_probability) * pure
    for axis in ("X", "Y", "Z"):
        op = _logical_pauli_matrix(axis)
        noisy += (logical_error_probability / 3) * (op @ pure @ op.conj().T)
    return DensityMatrix(noisy)


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


def logical_bloch_components(state: Statevector | DensityMatrix) -> dict[str, float]:
    """Return logical ``X``, ``Y``, and ``Z`` expectations on the Steane block."""

    values: dict[str, float] = {}
    matrix = state.data
    for axis in ("X", "Y", "Z"):
        op = _logical_pauli_matrix(axis)
        if isinstance(state, Statevector):
            value = np.vdot(matrix, op @ matrix)
        elif isinstance(state, DensityMatrix):
            value = np.trace(matrix @ op)
        else:
            raise TypeError("state must be a qiskit.quantum_info.Statevector or DensityMatrix")
        values[axis] = float(np.real_if_close(value).real)
    return values


def _append_injection_step(
    circuit: QuantumCircuit,
    q: QuantumRegister,
    inj: ClassicalRegister,
    *,
    theta: float,
    injection_error_probability: float,
    snapshots: list[tuple[str, QuantumCircuit]] | None = None,
) -> None:
    _validate_probability(injection_error_probability, "injection_error_probability")

    _reset(circuit, q, (1, 2, 3, 6, 7, 8, 5, 4, 9, 10, 0))
    circuit.ry(theta, q[STEANE_INJECTION_QUBIT])
    if injection_error_probability:
        circuit.append(
            _single_qubit_depolarizing_channel(injection_error_probability),
            [q[STEANE_INJECTION_QUBIT]],
        )
    _snapshot("1_physical_t_input", circuit, snapshots)
    circuit.barrier()

    _append_native_ops(circuit, q, _INJECTION_ENCODING_OPS)
    _snapshot("2_steane_encoding_network", circuit, snapshots)
    _measure(circuit, q, inj, INJECTION_DETECTOR_QUBITS)
    _snapshot("3_injection_detector_readout", circuit, snapshots)


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


def _snapshot(
    label: str,
    circuit: QuantumCircuit,
    snapshots: list[tuple[str, QuantumCircuit]] | None,
) -> None:
    if snapshots is not None:
        snapshots.append((label, circuit.copy(name=label)))


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


def _single_qubit_depolarizing_channel(probability: float) -> Kraus:
    return Kraus(
        [
            math.sqrt(1 - probability) * _PAULI_I,
            math.sqrt(probability / 3) * _PAULI_X,
            math.sqrt(probability / 3) * _PAULI_Y,
            math.sqrt(probability / 3) * _PAULI_Z,
        ]
    )


def _encoded_zero_vector() -> np.ndarray:
    vector = np.zeros(2**7, dtype=complex)
    for word in _steane_dual_codewords():
        vector[_word_to_index(word)] = 1 / math.sqrt(8)
    return vector


def _encoded_one_vector() -> np.ndarray:
    zero = _encoded_zero_vector()
    one = np.zeros_like(zero)
    for index, amplitude in enumerate(zero):
        one[index ^ _ALL_DATA_MASK] = amplitude
    return one


def _steane_dual_codewords() -> tuple[tuple[int, ...], ...]:
    words: set[tuple[int, ...]] = set()
    rows = STEANE_X_STABILIZERS
    for mask in range(2 ** len(rows)):
        word = [0] * 7
        for row_index, row in enumerate(rows):
            if (mask >> row_index) & 1:
                for bit_index in row:
                    word[bit_index] ^= 1
        words.add(tuple(word))
    return tuple(sorted(words))


def _word_to_index(word: tuple[int, ...]) -> int:
    index = 0
    for bit_index, bit in enumerate(word):
        index |= bit << bit_index
    return index


def _logical_pauli_matrix(axis: str) -> np.ndarray:
    single = {"X": _PAULI_X, "Y": _PAULI_Y, "Z": _PAULI_Z}[axis]
    matrix = np.array([[1]], dtype=complex)
    for _ in range(7):
        matrix = np.kron(single, matrix)
    return matrix


def _hadamard_all_matrix(num_qubits: int) -> np.ndarray:
    matrix = np.array([[1]], dtype=complex)
    for _ in range(num_qubits):
        matrix = np.kron(_HADAMARD, matrix)
    return matrix


def _identity_matrix(num_qubits: int) -> np.ndarray:
    return np.eye(2**num_qubits, dtype=complex)


def _validate_probability(probability: float, name: str) -> None:
    if not 0 <= probability <= 1:
        raise ValueError(f"{name} must be between 0 and 1")


def _reset(circuit: QuantumCircuit, q: QuantumRegister, qubits: tuple[int, ...]) -> None:
    for index in qubits:
        circuit.reset(q[index])


def _append_native_ops(
    circuit: QuantumCircuit,
    q: QuantumRegister,
    operations: tuple[_NativeOp, ...],
) -> None:
    for name, qubits, params in operations:
        if name == "h":
            circuit.h(q[qubits[0]])
        elif name == "cx":
            circuit.cx(q[qubits[0]], q[qubits[1]])
        elif name == "ry":
            circuit.ry(params[0], q[qubits[0]])
        else:
            raise ValueError(f"unsupported native op: {name}")


def _measure(
    circuit: QuantumCircuit,
    q: QuantumRegister,
    c: ClassicalRegister,
    qubits: tuple[int, ...],
) -> None:
    for bit_index, qubit_index in enumerate(qubits):
        circuit.measure(q[qubit_index], c[bit_index])
