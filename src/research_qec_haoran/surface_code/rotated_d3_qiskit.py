"""Qiskit helpers for the rotated distance-3 surface-code patch.

This module is split into two layers:

1. Circuit-building primitives for state preparation and stabilizer rounds.
2. Exact-state verification helpers used by tests and notebooks.

The public circuit API is intentionally small and compositional:

- prepare data qubits in ``|0>`` or ``|+>``
- measure one parallel Z-check round
- measure one parallel X-check round
- compose those pieces into a logical-``|+_L>`` initializer
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector

from research_qec_haoran.surface_code.rotated_d3_layout_qiskit import (
    ROTATED_D3_LOGICAL_X,
    ROTATED_D3_LOGICAL_Z,
    ROTATED_D3_SURFACE_ORDER,
    ROTATED_D3_X_STABILIZERS,
    ROTATED_D3_Z_STABILIZERS,
    rotated_d3_surface_label_map,
)
from research_qec_haoran.utils.normalize import normalize_bit

ROTATED_D3_INIT_Z_LABELS: tuple[str, ...] = tuple(
    f"init_{name.lower()}" for name in ROTATED_D3_Z_STABILIZERS
)

ROTATED_D3_Z_STABILIZER_SCHEDULE: dict[str, tuple[str, ...]] = {
    "Z_abde": ("a", "b", "d", "e"),
    "Z_cf": ("", "c", "", "f"),
    "Z_dg": ("", "d", "", "g"),
    "Z_efhi": ("e", "f", "h", "i"),
}

ROTATED_D3_X_STABILIZER_SCHEDULE: dict[str, tuple[str, ...]] = {
    "X_ab": ("", "a", "b", ""),
    "X_bcef": ("b", "c", "e", "f"),
    "X_degh": ("d", "e", "g", "h"),
    "X_hi": ("h", "i", "", ""),
}


# ---------------------------------------------------------------------------
# Circuit-building primitives
# ---------------------------------------------------------------------------



def _validate_round_width(
    *,
    ancillas: Sequence[object],
    classical_bits: Sequence[object],
    expected: int,
    label: str,
) -> None:
    if len(ancillas) != expected:
        raise ValueError(f"{label} ancillas must contain {expected} qubits.")
    if len(classical_bits) != expected:
        raise ValueError(f"{label} classical bits must contain {expected} bits.")


def _append_parallel_stabilizer_measurements(
    circuit: QuantumCircuit,
    *,
    label_to_qubit: Mapping[str, object],
    ancillas: Sequence[object],
    classical_bits: Sequence[object],
    schedules: Mapping[str, tuple[str, ...]],
    basis: str,
    prepare_ancillas: bool,
    add_barriers: bool,
    barrier_label: str | None,
) -> None:
    ancilla_list = list(ancillas)
    classical_list = list(classical_bits)
    stabilizer_names = tuple(schedules)
    _validate_round_width(
        ancillas=ancilla_list,
        classical_bits=classical_list,
        expected=len(stabilizer_names),
        label=f"{basis}-check",
    )

    if prepare_ancillas:
        append_prepare_zero_state(circuit, qubits=ancilla_list, add_barriers=False)

    if basis == "X":
        for ancilla in ancilla_list:
            circuit.h(ancilla)

    num_steps = len(next(iter(schedules.values())))
    for step in range(num_steps):
        for ancilla, stabilizer_name in zip(ancilla_list, stabilizer_names, strict=False):
            data_label = schedules[stabilizer_name][step]
            if not data_label:
                continue
            if basis == "Z":
                circuit.cx(label_to_qubit[data_label], ancilla)
            elif basis == "X":
                circuit.cx(ancilla, label_to_qubit[data_label])
            else:
                raise ValueError(f"unsupported stabilizer basis {basis!r}")

    if basis == "X":
        for ancilla in ancilla_list:
            circuit.h(ancilla)

    for ancilla, classical_bit in zip(ancilla_list, classical_list, strict=False):
        circuit.measure(ancilla, classical_bit)

    if add_barriers:
        circuit.barrier(label=barrier_label or f"parallel {basis} stabilizer round")



def append_prepare_zero_state(
    circuit: QuantumCircuit,
    *,
    qubits: Sequence[object],
    add_barriers: bool = False,
    barrier_label: str | None = None,
) -> None:
    """Reset a collection of qubits to ``|0>``."""
    qubit_list = list(qubits)
    for qubit in qubit_list:
        circuit.reset(qubit)
    if add_barriers:
        circuit.barrier(label=barrier_label or "prepare |0>")


def append_prepare_plus_state(
    circuit: QuantumCircuit,
    *,
    qubits: Sequence[object],
    add_barriers: bool = False,
    barrier_label: str | None = None,
) -> None:
    """Prepare a collection of qubits in ``|+>``."""
    qubit_list = list(qubits)
    for qubit in qubit_list:
        circuit.h(qubit)
    if add_barriers:
        circuit.barrier(label=barrier_label or "prepare |+>")


def append_rotated_d3_z_stabilizer_measurements(
    circuit: QuantumCircuit,
    *,
    surface_qubits: Sequence[object],
    z_ancillas: Sequence[object],
    classical_bits: Sequence[object],
    round_label: str,
    prepare_ancillas: bool = False,
    add_barriers: bool = True,
) -> None:
    """Append one parallel rotated-``d=3`` Z-check round."""
    _append_parallel_stabilizer_measurements(
        circuit,
        label_to_qubit=rotated_d3_surface_label_map(surface_qubits),
        ancillas=z_ancillas,
        classical_bits=classical_bits,
        schedules=ROTATED_D3_Z_STABILIZER_SCHEDULE,
        basis="Z",
        prepare_ancillas=prepare_ancillas,
        add_barriers=add_barriers,
        barrier_label=f"{round_label} parallel Z checks",
    )


def append_rotated_d3_x_stabilizer_measurements(
    circuit: QuantumCircuit,
    *,
    surface_qubits: Sequence[object],
    x_ancillas: Sequence[object],
    classical_bits: Sequence[object],
    round_label: str,
    prepare_ancillas: bool = False,
    add_barriers: bool = True,
) -> None:
    """Append one parallel rotated-``d=3`` X-check round."""
    _append_parallel_stabilizer_measurements(
        circuit,
        label_to_qubit=rotated_d3_surface_label_map(surface_qubits),
        ancillas=x_ancillas,
        classical_bits=classical_bits,
        schedules=ROTATED_D3_X_STABILIZER_SCHEDULE,
        basis="X",
        prepare_ancillas=prepare_ancillas,
        add_barriers=add_barriers,
        barrier_label=f"{round_label} parallel X checks",
    )


def append_surface_d3_stabilizer_round(
    circuit: QuantumCircuit,
    *,
    surface_qubits: Sequence[object],
    z_ancillas: Sequence[object],
    x_ancillas: Sequence[object],
    z_bits: Sequence[object],
    x_bits: Sequence[object],
    round_label: str,
    prepare_z_ancillas: bool = False,
    prepare_x_ancillas: bool = False,
    add_barriers: bool = True,
) -> None:
    """Append one full rotated-``d=3`` stabilizer round."""
    append_rotated_d3_z_stabilizer_measurements(
        circuit,
        surface_qubits=surface_qubits,
        z_ancillas=z_ancillas,
        classical_bits=z_bits,
        round_label=f"{round_label} Z",
        prepare_ancillas=prepare_z_ancillas,
        add_barriers=add_barriers,
    )
    append_rotated_d3_x_stabilizer_measurements(
        circuit,
        surface_qubits=surface_qubits,
        x_ancillas=x_ancillas,
        classical_bits=x_bits,
        round_label=f"{round_label} X",
        prepare_ancillas=prepare_x_ancillas,
        add_barriers=add_barriers,
    )


# ---------------------------------------------------------------------------
# Verification helpers (exact-state analysis for tests and notebooks only)
# ---------------------------------------------------------------------------


def rotated_d3_z_syndrome_bits_from_measurements(
    measurements: Mapping[str, int | bool],
) -> dict[str, int]:
    """Parse a labeled measurement dict into the four Z-stabilizer bits.

    **Verification only** — not used in hardware-style circuit construction.

    Accepts keys in either init form (``"init_z_abde"``) or round form
    (``"Z_abde"``).  Returns ``{"Z_abde": 0|1, "Z_cf": …, "Z_dg": …,
    "Z_efhi": …}``.  Used by verification helpers and the notebook to
    convert raw simulator output into a canonical syndrome dict.
    """
    syndrome: dict[str, int] = {}
    for name in ROTATED_D3_Z_STABILIZERS:
        init_label = f"init_{name.lower()}"
        if init_label in measurements:
            bit = measurements[init_label]
        elif name in measurements:
            bit = measurements[name]
        else:
            raise KeyError(f"missing measurement for {name!r} / {init_label!r}")
        syndrome[name] = normalize_bit(bit)
    return syndrome


def build_rotated_d3_pauli_operator(
    pauli_char: str,
    support: Sequence[str],
) -> SparsePauliOp:
    """Build a 9-qubit ``SparsePauliOp`` with *pauli_char* on *support* qubits and I elsewhere.

    **Verification only** — not used in hardware-style circuit construction.

    Used by the verification helpers to construct stabilizer and logical
    operators for exact expectation-value checks, e.g.
    ``build_rotated_d3_pauli_operator("Z", ("a", "b", "d", "e"))`` gives
    the Z_abde stabilizer operator.
    """
    label = ["I"] * len(ROTATED_D3_SURFACE_ORDER)
    for qubit_label in support:
        label[ROTATED_D3_SURFACE_ORDER.index(qubit_label)] = pauli_char
    return SparsePauliOp("".join(reversed(label)))


def build_rotated_d3_logical_plus_statevector(
    z_syndrome_bits: Mapping[str, int | bool] | None = None,
) -> Statevector:
    """Return the exact logical-``|+_L>`` branch for a chosen Z-syndrome.

    **Verification only** — for exact verification and notebook analysis, not for
    building hardware-style syndrome-extraction circuits.
    """
    state = Statevector.from_label("+" * len(ROTATED_D3_SURFACE_ORDER))
    identity = np.eye(2 ** len(ROTATED_D3_SURFACE_ORDER), dtype=complex)
    syndrome = {name: 0 for name in ROTATED_D3_Z_STABILIZERS}
    if z_syndrome_bits is not None:
        syndrome.update(rotated_d3_z_syndrome_bits_from_measurements(z_syndrome_bits))

    for name, support in ROTATED_D3_Z_STABILIZERS.items():
        operator = build_rotated_d3_pauli_operator("Z", support).to_matrix()
        sign = -1 if syndrome[name] else 1
        projected = ((identity + sign * operator) @ state.data) / 2
        state = Statevector(projected / np.linalg.norm(projected))

    return state


def rotated_d3_logical_plus_expectations(
    z_syndrome_bits: Mapping[str, int | bool] | None = None,
) -> dict[str, float]:
    """Return exact stabilizer and logical expectations for a chosen branch.

    **Verification only** — not used in hardware-style circuit construction.
    """
    syndrome = {name: 0 for name in ROTATED_D3_Z_STABILIZERS}
    if z_syndrome_bits is not None:
        syndrome.update(rotated_d3_z_syndrome_bits_from_measurements(z_syndrome_bits))

    state = build_rotated_d3_logical_plus_statevector(syndrome)
    expectations: dict[str, float] = {}

    for name, support in ROTATED_D3_Z_STABILIZERS.items():
        expectations[name] = state.expectation_value(
            build_rotated_d3_pauli_operator("Z", support)
        ).real
    for name, support in ROTATED_D3_X_STABILIZERS.items():
        expectations[name] = state.expectation_value(
            build_rotated_d3_pauli_operator("X", support)
        ).real

    expectations["logical_X"] = state.expectation_value(
        build_rotated_d3_pauli_operator("X", ROTATED_D3_LOGICAL_X)
    ).real
    expectations["logical_Z"] = state.expectation_value(
        build_rotated_d3_pauli_operator("Z", ROTATED_D3_LOGICAL_Z)
    ).real
    return expectations
