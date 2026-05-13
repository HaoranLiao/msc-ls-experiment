from __future__ import annotations

from collections.abc import Sequence

from qiskit import QuantumCircuit, QuantumRegister

from research_qec_haoran.steane_code.magic_state_cultivation import (
    STEANE_TO_11_SITE_ORDER,
    append_steane_state_injection_encoder,
)


# This permutation is chosen so that the Steane encoder used in this repo,
# when written in paper order, reproduces the paper's color-code plaquette
# supports 0145, 0235, and 0246 and includes the logical X line 146.
GOOGLE_TO_PAPER_COLOR: tuple[int, ...] = (1, 4, 2, 3, 0, 5, 6)
PAPER_TO_GOOGLE_COLOR: tuple[int, ...] = (4, 0, 2, 3, 1, 5, 6)

PAPER_COLOR_ORDER_AS_PATCH_SITES: tuple[int, ...] = tuple(
    STEANE_TO_11_SITE_ORDER[google_index]
    for google_index in PAPER_TO_GOOGLE_COLOR
)

PAPER_COLOR_COORDS: dict[int, tuple[int, int]] = {
    5: (0, 0),
    4: (-1, 1),
    2: (1, 1),
    0: (0, 2),
    1: (-2, 3),
    6: (0, 3),
    3: (2, 3),
}

PAPER_COLOR_X_STABILIZERS: dict[str, tuple[int, ...]] = {
    "X0145": (0, 1, 4, 5),
    "X0235": (0, 2, 3, 5),
    "X0246": (0, 2, 4, 6),
}

PAPER_COLOR_Z_STABILIZERS: dict[str, tuple[int, ...]] = {
    "Z0145": (0, 1, 4, 5),
    "Z0235": (0, 2, 3, 5),
    "Z0246": (0, 2, 4, 6),
}

PAPER_COLOR_LOGICAL_X: tuple[int, ...] = (1, 4, 6)


def google_order_qubits_from_paper_order(paper_order_qubits: Sequence) -> list:
    """Return color-code qubits in the Steane encoder's Google order."""
    if len(paper_order_qubits) != 7:
        raise ValueError("paper_order_qubits must contain exactly 7 qubits.")
    return [paper_order_qubits[paper_index] for paper_index in GOOGLE_TO_PAPER_COLOR]


def build_paper_ordered_cultivated_color_state_circuit(
    injected_state: str = "H",
    *,
    add_barriers: bool = True,
) -> QuantumCircuit:
    """Build the cultivated Steane/color-code state in the paper's 0..6 order."""
    color = QuantumRegister(7, "color")
    circuit = QuantumCircuit(color, name="paper_ordered_HL")
    google_order_qubits = google_order_qubits_from_paper_order(list(color))
    append_steane_state_injection_encoder(
        circuit,
        steane_qubits=google_order_qubits,
        injected_state=injected_state,
    )
    if add_barriers:
        circuit.barrier(label="paper-ordered cultivated color state")
    return circuit
