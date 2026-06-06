"""Qiskit helpers for Steane-to-rotated-d3 transfer by lattice surgery.

This module builds a paper-labeled demonstrator for transferring a logical
state from the distance-3 color code / Steane code onto a rotated distance-3
surface-code patch.

Important scope note
--------------------
The paper's explicit lattice-surgery figure uses a rotated distance-5 target
patch with merged checks ``Z1a``, ``Z35bc``, and ``Zde``.  This repo currently
contains a modular rotated distance-3 implementation.  The circuit below is
therefore a distance-3 adaptation using the top-boundary checks available on
the smaller patch:

- the Steane / color-code data qubits keep the paper labels ``0..6``
- the rotated surface-code data qubits keep the paper-style labels ``a..i``
- the merged-boundary checks are ``Z1a`` and ``Z35bc``

This is a minimal logical-transfer circuit for notebook and unit-test use.  It
is not a full reproduction of the paper's repeated noisy rounds, decoder
confidence logic, or code-expansion stage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

from research_qec_haoran.decoding.pauli_frame import PauliFrame
from research_qec_haoran.steane_code.magic_state_cultivation import (
    InjectedStateName,
    append_hadamard_axis_projection_round,
    append_steane_state_injection_encoder,
)
from research_qec_haoran.surface_code.rotated_d3_layout_qiskit import (
    ROTATED_D3_LOGICAL_X,
    ROTATED_D3_LOGICAL_Z,
    ROTATED_D3_SURFACE_ORDER,
)
from research_qec_haoran.surface_code.rotated_d3_qiskit import (
    append_prepare_plus_state,
    append_rotated_d3_z_stabilizer_measurements,
)

SurfaceReadoutBasis = Literal["X", "Z"]

PAPER_STEANE_DATA_ORDER: tuple[str, ...] = ("0", "1", "2", "3", "4", "5", "6")
PAPER_STEANE_GEOMETRIC_ORDER: tuple[str, ...] = ("6", "4", "0", "2", "1", "5", "3")
# Under the repo's current Steane encoder ordering, the weight-3 logical-X
# representative used in the transfer demonstrator is ``X016``.
PAPER_STEANE_LOGICAL_X: tuple[str, ...] = ("0", "1", "6")
PAPER_STEANE_LOGICAL_Z: tuple[str, ...] = ("1", "5", "3")
ROTATED_D3_TOP_BOUNDARY: tuple[str, ...] = ("a", "b", "c")

# Map the repo's 11-site cultivation patch labels onto the paper's 7 color-code
# data labels.

# Taken from magic_state_cultivation.py:
# The physical sites are labeled as
#
#               0
#
#           1       2
#
#       3       4       5
#
#           6       7
#
#       8       9       10
#
# The seven Steane/color-code data sites are
#
#     DATA_SITES = (0, 3, 4, 5, 8, 9, 10)
#
# and the four helper/measurement sites are
#
#     HELPER_SITES = (1, 2, 6, 7).

PATCH_SITE_TO_PAPER_DATA_LABEL: dict[int, str] = {
    0: "6",
    3: "4",
    4: "0",
    5: "2",
    8: "1",
    9: "5",
    10: "3",
}
PAPER_DATA_LABEL_TO_PATCH_SITE: dict[str, int] = {
    label: site for site, label in PATCH_SITE_TO_PAPER_DATA_LABEL.items()
}

STEANE_HELPER_PATCH_LABELS: tuple[str, ...] = ("h1", "h2", "h6", "h7")
STEANE_HELPER_LABEL_TO_PATCH_SITE: dict[str, int] = {
    "h1": 1,
    "h2": 2,
    "h6": 6,
    "h7": 7,
}

SURFACE_Z_ANCILLA_LABELS: tuple[str, ...] = ("z_abde", "z_cf", "z_dg", "z_efhi")
MERGED_BOUNDARY_CHECKS: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
    ("Z1a", ("1",), ("a",)),
    ("Z35bc", ("3", "5"), ("b", "c")),
)


@dataclass(frozen=True)
class PaperLabeledRegisters:
    """Paper-labeled qubit registers used by the transfer circuit."""

    steane_data: dict[str, QuantumRegister]
    steane_helpers: dict[str, QuantumRegister]
    surface_data: dict[str, QuantumRegister]
    surface_z_ancillas: dict[str, QuantumRegister]
    merge_ancillas: dict[str, QuantumRegister]


def build_paper_labeled_registers() -> PaperLabeledRegisters:
    """Create one-qubit Qiskit registers with paper-style labels."""
    steane_data = {
        label: QuantumRegister(1, label)
        for label in PAPER_STEANE_DATA_ORDER
    }
    steane_helpers = {
        label: QuantumRegister(1, label)
        for label in STEANE_HELPER_PATCH_LABELS
    }
    surface_data = {
        label: QuantumRegister(1, label)
        for label in ROTATED_D3_SURFACE_ORDER
    }
    surface_z_ancillas = {
        label: QuantumRegister(1, label)
        for label in SURFACE_Z_ANCILLA_LABELS
    }
    merge_ancillas = {
        f"zz_{check_name[1:].lower()}": QuantumRegister(1, f"zz_{check_name[1:].lower()}")
        for check_name, _, _ in MERGED_BOUNDARY_CHECKS
    }
    return PaperLabeledRegisters(
        steane_data=steane_data,
        steane_helpers=steane_helpers,
        surface_data=surface_data,
        surface_z_ancillas=surface_z_ancillas,
        merge_ancillas=merge_ancillas,
    )


def _paper_labeled_patch_qubits(registers: PaperLabeledRegisters) -> list:
    """Return the 11-site cultivation patch qubits in patch-label order."""
    patch_qubits: list[object] = [None] * 11
    for site, label in PATCH_SITE_TO_PAPER_DATA_LABEL.items():
        patch_qubits[site] = registers.steane_data[label][0]
    for label, site in STEANE_HELPER_LABEL_TO_PATCH_SITE.items():
        patch_qubits[site] = registers.steane_helpers[label][0]
    return patch_qubits


def _surface_data_qubits(registers: PaperLabeledRegisters) -> list:
    """Return the rotated-d3 surface data qubits in ``a..i`` order."""
    return [registers.surface_data[label][0] for label in ROTATED_D3_SURFACE_ORDER]


def _parse_register_chunk(chunk: str, labels: tuple[str, ...]) -> dict[str, int]:
    bits = chunk.replace(" ", "")
    return {
        label: int(bit)
        for label, bit in zip(labels, reversed(bits), strict=False)
    }


def append_measure_z_parity(
    circuit: QuantumCircuit,
    *,
    data_qubits,
    ancilla,
    classical_bit,
) -> None:
    """Measure the Z-basis parity of data qubits with a fresh ancilla."""
    circuit.reset(ancilla)
    for qubit in data_qubits:
        circuit.cx(qubit, ancilla)
    circuit.measure(ancilla, classical_bit)


def append_boundary_zz_checks(
    circuit: QuantumCircuit,
    *,
    registers: PaperLabeledRegisters,
    classical_bits: list,
    merged_rounds: int = 1,
    add_barriers: bool = True,
) -> None:
    """Append repeated merged-boundary ``ZZ`` checks for the d=3 adaptation."""
    expected_bits = merged_rounds * len(MERGED_BOUNDARY_CHECKS)
    if len(classical_bits) != expected_bits:
        raise ValueError(
            f"classical_bits must contain {expected_bits} bits for {merged_rounds} merged rounds."
        )

    for round_index in range(merged_rounds):
        for check_index, (check_name, steane_labels, surface_labels) in enumerate(
            MERGED_BOUNDARY_CHECKS
        ):
            bit_index = round_index * len(MERGED_BOUNDARY_CHECKS) + check_index
            ancilla_name = f"zz_{check_name[1:].lower()}"
            data_qubits = [
                registers.steane_data[label][0]
                for label in steane_labels
            ] + [
                registers.surface_data[label][0]
                for label in surface_labels
            ]
            append_measure_z_parity(
                circuit,
                data_qubits=data_qubits,
                ancilla=registers.merge_ancillas[ancilla_name][0],
                classical_bit=classical_bits[bit_index],
            )
        if add_barriers:
            circuit.barrier(label=f"merged ZZ round {round_index + 1}")


def append_measure_steane_data_in_x_basis(
    circuit: QuantumCircuit,
    *,
    registers: PaperLabeledRegisters,
    classical_bits: list,
    add_barriers: bool = True,
) -> None:
    """Demolish the Steane block by measuring all seven data qubits in ``X``."""
    if len(classical_bits) != len(PAPER_STEANE_DATA_ORDER):
        raise ValueError("classical_bits must contain seven Steane-data bits.")

    for label, classical_bit in zip(PAPER_STEANE_DATA_ORDER, classical_bits, strict=False):
        qubit = registers.steane_data[label][0]
        circuit.h(qubit)
        circuit.measure(qubit, classical_bit)

    if add_barriers:
        circuit.barrier(label="measure Steane data in X")


def append_surface_logical_readout(
    circuit: QuantumCircuit,
    *,
    registers: PaperLabeledRegisters,
    classical_bits: list,
    basis: SurfaceReadoutBasis,
    add_barriers: bool = True,
) -> None:
    """Measure all rotated-d3 data qubits in ``basis`` for logical readout."""
    if len(classical_bits) != len(ROTATED_D3_SURFACE_ORDER):
        raise ValueError("classical_bits must contain nine surface-data bits.")

    normalized_basis = basis.upper()
    if normalized_basis not in {"X", "Z"}:
        raise ValueError("basis must be 'X' or 'Z'.")

    for label, classical_bit in zip(ROTATED_D3_SURFACE_ORDER, classical_bits, strict=False):
        qubit = registers.surface_data[label][0]
        if normalized_basis == "X":
            circuit.h(qubit)
        circuit.measure(qubit, classical_bit)

    if add_barriers:
        circuit.barrier(label=f"surface logical {normalized_basis} readout")


def build_steane_to_rotated_d3_lattice_surgery_circuit(
    *,
    injected_state: InjectedStateName | tuple[complex, complex] = "H",
    include_cultivation: bool = True,
    merged_rounds: int = 1,
    surface_readout_basis: SurfaceReadoutBasis | None = None,
    add_barriers: bool = True,
) -> QuantumCircuit:
    """Build the paper-labeled Steane-to-rotated-d3 transfer circuit.

    The returned circuit uses one-qubit Qiskit registers so the wire labels
    match the paper notation directly:

    - Steane / color-code data wires: ``0..6``
    - rotated surface-code data wires: ``a..i``

    Parameters
    ----------
    injected_state
        Physical state injected into the Steane encoder.
    include_cultivation
        If ``True``, append the repo's existing ``H_L`` cultivation round
        before transfer.  If ``False``, transfer the encoded Steane state
        directly; this is useful for teleportation sanity checks on logical
        basis states.
    merged_rounds
        Number of repeated merged-boundary ``ZZ`` rounds.  The paper uses
        repeated rounds for fault detection; the default here is one noiseless
        round to keep the d=3 adaptation compact.
    surface_readout_basis
        Optional final surface readout basis.  If given, all nine surface data
        qubits are measured in that basis at the end of the circuit.
    add_barriers
        Whether to insert conceptual barriers.
    """
    if merged_rounds < 1:
        raise ValueError("merged_rounds must be at least 1.")

    registers = build_paper_labeled_registers()
    cultivation_bits = ClassicalRegister(5, "cult") if include_cultivation else None
    init_z_bits = ClassicalRegister(4, "init_z")
    merged_zz_bits = ClassicalRegister(len(MERGED_BOUNDARY_CHECKS) * merged_rounds, "ls_zz")
    steane_x_bits = ClassicalRegister(7, "steane_x")
    surface_readout_bits = (
        ClassicalRegister(9, f"surface_{surface_readout_basis.lower()}")
        if surface_readout_basis is not None
        else None
    )

    qubit_registers = (
        list(registers.steane_data.values())
        + list(registers.steane_helpers.values())
        + list(registers.surface_data.values())
        + list(registers.surface_z_ancillas.values())
        + list(registers.merge_ancillas.values())
    )
    classical_registers = []
    if cultivation_bits is not None:
        classical_registers.append(cultivation_bits)
    classical_registers.extend([init_z_bits, merged_zz_bits, steane_x_bits])
    if surface_readout_bits is not None:
        classical_registers.append(surface_readout_bits)

    circuit = QuantumCircuit(*qubit_registers, *classical_registers)
    patch_qubits = _paper_labeled_patch_qubits(registers)

    append_steane_state_injection_encoder(
        circuit,
        patch_qubits=patch_qubits,
        injected_state=injected_state,
    )

    if include_cultivation:
        append_hadamard_axis_projection_round(
            circuit,
            patch_qubits=patch_qubits,
            classical_bits=list(cultivation_bits),
            add_barriers=add_barriers,
            skip_cx_1_4=True,
        )

    surface_qubits = _surface_data_qubits(registers)
    append_prepare_plus_state(
        circuit,
        qubits=surface_qubits,
        add_barriers=add_barriers,
        barrier_label="prepare rotated d3 |+> product state",
    )
    append_rotated_d3_z_stabilizer_measurements(
        circuit,
        surface_qubits=surface_qubits,
        z_ancillas=[registers.surface_z_ancillas[label][0] for label in SURFACE_Z_ANCILLA_LABELS],
        classical_bits=list(init_z_bits),
        round_label="surface init",
        prepare_ancillas=False,
        add_barriers=add_barriers,
    )

    append_boundary_zz_checks(
        circuit,
        registers=registers,
        classical_bits=list(merged_zz_bits),
        merged_rounds=merged_rounds,
        add_barriers=add_barriers,
    )
    append_measure_steane_data_in_x_basis(
        circuit,
        registers=registers,
        classical_bits=list(steane_x_bits),
        add_barriers=add_barriers,
    )

    if surface_readout_bits is not None:
        append_surface_logical_readout(
            circuit,
            registers=registers,
            classical_bits=list(surface_readout_bits),
            basis=surface_readout_basis,
            add_barriers=add_barriers,
        )

    return circuit


def parse_transfer_measurements(
    bitstring: str,
    *,
    include_cultivation: bool = True,
    merged_rounds: int = 1,
    surface_readout_basis: SurfaceReadoutBasis | None = None,
) -> dict[str, object]:
    """Parse a Qiskit count key from the transfer circuit into labeled dicts.

    Qiskit returns measurement results as a single space-separated bitstring
    with one chunk per classical register, ordered from **last register to
    first** (i.e. the rightmost chunk is the first register added to the
    circuit).  Within each chunk, bit ordering is big-endian (MSB first).

    This function splits that raw string into human-readable dicts keyed by
    the paper-style labels used in the transfer circuit.

    Parameters
    ----------
    bitstring
        A Qiskit counts key, e.g. ``"0000000 000 0000 00000"``.
    include_cultivation
        Whether the circuit included the cultivation round (adds a 5-bit
        ``cult`` register).
    merged_rounds
        Number of repeated merged-boundary ZZ rounds (each adds 2 bits).
    surface_readout_basis
        If set, the circuit included a final 9-bit surface readout register.

    Returns
    -------
    dict with the following keys (all present unless noted):

    ``"cultivation"`` *(only if include_cultivation)*
        Dict mapping ``{"root_x", "helper_h1", ..., "helper_h7"}`` → 0 or 1.
        These are the 5 postselection bits from the H_L cultivation round.

    ``"surface_init_z"``
        Dict mapping ``{"Z_abde", "Z_cf", "Z_dg", "Z_efhi"}`` → 0 or 1.
        The 4 Z-stabilizer outcomes from the surface-code initialization
        round (projects the |+⟩ product state into the code space).

    ``"merged_zz_rounds"``
        List of dicts (one per merged round), each mapping
        ``{"Z1a", "Z35bc"}`` → 0 or 1.
        These are the boundary ZZ parity checks between the Steane and
        surface codes that implement the lattice-surgery teleportation.

    ``"steane_x"``
        Dict mapping ``{"0", "1", ..., "6"}`` → 0 or 1.
        The X-basis demolition measurement of all 7 Steane data qubits.

    ``"surface_readout"`` *(only if surface_readout_basis is set)*
        Dict mapping ``{"a", "b", ..., "i"}`` → 0 or 1.
        The final logical-readout measurement of the 9 surface data qubits.
    """
    parts = bitstring.split()
    expected_parts = 3 + int(include_cultivation) + int(surface_readout_basis is not None)
    if len(parts) != expected_parts:
        raise ValueError(
            f"Expected {expected_parts} space-separated register chunks, got {len(parts)}."
        )

    cursor = 0
    parsed: dict[str, object] = {}
    if surface_readout_basis is not None:
        parsed["surface_readout"] = _parse_register_chunk(parts[cursor], ROTATED_D3_SURFACE_ORDER)
        cursor += 1

    parsed["steane_x"] = _parse_register_chunk(parts[cursor], PAPER_STEANE_DATA_ORDER)
    cursor += 1

    merged_bits = _parse_register_chunk(
        parts[cursor],
        tuple(
            f"round{round_index + 1}_{check_name}"
            for round_index in range(merged_rounds)
            for check_name, _, _ in MERGED_BOUNDARY_CHECKS
        ),
    )
    cursor += 1
    parsed["merged_zz_rounds"] = [
        {
            check_name: merged_bits[
                f"round{round_index + 1}_{check_name}"
            ]
            for check_name, _, _ in MERGED_BOUNDARY_CHECKS
        }
        for round_index in range(merged_rounds)
    ]

    parsed["surface_init_z"] = _parse_register_chunk(
        parts[cursor],
        ("Z_abde", "Z_cf", "Z_dg", "Z_efhi"),
    )
    cursor += 1

    if include_cultivation:
        parsed["cultivation"] = _parse_register_chunk(
            parts[cursor],
            ("root_x", "helper_h1", "helper_h2", "helper_h6", "helper_h7"),
        )

    return parsed


def merged_boundary_logical_zz_outcome(
    measurements: dict[str, object],
) -> int:
    """Return the logical ``Z_L Z_L`` parity inferred from the merged checks."""
    round_parities = [
        sum(round_bits.values()) % 2
        for round_bits in measurements["merged_zz_rounds"]
    ]
    if len(set(round_parities)) != 1:
        raise ValueError(
            "Merged-boundary rounds disagree; this noiseless demonstrator expects a stable parity."
        )
    return round_parities[0]


def steane_logical_x_outcome(measurements: dict[str, object]) -> int:
    """Return the Steane logical-``X`` outcome from the chosen weight-3 support."""
    steane_x = measurements["steane_x"]
    return sum(steane_x[label] for label in PAPER_STEANE_LOGICAL_X) % 2


def transfer_pauli_frame(measurements: dict[str, object]) -> PauliFrame:
    """Build the target Pauli frame from the transfer measurement outcomes."""
    frame = PauliFrame()
    frame.register("surface")
    frame.update_x("surface", bool(merged_boundary_logical_zz_outcome(measurements)))
    frame.update_z("surface", bool(steane_logical_x_outcome(measurements)))
    return frame


def raw_surface_logical_readout(
    measurements: dict[str, object],
    *,
    basis: SurfaceReadoutBasis,
) -> int:
    """Return the uncorrected surface logical readout from physical data bits."""
    if "surface_readout" not in measurements:
        raise ValueError("measurements do not include a surface logical readout.")

    support = ROTATED_D3_LOGICAL_X if basis.upper() == "X" else ROTATED_D3_LOGICAL_Z
    return sum(measurements["surface_readout"][label] for label in support) % 2


def corrected_surface_logical_readout(
    measurements: dict[str, object],
    *,
    basis: SurfaceReadoutBasis,
) -> int:
    """Return the surface logical readout corrected by the transfer Pauli frame."""
    frame = transfer_pauli_frame(measurements)
    raw_bit = raw_surface_logical_readout(measurements, basis=basis)
    if basis.upper() == "X":
        return frame.interpret_x_readout("surface", raw_bit)
    return frame.interpret_z_readout("surface", raw_bit)


def cultivation_succeeds(measurements: dict[str, object]) -> bool:
    """Return whether the cultivation bits are all zero."""
    cultivation = measurements.get("cultivation")
    if cultivation is None:
        return True
    return sum(cultivation.values()) == 0
