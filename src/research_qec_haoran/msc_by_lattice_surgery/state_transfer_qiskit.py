"""Qiskit helpers for Steane-to-rotated-d3 transfer by lattice surgery.

This module builds a Hirano-labeled demonstrator for transferring a logical
state from the distance-3 color code / Steane code onto a rotated distance-3
surface-code patch.

Important scope note
--------------------
The paper's explicit lattice-surgery figure uses a rotated distance-5 target
patch with merged checks ``Z1a``, ``Z35bc``, and ``Zde``.  This repo currently
contains a modular rotated distance-3 implementation.  The circuit below is
therefore a distance-3 adaptation using the top-boundary checks available on
the smaller patch:

- the Steane / color-code data qubits keep the Hirano labels ``0..6``
- the rotated surface-code data qubits keep the Hirano-style labels ``a..i``
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
from research_qec_haoran.steane_code.magic_state_cultivation_qiskit import (
    append_hadamard_axis_projection_round,
)
from research_qec_haoran.steane_code.state_injection_qiskit import (
    InjectedStateName,
    append_patch_steane_state_injection_encoder,
)
from research_qec_haoran.surface_code.rotated_d3_qiskit import (
    DEFAULT_LOGICAL_X,
    DEFAULT_LOGICAL_Z,
    SURFACE_ORDER,
)
from research_qec_haoran.surface_code.rotated_d3_qiskit import (
    prepare_plus,
    append_z_stabilizer_measurements,
)

SurfaceReadoutBasis = Literal["X", "Z"]
SurfaceReadoutMode = Literal["data", "logical_pauli", "logical_pauli_indirect"]

HIRANO_STEANE_DATA_ORDER: tuple[str, ...] = ("0", "1", "2", "3", "4", "5", "6")
HIRANO_STEANE_GEOMETRIC_ORDER: tuple[str, ...] = ("6", "4", "0", "2", "1", "5", "3")
# Under the repo's current Steane encoder ordering, the default weight-3
# Steane logical representatives map to Hirano labels ``1, 5, 3``.
HIRANO_STEANE_LOGICAL_X: tuple[str, ...] = ("1", "5", "3")
HIRANO_STEANE_LOGICAL_Z: tuple[str, ...] = ("1", "5", "3")
TOP_BOUNDARY: tuple[str, ...] = ("a", "b", "c")
SURFACE_X_AB_LABELS: tuple[str, ...] = ("X_ab",)
SURFACE_INIT_Z_LABELS: tuple[str, ...] = ("Z_abde", "Z_cf", "Z_dg", "Z_efhi")
STEANE_X_RECONSTRUCTED_CHECKS: dict[str, tuple[str, ...]] = {
    "X0235": ("0", "2", "3", "5"),
    "X0246": ("0", "2", "4", "6"),
    "X0145": ("0", "1", "4", "5"),
}
LOCAL_STEANE_X_STABILIZER_CHECKS: dict[str, tuple[str, ...]] = {
    "X0246": ("0", "2", "4", "6"),
    "X0235": ("0", "2", "3", "5"),
}
LOCAL_X_AB_CORRELATION_CHECKS: dict[str, tuple[str, ...]] = {
    "X0145ab": ("0", "1", "4", "5", "X_ab"),
}

# Map the repo's 11-site cultivation patch labels onto Hirano's 7 color-code
# data labels.

# Taken from magic_state_cultivation_qiskit.py:
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

GOOGLE_SITE_TO_HIRANO_LABEL: dict[int, str] = {
    0: "6",
    3: "4",
    4: "0",
    5: "2",
    8: "1",
    9: "5",
    10: "3",
}
HIRANO_LABEL_TO_GOOGLE_SITE: dict[str, int] = {
    label: site for site, label in GOOGLE_SITE_TO_HIRANO_LABEL.items()
}

STEANE_HELPER_PATCH_LABELS: tuple[str, ...] = ("h1", "h2", "h6", "h7")
STEANE_HELPER_LABEL_TO_GOOGLE_SITE: dict[str, int] = {
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
class HiranoLabeledQRegs:
    """Paper-labeled qubit registers used by the transfer circuit."""

    steane_data: dict[str, QuantumRegister]
    steane_helpers: dict[str, QuantumRegister]
    surface_data: dict[str, QuantumRegister]
    surface_z_ancillas: dict[str, QuantumRegister]
    surface_x_ancillas: dict[str, QuantumRegister]
    merge_ancillas: dict[str, QuantumRegister]
    logical_measurement_ancillas: dict[str, QuantumRegister]


@dataclass(frozen=True)
class ClassicalRegisterSpec:
    """A human-readable description of one Qiskit classical register."""

    name: str
    labels: tuple[str, ...]


def build_hirano_labeled_qregs(
    *,
    include_noisy_verification: bool = False,
    surface_logical_measurement_basis: SurfaceReadoutBasis | None = None,
) -> HiranoLabeledQRegs:
    """Create one-qubit Qiskit registers with Hirano-style labels."""
    steane_data = {
        label: QuantumRegister(1, label)
        for label in HIRANO_STEANE_DATA_ORDER
    }
    steane_helpers = {
        label: QuantumRegister(1, label)
        for label in STEANE_HELPER_PATCH_LABELS
    }
    surface_data = {
        label: QuantumRegister(1, label)
        for label in SURFACE_ORDER
    }
    surface_z_ancillas = {
        label: QuantumRegister(1, label)
        for label in SURFACE_Z_ANCILLA_LABELS
    }
    surface_x_ancillas = (
        {"x_ab": QuantumRegister(1, "x_ab")}
        if include_noisy_verification
        else {}
    )
    merge_ancillas = {
        f"zz_{check_name[1:].lower()}": QuantumRegister(1, f"zz_{check_name[1:].lower()}")
        for check_name, _, _ in MERGED_BOUNDARY_CHECKS
    }
    logical_measurement_ancillas = {}
    if surface_logical_measurement_basis is not None:
        basis = surface_logical_measurement_basis.lower()
        logical_measurement_ancillas[f"logical_{basis}"] = QuantumRegister(1, f"logical_{basis}")
    return HiranoLabeledQRegs(
        steane_data=steane_data,
        steane_helpers=steane_helpers,
        surface_data=surface_data,
        surface_z_ancillas=surface_z_ancillas,
        surface_x_ancillas=surface_x_ancillas,
        merge_ancillas=merge_ancillas,
        logical_measurement_ancillas=logical_measurement_ancillas,
    )


def _hirano_labeled_patch_qubits(registers: HiranoLabeledQRegs) -> list:
    """Return the 11-site cultivation patch qubits in patch-label order."""
    patch_qubits: list[object] = [None] * 11
    for site, label in GOOGLE_SITE_TO_HIRANO_LABEL.items():
        patch_qubits[site] = registers.steane_data[label][0]
    for label, site in STEANE_HELPER_LABEL_TO_GOOGLE_SITE.items():
        patch_qubits[site] = registers.steane_helpers[label][0]
    return patch_qubits


def _surface_data_qubits(registers: HiranoLabeledQRegs) -> list:
    """Return the rotated-d3 surface data qubits in ``a..i`` order."""
    return [registers.surface_data[label][0] for label in SURFACE_ORDER]


def _parse_register_chunk(chunk: str, labels: tuple[str, ...]) -> dict[str, int]:
    bits = chunk.replace(" ", "")
    return {
        label: int(bit)
        for label, bit in zip(labels, reversed(bits), strict=False)
    }


def _parity(bits: dict[str, int], labels: tuple[str, ...]) -> int:
    return sum(bits[label] for label in labels) % 2


def _merged_zz_labels(merged_rounds: int) -> tuple[str, ...]:
    return tuple(
        f"round{round_index + 1}_{check_name}"
        for round_index in range(merged_rounds)
        for check_name, _, _ in MERGED_BOUNDARY_CHECKS
    )


def _surface_logical_readout_label(basis: SurfaceReadoutBasis) -> str:
    support = DEFAULT_LOGICAL_X if basis.upper() == "X" else DEFAULT_LOGICAL_Z
    return f"{basis.upper()}_{''.join(support)}"


def _classical_register_specs(
    *,
    include_cultivation: bool,
    merged_rounds: int,
    noisy: bool,
    surface_readout_basis: SurfaceReadoutBasis | None,
    surface_readout_mode: SurfaceReadoutMode,
) -> list[ClassicalRegisterSpec]:
    specs: list[ClassicalRegisterSpec] = []
    if include_cultivation:
        specs.append(
            ClassicalRegisterSpec(
                "cult",
                ("root_x", "helper_h1", "helper_h2", "helper_h6", "helper_h7"),
            )
        )
    specs.extend(
        [
            ClassicalRegisterSpec("init_z", SURFACE_INIT_Z_LABELS),
            ClassicalRegisterSpec("ls_zz", _merged_zz_labels(merged_rounds)),
            ClassicalRegisterSpec("steane_x", HIRANO_STEANE_DATA_ORDER),
        ]
    )
    if noisy:
        specs.append(ClassicalRegisterSpec("surface_x_recovery", SURFACE_X_AB_LABELS))
    if surface_readout_basis is not None:
        if surface_readout_mode == "data":
            specs.append(
                ClassicalRegisterSpec(
                    f"surface_{surface_readout_basis.lower()}",
                    SURFACE_ORDER,
                )
            )
        elif surface_readout_mode == "logical_pauli":
            support = (
                DEFAULT_LOGICAL_X
                if surface_readout_basis.upper() == "X"
                else DEFAULT_LOGICAL_Z
            )
            specs.append(
                ClassicalRegisterSpec(
                    f"surface_logical_{surface_readout_basis.lower()}",
                    support,
                )
            )
        elif surface_readout_mode == "logical_pauli_indirect":
            specs.append(
                ClassicalRegisterSpec(
                    f"surface_logical_{surface_readout_basis.lower()}",
                    (_surface_logical_readout_label(surface_readout_basis),),
                )
            )
        else:
            raise ValueError(
                "surface_readout_mode must be 'data', 'logical_pauli', "
                "or 'logical_pauli_indirect'."
            )
    return specs


def append_measure_z_parity(
    circuit: QuantumCircuit,
    *,
    data_qubits,
    ancilla,
    classical_bit,
) -> None:
    """Measure the Z-basis parity of data qubits with a fresh ancilla."""
    for qubit in data_qubits:
        circuit.cx(qubit, ancilla)
    circuit.measure(ancilla, classical_bit)


def append_measure_x_parity(
    circuit: QuantumCircuit,
    *,
    data_qubits,
    ancilla,
    classical_bit,
) -> None:
    """Measure the X-basis parity of data qubits with a fresh ancilla."""
    circuit.h(ancilla)
    for qubit in data_qubits:
        circuit.cx(ancilla, qubit)
    circuit.h(ancilla)
    circuit.measure(ancilla, classical_bit)


def append_boundary_zz_checks(
    circuit: QuantumCircuit,
    *,
    registers: HiranoLabeledQRegs,
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
    registers: HiranoLabeledQRegs,
    classical_bits: list,
    add_barriers: bool = True,
) -> None:
    """Demolish the Steane block by measuring all seven data qubits in ``X``."""
    if len(classical_bits) != len(HIRANO_STEANE_DATA_ORDER):
        raise ValueError("classical_bits must contain seven Steane-data bits.")

    for label, classical_bit in zip(HIRANO_STEANE_DATA_ORDER, classical_bits, strict=False):
        qubit = registers.steane_data[label][0]
        circuit.h(qubit)
        circuit.measure(qubit, classical_bit)

    if add_barriers:
        circuit.barrier(label="measure Steane data in X")


def append_surface_logical_readout(
    circuit: QuantumCircuit,
    *,
    registers: HiranoLabeledQRegs,
    classical_bits: list,
    basis: SurfaceReadoutBasis,
    add_barriers: bool = True,
) -> None:
    """Measure all rotated-d3 data qubits in ``basis`` for logical readout."""
    if len(classical_bits) != len(SURFACE_ORDER):
        raise ValueError("classical_bits must contain nine surface-data bits.")

    normalized_basis = basis.upper()
    if normalized_basis not in {"X", "Z"}:
        raise ValueError("basis must be 'X' or 'Z'.")

    for label, classical_bit in zip(SURFACE_ORDER, classical_bits, strict=False):
        qubit = registers.surface_data[label][0]
        if normalized_basis == "X":
            circuit.h(qubit)
        circuit.measure(qubit, classical_bit)

    if add_barriers:
        circuit.barrier(label=f"surface logical {normalized_basis} readout")


def append_surface_x_ab_recovery_measurement(
    circuit: QuantumCircuit,
    *,
    registers: HiranoLabeledQRegs,
    classical_bit,
    add_barriers: bool = True,
) -> None:
    """Recover the top-boundary ``X_ab`` stabilizer used by paper postselection."""
    if "x_ab" not in registers.surface_x_ancillas:
        raise ValueError("registers do not include the x_ab recovery ancilla.")
    append_measure_x_parity(
        circuit,
        data_qubits=[
            registers.surface_data["a"][0],
            registers.surface_data["b"][0],
        ],
        ancilla=registers.surface_x_ancillas["x_ab"][0],
        classical_bit=classical_bit,
    )
    if add_barriers:
        circuit.barrier(label="recover surface X_ab")


def append_surface_logical_pauli_measurement(
    circuit: QuantumCircuit,
    *,
    registers: HiranoLabeledQRegs,
    classical_bit,
    basis: SurfaceReadoutBasis,
    add_barriers: bool = True,
) -> None:
    """Measure the weight-3 rotated-d3 logical Pauli directly with an ancilla."""
    normalized_basis = basis.upper()
    if normalized_basis not in {"X", "Z"}:
        raise ValueError("basis must be 'X' or 'Z'.")

    ancilla_name = f"logical_{normalized_basis.lower()}"
    if ancilla_name not in registers.logical_measurement_ancillas:
        raise ValueError(f"registers do not include the {ancilla_name} measurement ancilla.")

    support = DEFAULT_LOGICAL_X if normalized_basis == "X" else DEFAULT_LOGICAL_Z
    data_qubits = [registers.surface_data[label][0] for label in support]
    if normalized_basis == "X":
        append_measure_x_parity(
            circuit,
            data_qubits=data_qubits,
            ancilla=registers.logical_measurement_ancillas[ancilla_name][0],
            classical_bit=classical_bit,
        )
    else:
        append_measure_z_parity(
            circuit,
            data_qubits=data_qubits,
            ancilla=registers.logical_measurement_ancillas[ancilla_name][0],
            classical_bit=classical_bit,
        )

    if add_barriers:
        circuit.barrier(label=f"measure logical {normalized_basis}_{''.join(support)}")


def append_surface_logical_pauli_destructive(
    circuit: QuantumCircuit,
    *,
    registers: HiranoLabeledQRegs,
    classical_bits: list,
    basis: SurfaceReadoutBasis,
    add_barriers: bool = True,
) -> None:
    """Destructively measure the weight-3 logical Pauli by measuring data qubits directly.

    Measures each qubit in the logical operator's support in the appropriate
    basis (X or Z) and records the individual outcomes.  The logical readout
    is the parity of the three bits (computed classically by the parser).

    This avoids allocating an ancilla qubit and does not use ``reset``, making
    it compatible with Aer's ``extended_stabilizer`` method.
    """
    normalized_basis = basis.upper()
    if normalized_basis not in {"X", "Z"}:
        raise ValueError("basis must be 'X' or 'Z'.")

    support = DEFAULT_LOGICAL_X if normalized_basis == "X" else DEFAULT_LOGICAL_Z
    if len(classical_bits) != len(support):
        raise ValueError(
            f"classical_bits must have {len(support)} entries for destructive measurement."
        )

    for label, classical_bit in zip(support, classical_bits, strict=False):
        qubit = registers.surface_data[label][0]
        if normalized_basis == "X":
            circuit.h(qubit)
        circuit.measure(qubit, classical_bit)

    if add_barriers:
        circuit.barrier(label=f"destructive logical {normalized_basis}_{''.join(support)}")


def build_steane_to_rotated_d3_lattice_surgery_circuit(
    *,
    injected_state: InjectedStateName | tuple[complex, complex] = "H",
    include_cultivation: bool = True,
    merged_rounds: int = 1,
    noisy: bool = False,
    surface_readout_basis: SurfaceReadoutBasis | None = None,
    surface_readout_mode: SurfaceReadoutMode = "data",
    add_barriers: bool = True,
) -> QuantumCircuit:
    """Build the Hirano-labeled Steane-to-rotated-d3 transfer circuit.

    The returned circuit uses one-qubit Qiskit registers so the wire labels
    match the Hirano notation directly:

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
    noisy
        If ``True``, include the recovered ``X_ab`` stabilizer measurement and
        parser-side reconstructed X-checks used for noisy postselection.  If
        ``False``, omit those verification-only measurements.
    surface_readout_basis
        Optional final surface readout basis.  If given, all nine surface data
        qubits are measured in that basis by default.
    surface_readout_mode
        ``"data"`` destructively measures all nine surface data qubits.
        ``"logical_pauli"`` destructively measures only the weight-3 logical
        operator support qubits directly; the logical bit is the parity of
        the three outcomes.
        ``"logical_pauli_indirect"`` measures the weight-3 logical Pauli
        non-destructively with a fresh ancilla qubit.
    add_barriers
        Whether to insert conceptual barriers.
    """
    if merged_rounds < 1:
        raise ValueError("merged_rounds must be at least 1.")
    if surface_readout_mode not in {"data", "logical_pauli", "logical_pauli_indirect"}:
        raise ValueError(
            "surface_readout_mode must be 'data', 'logical_pauli', "
            "or 'logical_pauli_indirect'."
        )

    qregs = build_hirano_labeled_qregs(
        include_noisy_verification=noisy,
        surface_logical_measurement_basis=(
            surface_readout_basis
            if surface_readout_basis is not None and surface_readout_mode == "logical_pauli_indirect"
            else None
        ),
    )
    classical_specs = _classical_register_specs(
        include_cultivation=include_cultivation,
        merged_rounds=merged_rounds,
        noisy=noisy,
        surface_readout_basis=surface_readout_basis,
        surface_readout_mode=surface_readout_mode,
    )
    classical_registers_by_name = {
        spec.name: ClassicalRegister(len(spec.labels), spec.name)
        for spec in classical_specs
    }

    qubit_registers = (
        list(qregs.steane_data.values())
        + list(qregs.steane_helpers.values())
        + list(qregs.surface_data.values())
        + list(qregs.surface_z_ancillas.values())
        + list(qregs.surface_x_ancillas.values())
        + list(qregs.merge_ancillas.values())
        + list(qregs.logical_measurement_ancillas.values())
    )
    classical_registers = [classical_registers_by_name[spec.name] for spec in classical_specs]

    circuit = QuantumCircuit(*qubit_registers, *classical_registers)
    patch_qubits = _hirano_labeled_patch_qubits(qregs)

    append_patch_steane_state_injection_encoder(
        circuit,
        patch_qubits=patch_qubits,
        injected_state=injected_state,
    )

    if include_cultivation:
        append_hadamard_axis_projection_round(
            circuit,
            patch_qubits=patch_qubits,
            classical_bits=list(classical_registers_by_name["cult"]),
            add_barriers=add_barriers,
        )

    surface_qubits = _surface_data_qubits(qregs)
    prepare_plus(
        circuit,
        qubits=surface_qubits,
        add_barriers=add_barriers,
        barrier_label="prepare rotated d3 |+> product state",
    )
    append_z_stabilizer_measurements(
        circuit,
        surface_qubits=surface_qubits,
        z_ancillas=[qregs.surface_z_ancillas[label][0] for label in SURFACE_Z_ANCILLA_LABELS],
        classical_bits=list(classical_registers_by_name["init_z"]),
        round_label="surface init",
        prepare_ancillas=False,
        add_barriers=add_barriers,
    )

    append_boundary_zz_checks(
        circuit,
        registers=qregs,
        classical_bits=list(classical_registers_by_name["ls_zz"]),
        merged_rounds=merged_rounds,
        add_barriers=add_barriers,
    )
    append_measure_steane_data_in_x_basis(
        circuit,
        registers=qregs,
        classical_bits=list(classical_registers_by_name["steane_x"]),
        add_barriers=add_barriers,
    )

    if noisy:
        append_surface_x_ab_recovery_measurement(
            circuit,
            registers=qregs,
            classical_bit=classical_registers_by_name["surface_x_recovery"][0],
            add_barriers=add_barriers,
        )

    if surface_readout_basis is not None:
        if surface_readout_mode == "data":
            append_surface_logical_readout(
                circuit,
                registers=qregs,
                classical_bits=list(classical_registers_by_name[f"surface_{surface_readout_basis.lower()}"]),
                basis=surface_readout_basis,
                add_barriers=add_barriers,
            )
        elif surface_readout_mode == "logical_pauli":
            append_surface_logical_pauli_destructive(
                circuit,
                registers=qregs,
                classical_bits=list(classical_registers_by_name[f"surface_logical_{surface_readout_basis.lower()}"]),
                basis=surface_readout_basis,
                add_barriers=add_barriers,
            )
        else:  # logical_pauli_indirect
            append_surface_logical_pauli_measurement(
                circuit,
                registers=qregs,
                classical_bit=classical_registers_by_name[f"surface_logical_{surface_readout_basis.lower()}"][0],
                basis=surface_readout_basis,
                add_barriers=add_barriers,
            )

    return circuit


def parse_transfer_measurements(
    bitstring: str,
    *,
    include_cultivation: bool = True,
    merged_rounds: int = 1,
    noisy: bool = False,
    surface_readout_basis: SurfaceReadoutBasis | None = None,
    surface_readout_mode: SurfaceReadoutMode = "data",
) -> dict[str, object]:
    """Parse a Qiskit count key from the transfer circuit into labeled dicts.

    Qiskit returns measurement results as a single space-separated bitstring
    with one chunk per classical register, ordered from **last register to
    first**.  This function maps each chunk back to the register name used by
    the circuit and exposes both the raw chunk and Hirano-labeled bits under
    ``measurements["registers"]``.

    Parameters
    ----------
    bitstring
        A Qiskit counts key, e.g. ``"0000000 000 0000 00000"``.
    include_cultivation
        Whether the circuit included the cultivation round (adds a 5-bit
        ``cult`` register).
    merged_rounds
        Number of repeated merged-boundary ZZ rounds (each adds 2 bits).
    noisy
        Whether the circuit included the ``surface_x_recovery`` register used
        to reconstruct noisy-transfer X checks.
    surface_readout_basis
        If set, the circuit included a final surface readout register.
    surface_readout_mode
        ``"data"`` parses a nine-bit data readout. ``"logical_pauli"`` parses
        a one-bit direct logical-Pauli measurement.

    Returns
    -------
    dict with the following keys (all present unless noted):

    ``"registers"``
        Dict keyed by classical-register name.  Each value contains the raw
        Qiskit count chunk, semantic labels, and parsed bit dict.

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

    ``"surface_logical_readout"`` *(only for logical-Pauli readout)*
        One-bit dict such as ``{"Z_abc": 0}`` or ``{"X_adg": 1}``.
    """
    specs = _classical_register_specs(
        include_cultivation=include_cultivation,
        merged_rounds=merged_rounds,
        noisy=noisy,
        surface_readout_basis=surface_readout_basis,
        surface_readout_mode=surface_readout_mode,
    )
    parts = bitstring.split()
    if len(parts) != len(specs):
        raise ValueError(
            f"Expected {len(specs)} space-separated register chunks, got {len(parts)}."
        )

    register_chunks: dict[str, dict[str, object]] = {}
    for spec, chunk in zip(reversed(specs), parts, strict=True):
        register_chunks[spec.name] = {
            "raw": chunk,
            "labels": spec.labels,
            "bits": _parse_register_chunk(chunk, spec.labels),
        }

    parsed: dict[str, object] = {"registers": register_chunks}

    def register_bits(name: str) -> dict[str, int]:
        return register_chunks[name]["bits"]

    if surface_readout_basis is not None:
        if surface_readout_mode == "data":
            parsed["surface_readout"] = register_bits(f"surface_{surface_readout_basis.lower()}")
        else:  # logical_pauli or logical_pauli_indirect
            parsed["surface_logical_readout"] = register_bits(
                f"surface_logical_{surface_readout_basis.lower()}"
            )

    parsed["steane_x"] = register_bits("steane_x")

    merged_bits = register_bits("ls_zz")
    parsed["merged_zz_rounds"] = [
        {
            check_name: merged_bits[
                f"round{round_index + 1}_{check_name}"
            ]
            for check_name, _, _ in MERGED_BOUNDARY_CHECKS
        }
        for round_index in range(merged_rounds)
    ]

    parsed["surface_init_z"] = register_bits("init_z")

    if include_cultivation:
        parsed["cultivation"] = register_bits("cult")

    if noisy:
        parsed["surface_x_recovery"] = register_bits("surface_x_recovery")
        parsed["reconstructed_x_checks"] = reconstructed_x_checks(parsed)
        parsed["local_x_verification_checks"] = local_x_verification_checks(parsed)
        parsed["x_verification_succeeds"] = x_verification_succeeds(parsed)

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
    return sum(steane_x[label] for label in HIRANO_STEANE_LOGICAL_X) % 2


def reconstructed_x_checks(measurements: dict[str, object]) -> dict[str, int]:
    """Reconstruct paper-style X checks from destructive and recovered X bits."""
    steane_x = measurements["steane_x"]
    checks = {
        name: _parity(steane_x, support)
        for name, support in STEANE_X_RECONSTRUCTED_CHECKS.items()
        if name != "X0145"
    }
    surface_x_recovery = measurements.get("surface_x_recovery")
    if surface_x_recovery is not None:
        checks["X0145ab"] = (
            _parity(steane_x, STEANE_X_RECONSTRUCTED_CHECKS["X0145"])
            ^ surface_x_recovery["X_ab"]
        )
    return checks


def local_x_verification_checks(measurements: dict[str, object]) -> dict[str, int]:
    """Return X checks that are deterministic for this repo's current labeling."""
    steane_x = measurements["steane_x"]
    checks = {
        name: _parity(steane_x, support)
        for name, support in LOCAL_STEANE_X_STABILIZER_CHECKS.items()
    }
    surface_x_recovery = measurements.get("surface_x_recovery")
    if surface_x_recovery is None:
        return checks

    mixed_bits = {**steane_x, "X_ab": surface_x_recovery["X_ab"]}
    checks.update(
        {
            name: _parity(mixed_bits, support)
            for name, support in LOCAL_X_AB_CORRELATION_CHECKS.items()
        }
    )
    return checks


def x_verification_succeeds(measurements: dict[str, object]) -> bool:
    """Return whether all reconstructed X checks have trivial syndrome."""
    return all(bit == 0 for bit in local_x_verification_checks(measurements).values())


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
        if "surface_logical_readout" not in measurements:
            raise ValueError("measurements do not include a surface logical readout.")
        logical_dict = measurements["surface_logical_readout"]
        # Destructive mode: multiple bits, compute parity.
        # Indirect mode: single bit with the combined label.
        combined_label = _surface_logical_readout_label(basis)
        if combined_label in logical_dict:
            return logical_dict[combined_label]
        # Destructive: parity of individual support-qubit bits.
        return sum(logical_dict.values()) % 2

    support = DEFAULT_LOGICAL_X if basis.upper() == "X" else DEFAULT_LOGICAL_Z
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
