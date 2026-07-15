"""Shared Qiskit helpers for Steane-to-rotated-surface state transfer.

The distance-specific modules (`state_transfer_to_d3_rotated_qiskit` and
`state_transfer_to_d5_rotated_qiskit`) define the target patch geometry and
which checks exist on that patch.  This module keeps the target-independent
parts in one place:

- Hirano/paper labels for the Steane/color-code data qubits.
- Mapping between those labels and the 11-site Crumble encoder patch.
- Mixed Steane-surface parity measurements used by lattice surgery.
- Classical-register layout shared by circuit construction and parsing.
- The complete target-independent stabilizer-level transfer builder.

Result parsing, acceptance policy, syndrome decoding, and Pauli-frame
interpretation live in ``state_transfer_analysis_qiskit``. The
``ClassicalRegisterSpec`` list is the single source of truth shared by both
modules.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

from research_qec_haoran.steane_code.magic_state_cultivation_qiskit import (
    append_hadamard_axis_projection_round,
)
from research_qec_haoran.steane_code.state_injection_qiskit import (
    InjectedStateName,
    append_patch_logical_measurement,
    append_patch_steane_state_injection_encoder_crumble_sequences,
)

SurfaceReadoutBasis = Literal["X", "Z"]
SurfaceReadoutMode = Literal["data", "logical_pauli", "logical_pauli_indirect"]
LATTICE_SURGERY_ROUNDS = 3
MergedBoundaryCheck = tuple[str, tuple[str, ...], tuple[str, ...]]
TopXRecoveryCheck = tuple[str, tuple[str, ...]]
CircuitAppender = Callable[..., None]

HIRANO_STEANE_DATA_ORDER: tuple[str, ...] = ("0", "1", "2", "3", "4", "5", "6")
HIRANO_STEANE_GEOMETRIC_ORDER: tuple[str, ...] = ("6", "4", "0", "2", "1", "5", "3")
HIRANO_STEANE_LOGICAL_X: tuple[str, ...] = ("1", "4", "6")
HIRANO_STEANE_LOGICAL_Z: tuple[str, ...] = ("1", "5", "3")

STEANE_LS_VERIFICATION_CHECKS: dict[str, tuple[str, tuple[str, ...]]] = {
    "Z0145": ("Z", ("0", "1", "4", "5")),
    "Z0235": ("Z", ("0", "2", "3", "5")),
    "X0235": ("X", ("0", "2", "3", "5")),
    "Z0246": ("Z", ("0", "2", "4", "6")),
    "X0246": ("X", ("0", "2", "4", "6")),
}
STEANE_LS_VERIFICATION_ROUNDS: tuple[tuple[str, ...], ...] = (
    ("Z0145", "Z0235", "X0235", "Z0246", "X0246"),
    ("Z0145", "Z0235", "Z0246"),
)
STEANE_LS_VERIFICATION_LABELS: tuple[str, ...] = tuple(
    f"round{round_index + 1}_{check_name}"
    for round_index, round_checks in enumerate(STEANE_LS_VERIFICATION_ROUNDS)
    for check_name in round_checks
)
STEANE_LS_VERIFICATION_ANCILLA_HELPER: dict[str, str] = {
    "Z0145": "h6",
    "Z0235": "h7",
    "X0235": "h7",
    "Z0246": "h2",
    "X0246": "h2",
}
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


@dataclass(frozen=True)
class HiranoLabeledQRegs:
    """Paper-labeled one-qubit registers used by the transfer circuit.

    The transfer code intentionally gives each physical qubit its own
    `QuantumRegister`.  This makes Qiskit diagrams and count-register names
    match the paper notation:

    - `steane_data`: Hirano color-code labels `"0"` through `"6"`.
    - `steane_helpers`: helper/routing sites from the 11-site patch.
    - `surface_data`: target rotated-patch labels such as `a..i` or `a..y`.
    - `surface_z_ancillas` and `surface_x_ancillas`: syndrome ancillas reused
      across initialization, merged-interval checks, recovery, and optional QEC.
    - `merge_ancillas`: ancillas for mixed Steane-surface `ZZ` checks.
    - `logical_measurement_ancillas`: optional non-destructive final logical
      Pauli readout ancillas.
    """

    steane_data: dict[str, QuantumRegister]
    steane_helpers: dict[str, QuantumRegister]
    surface_data: dict[str, QuantumRegister]
    surface_z_ancillas: dict[str, QuantumRegister]
    surface_x_ancillas: dict[str, QuantumRegister]
    merge_ancillas: dict[str, QuantumRegister]
    logical_measurement_ancillas: dict[str, QuantumRegister]


@dataclass(frozen=True)
class ClassicalRegisterSpec:
    """Human-readable description of one Qiskit classical register.

    `name` is the actual Qiskit classical-register name.  `labels` are the
    left-to-right semantic bit labels used after parsing.  Keeping these labels
    close to the circuit builder avoids duplicating Qiskit's bit-ordering rules
    in tests and analysis code.
    """

    name: str
    labels: tuple[str, ...]


@dataclass(frozen=True)
class SurfaceCodeTransferSpec:
    """Distance-specific data and circuit operations for one transfer target.

    The d=3 and d=5 transfer modules differ only in patch geometry, logical
    supports, merged-boundary checks, and top-edge X checks.  Binding those
    values once avoids maintaining two nearly identical protocol builders.

    ``top_x_recovery_checks`` preserves measurement order.  Each item is a
    paper-style check label (for example ``X_ab``) and its surface-data support.
    The callable fields bind the shared rotated-patch API implemented by the
    distance-specific surface-code module.
    """

    name: str
    surface_order: tuple[str, ...]
    logical_x: tuple[str, ...]
    logical_z: tuple[str, ...]
    z_stabilizers: Mapping[str, tuple[str, ...]]
    x_stabilizers: Mapping[str, tuple[str, ...]]
    merged_boundary_checks: tuple[MergedBoundaryCheck, ...]
    top_x_recovery_checks: tuple[TopXRecoveryCheck, ...]
    prepare_plus: CircuitAppender
    append_z_stabilizer_measurements: CircuitAppender
    append_x_stabilizer_measurements: CircuitAppender
    append_logical_measurement: CircuitAppender

    @property
    def z_labels(self) -> tuple[str, ...]:
        """All ordinary surface Z-stabilizer labels in circuit order."""
        return tuple(self.z_stabilizers)

    @property
    def x_labels(self) -> tuple[str, ...]:
        """All ordinary surface X-stabilizer labels in circuit order."""
        return tuple(self.x_stabilizers)

    @property
    def merged_x_labels(self) -> tuple[str, ...]:
        """Surface X checks commuting with the active merged-boundary checks."""
        withheld = set(self.top_x_recovery_labels)
        return tuple(label for label in self.x_stabilizers if label not in withheld)

    @property
    def top_x_recovery_labels(self) -> tuple[str, ...]:
        """Top-edge X checks recovered after Steane demolition."""
        return tuple(label for label, _ in self.top_x_recovery_checks)

    @property
    def z_ancilla_labels(self) -> tuple[str, ...]:
        """Qubit-register labels used for surface Z-check ancillas."""
        return tuple(label.lower() for label in self.z_labels)

    @property
    def x_ancilla_labels(self) -> tuple[str, ...]:
        """Qubit-register labels used for surface X-check ancillas."""
        return tuple(label.lower() for label in self.x_labels)


def build_transfer_qregs(
    spec: SurfaceCodeTransferSpec,
    *,
    surface_logical_measurement_basis: SurfaceReadoutBasis | None = None,
) -> HiranoLabeledQRegs:
    """Create all quantum registers required by the paper transfer."""
    logical_measurement_ancillas = {}
    if surface_logical_measurement_basis is not None:
        basis = surface_logical_measurement_basis.lower()
        logical_measurement_ancillas[f"logical_{basis}"] = QuantumRegister(1, f"logical_{basis}")

    return HiranoLabeledQRegs(
        steane_data={
            label: QuantumRegister(1, label) for label in HIRANO_STEANE_DATA_ORDER
        },
        steane_helpers={
            label: QuantumRegister(1, label) for label in STEANE_HELPER_PATCH_LABELS
        },
        surface_data={label: QuantumRegister(1, label) for label in spec.surface_order},
        surface_z_ancillas={
            label: QuantumRegister(1, label) for label in spec.z_ancilla_labels
        },
        surface_x_ancillas={
            label: QuantumRegister(1, label) for label in spec.x_ancilla_labels
        },
        merge_ancillas={
            f"zz_{name[1:].lower()}": QuantumRegister(1, f"zz_{name[1:].lower()}")
            for name, _, _ in spec.merged_boundary_checks
        },
        logical_measurement_ancillas=logical_measurement_ancillas,
    )


def hirano_labeled_patch_qubits(registers: HiranoLabeledQRegs) -> list:
    """Return the 11-site cultivation patch qubits in patch-label order."""
    patch_qubits: list[object] = [None] * 11
    for site, label in GOOGLE_SITE_TO_HIRANO_LABEL.items():
        patch_qubits[site] = registers.steane_data[label][0]
    for label, site in STEANE_HELPER_LABEL_TO_GOOGLE_SITE.items():
        patch_qubits[site] = registers.steane_helpers[label][0]
    return patch_qubits


def surface_data_qubits(
    registers: HiranoLabeledQRegs,
    surface_order: Sequence[str],
) -> list:
    """Return surface data qubits in the target patch's label order."""
    return [registers.surface_data[label][0] for label in surface_order]


def parity(bits: Mapping[str, int], labels: Sequence[str]) -> int:
    """Return the mod-2 parity over a labeled bit dictionary."""
    return sum(bits[label] for label in labels) % 2


def merged_zz_labels(
    merged_boundary_checks: Sequence[tuple[str, tuple[str, ...], tuple[str, ...]]],
) -> tuple[str, ...]:
    """Return labels for all repeated merged-boundary check outcomes."""
    return tuple(
        f"round{round_index + 1}_{check_name}"
        for round_index in range(LATTICE_SURGERY_ROUNDS)
        for check_name, _, _ in merged_boundary_checks
    )


def round_labels(labels: Sequence[str]) -> tuple[str, ...]:
    """Prefix repeated-round labels with ``roundN_`` for flat registers."""
    return tuple(
        f"round{round_index + 1}_{label}"
        for round_index in range(LATTICE_SURGERY_ROUNDS)
        for label in labels
    )


def surface_logical_readout_label(
    basis: SurfaceReadoutBasis,
    *,
    logical_x: Sequence[str],
    logical_z: Sequence[str],
) -> str:
    """Return the combined one-bit logical-readout label for indirect mode."""
    support = logical_x if basis.upper() == "X" else logical_z
    return f"{basis.upper()}_{''.join(support)}"


def classical_register_specs(
    *,
    include_cultivation: bool,
    merged_rounds: int,
    include_merged_surface_syndromes: bool,
    include_steane_verification: bool,
    include_surface_qec: bool,
    surface_qec_rounds: int,
    surface_readout_basis: SurfaceReadoutBasis | None,
    surface_readout_mode: SurfaceReadoutMode,
    merged_boundary_checks: Sequence[tuple[str, tuple[str, ...], tuple[str, ...]]],
    merged_surface_z_labels: Sequence[str],
    merged_surface_x_labels: Sequence[str],
    surface_init_z_labels: Sequence[str],
    surface_post_z_labels: Sequence[str],
    surface_post_x_labels: Sequence[str],
    surface_x_recovery_labels: Sequence[str],
    surface_order: Sequence[str],
    logical_x: Sequence[str],
    logical_z: Sequence[str],
) -> list[ClassicalRegisterSpec]:
    """Return the classical-register layout for a transfer circuit.

    The order returned here must exactly match the order used when constructing
    the `QuantumCircuit`.  `parse_transfer_measurements_common` relies on the
    same list to parse Qiskit count keys back into protocol-stage dictionaries.

    Register groups, in protocol order:

    - `enc_helpers`: Crumble encoder helper-site postselection.
    - `cult`: optional Steane H-axis cultivation checks.
    - `init_z`: initial surface Z stabilizer frame after preparing `|+>`.
    - `ls_zz`: repeated mixed Steane-surface merged-boundary checks.
    - `merged_surface_z/x`: compatible surface stabilizers measured during the
      merged-code interval.
    - `steane_verify`: Steane/color-code LS verification checks.
    - `steane_x`: destructive X-basis readout of Steane data qubits.
    - `surface_x_recovery`: recovered top-edge X checks after demolition.
    - `surface_post_z/x`: optional full post-transfer surface-code QEC rounds.
    - final surface logical readout, if requested.
    """
    specs: list[ClassicalRegisterSpec] = [
        ClassicalRegisterSpec(
            "enc_helpers",
            ("enc_h1", "enc_h2", "enc_h6", "enc_h7"),
        )
    ]
    if include_cultivation:
        specs.append(
            ClassicalRegisterSpec(
                "cult",
                ("root_x", "helper_h1", "helper_h2", "helper_h6", "helper_h7"),
            )
        )
    specs.extend(
        [
            ClassicalRegisterSpec("init_z", tuple(surface_init_z_labels)),
            ClassicalRegisterSpec(
                "ls_zz",
                merged_zz_labels(merged_rounds, merged_boundary_checks),
            ),
        ]
    )
    if include_merged_surface_syndromes:
        specs.extend(
            [
                ClassicalRegisterSpec(
                    "merged_surface_z",
                    round_labels(merged_rounds, merged_surface_z_labels),
                ),
                ClassicalRegisterSpec(
                    "merged_surface_x",
                    round_labels(merged_rounds, merged_surface_x_labels),
                ),
            ]
        )
    if include_steane_verification:
        specs.append(ClassicalRegisterSpec("steane_verify", STEANE_LS_VERIFICATION_LABELS))
    specs.append(ClassicalRegisterSpec("steane_x", HIRANO_STEANE_DATA_ORDER))
    if surface_x_recovery_labels:
        specs.append(ClassicalRegisterSpec("surface_x_recovery", tuple(surface_x_recovery_labels)))
    if include_surface_qec:
        specs.extend(
            [
                ClassicalRegisterSpec(
                    "surface_post_z",
                    round_labels(surface_qec_rounds, surface_post_z_labels),
                ),
                ClassicalRegisterSpec(
                    "surface_post_x",
                    round_labels(surface_qec_rounds, surface_post_x_labels),
                ),
            ]
        )
    if surface_readout_basis is not None:
        if surface_readout_mode == "data":
            specs.append(
                ClassicalRegisterSpec(
                    f"surface_{surface_readout_basis.lower()}",
                    tuple(surface_order),
                )
            )
        elif surface_readout_mode == "logical_pauli":
            support = logical_x if surface_readout_basis.upper() == "X" else logical_z
            specs.append(
                ClassicalRegisterSpec(
                    f"surface_logical_{surface_readout_basis.lower()}",
                    tuple(support),
                )
            )
        elif surface_readout_mode == "logical_pauli_indirect":
            specs.append(
                ClassicalRegisterSpec(
                    f"surface_logical_{surface_readout_basis.lower()}",
                    (
                        surface_logical_readout_label(
                            surface_readout_basis,
                            logical_x=logical_x,
                            logical_z=logical_z,
                        ),
                    ),
                )
            )
        else:
            raise ValueError(
                "surface_readout_mode must be 'data', 'logical_pauli', or 'logical_pauli_indirect'."
            )
    return specs


def transfer_classical_register_specs(
    spec: SurfaceCodeTransferSpec,
    *,
    include_cultivation: bool,
    merged_rounds: int,
    include_merged_surface_syndromes: bool,
    include_steane_verification: bool,
    include_surface_qec: bool,
    surface_qec_rounds: int,
    surface_readout_basis: SurfaceReadoutBasis | None,
    surface_readout_mode: SurfaceReadoutMode,
) -> list[ClassicalRegisterSpec]:
    """Return the classical-register layout bound to ``spec``."""
    return classical_register_specs(
        include_cultivation=include_cultivation,
        merged_rounds=merged_rounds,
        include_merged_surface_syndromes=include_merged_surface_syndromes,
        include_steane_verification=include_steane_verification,
        include_surface_qec=include_surface_qec,
        surface_qec_rounds=surface_qec_rounds,
        surface_readout_basis=surface_readout_basis,
        surface_readout_mode=surface_readout_mode,
        merged_boundary_checks=spec.merged_boundary_checks,
        merged_surface_z_labels=spec.z_labels,
        merged_surface_x_labels=spec.merged_x_labels,
        surface_init_z_labels=spec.z_labels,
        surface_post_z_labels=spec.z_labels,
        surface_post_x_labels=spec.x_labels,
        surface_x_recovery_labels=spec.top_x_recovery_labels,
        surface_order=spec.surface_order,
        logical_x=spec.logical_x,
        logical_z=spec.logical_z,
    )


def _append_mixed_z_parity_measurement(
    circuit: QuantumCircuit,
    *,
    data_qubits,
    ancilla,
    classical_bit,
) -> None:
    """Measure a mixed Steane-surface Z parity with one ancilla."""
    circuit.reset(ancilla)
    for qubit in data_qubits:
        circuit.cx(qubit, ancilla)
    circuit.measure(ancilla, classical_bit)


def append_surface_logical_readout(
    circuit: QuantumCircuit,
    *,
    spec: SurfaceCodeTransferSpec,
    registers: HiranoLabeledQRegs,
    classical_bits: Sequence,
    basis: SurfaceReadoutBasis,
    add_barriers: bool = True,
) -> None:
    """Measure all surface data qubits in ``basis`` for logical readout."""
    if len(classical_bits) != len(spec.surface_order):
        raise ValueError(
            f"classical_bits must contain {len(spec.surface_order)} surface-data bits."
        )

    spec.append_logical_measurement(
        circuit,
        basis.upper(),
        surface_qubits=surface_data_qubits(registers, spec.surface_order),
        support=spec.surface_order,
        classical_bits=classical_bits,
        destructive=True,
    )

    if add_barriers:
        circuit.barrier(label=f"surface logical {basis.upper()} readout")


def append_boundary_zz_checks(
    circuit: QuantumCircuit,
    *,
    registers: HiranoLabeledQRegs,
    classical_bits: Sequence,
    merged_boundary_checks: Sequence[tuple[str, tuple[str, ...], tuple[str, ...]]],
    merged_rounds: int = 1,
    add_barriers: bool = True,
) -> None:
    """Append repeated merged-boundary ``ZZ`` checks.

    Each check may involve Steane/color-code data labels, surface-code labels,
    or both.  The d=5 paper schedule uses `Z1a`, `Z35bc`, and `Zde`; the d=3
    adaptation uses the subset available on the smaller top boundary.

    The parity of all merged-boundary checks in a round is later interpreted as
    the logical `Z_L Z_L` measurement.  A nonzero outcome becomes an X byproduct
    on the target surface-code Pauli frame.  Repetition is used to detect
    measurement disagreement through `merged_boundary_checks_stable`.
    """
    expected_bits = merged_rounds * len(merged_boundary_checks)
    if len(classical_bits) != expected_bits:
        raise ValueError(
            f"classical_bits must contain {expected_bits} bits for {merged_rounds} rounds."
        )

    for round_index in range(merged_rounds):
        for check_index, (check_name, steane_labels, surface_labels) in enumerate(
            merged_boundary_checks
        ):
            bit_index = round_index * len(merged_boundary_checks) + check_index
            data_qubits = [registers.steane_data[label][0] for label in steane_labels] + [
                registers.surface_data[label][0] for label in surface_labels
            ]
            _append_mixed_z_parity_measurement(
                circuit,
                data_qubits=data_qubits,
                ancilla=registers.merge_ancillas[f"zz_{check_name[1:].lower()}"][0],
                classical_bit=classical_bits[bit_index],
            )
        if add_barriers:
            circuit.barrier(label=f"merged ZZ round {round_index + 1}")


def append_measure_steane_data_in_x_basis(
    circuit: QuantumCircuit,
    *,
    registers: HiranoLabeledQRegs,
    classical_bits: Sequence,
    add_barriers: bool = True,
) -> None:
    """Demolish the Steane block by measuring all seven data qubits in X."""
    if len(classical_bits) != len(HIRANO_STEANE_DATA_ORDER):
        raise ValueError("classical_bits must contain seven Steane-data bits.")

    append_patch_logical_measurement(
        circuit,
        "X",
        patch_qubits=hirano_labeled_patch_qubits(registers),
        support=tuple(HIRANO_LABEL_TO_GOOGLE_SITE[label] for label in HIRANO_STEANE_DATA_ORDER),
        classical_bits=classical_bits,
        destructive=True,
    )

    if add_barriers:
        circuit.barrier(label="measure Steane data in X")


def append_steane_lattice_surgery_verification_round(
    circuit: QuantumCircuit,
    *,
    registers: HiranoLabeledQRegs,
    classical_bits: Sequence,
    round_index: int,
    add_barriers: bool = True,
) -> None:
    """Measure one paper-defined Steane check group during a merged round.

    The Qiskit model represents each commuting stabilizer as an ideal ancilla
    parity measurement.  It preserves the paper's measured observables and
    round ordering, while leaving the hardware-level superdense gate schedule
    to a lower-level circuit model.
    """
    if not 0 <= round_index < len(STEANE_LS_VERIFICATION_ROUNDS):
        raise ValueError(f"round_index must be in [0, {len(STEANE_LS_VERIFICATION_ROUNDS) - 1}].")

    round_checks = STEANE_LS_VERIFICATION_ROUNDS[round_index]
    if len(classical_bits) != len(round_checks):
        raise ValueError(
            f"classical_bits must contain {len(round_checks)} bits for Steane round "
            f"{round_index + 1}."
        )

    patch_qubits = hirano_labeled_patch_qubits(registers)
    for check_name, classical_bit in zip(round_checks, classical_bits, strict=True):
        basis, support = STEANE_LS_VERIFICATION_CHECKS[check_name]
        helper_label = STEANE_LS_VERIFICATION_ANCILLA_HELPER[check_name]
        ancilla = registers.steane_helpers[helper_label][0]
        circuit.reset(ancilla)
        append_patch_logical_measurement(
            circuit,
            basis,
            patch_qubits=patch_qubits,
            support=tuple(HIRANO_LABEL_TO_GOOGLE_SITE[label] for label in support),
            ancilla_qubit=ancilla,
            classical_bits=[classical_bit],
            destructive=False,
        )

    if add_barriers:
        circuit.barrier(label=f"Steane LS verification round {round_index + 1}")


def append_steane_lattice_surgery_verification(
    circuit: QuantumCircuit,
    *,
    registers: HiranoLabeledQRegs,
    classical_bits: Sequence,
    add_barriers: bool = True,
) -> None:
    """Measure both color-code check groups used for LS postselection.

    This compatibility helper appends both groups consecutively:

    - Round 1: `Z0145`, `Z0235`, `X0235`, `Z0246`, `X0246`.
    - Round 2: `Z0145`, `Z0235`, `Z0246`.

    The complete transfer builder calls
    ``append_steane_lattice_surgery_verification_round`` inside merged rounds 1
    and 2 so the protocol ordering matches Section III A.
    """
    if len(classical_bits) != len(STEANE_LS_VERIFICATION_LABELS):
        raise ValueError(f"classical_bits must contain {len(STEANE_LS_VERIFICATION_LABELS)} bits.")

    bit_start = 0
    for round_index, round_checks in enumerate(STEANE_LS_VERIFICATION_ROUNDS):
        bit_stop = bit_start + len(round_checks)
        append_steane_lattice_surgery_verification_round(
            circuit,
            registers=registers,
            classical_bits=classical_bits[bit_start:bit_stop],
            round_index=round_index,
            add_barriers=add_barriers,
        )
        bit_start = bit_stop


def append_merged_surface_syndrome_round(
    circuit: QuantumCircuit,
    *,
    spec: SurfaceCodeTransferSpec,
    registers: HiranoLabeledQRegs,
    surface_qubits: Sequence,
    z_classical_bits: Sequence,
    x_classical_bits: Sequence,
    round_label: str,
    add_barriers: bool = True,
) -> None:
    """Measure all surface checks compatible with the merged code.

    All ordinary Z checks commute with the merged-boundary Z checks.  The
    top-edge X checks listed by ``spec.top_x_recovery_checks`` do not, so they
    are withheld until the Steane patch has been demolished.
    """
    spec.append_z_stabilizer_measurements(
        circuit,
        surface_qubits=surface_qubits,
        z_ancillas=[registers.surface_z_ancillas[label.lower()][0] for label in spec.z_labels],
        classical_bits=z_classical_bits,
        round_label=f"{round_label} surface Z",
        stabilizer_names=spec.z_labels,
        prepare_ancillas=True,
        add_barriers=add_barriers,
    )
    spec.append_x_stabilizer_measurements(
        circuit,
        surface_qubits=surface_qubits,
        x_ancillas=[
            registers.surface_x_ancillas[label.lower()][0] for label in spec.merged_x_labels
        ],
        classical_bits=x_classical_bits,
        round_label=f"{round_label} surface X",
        stabilizer_names=spec.merged_x_labels,
        prepare_ancillas=True,
        add_barriers=add_barriers,
    )


def append_surface_top_x_recovery_measurements(
    circuit: QuantumCircuit,
    *,
    spec: SurfaceCodeTransferSpec,
    registers: HiranoLabeledQRegs,
    classical_bits: Sequence,
    add_barriers: bool = True,
) -> None:
    """Recover surface top-edge X checks after Steane demolition."""
    if len(classical_bits) != len(spec.top_x_recovery_checks):
        raise ValueError(
            "classical_bits must contain "
            f"{len(spec.top_x_recovery_checks)} top-edge X recovery bits."
        )

    surface_qubits = surface_data_qubits(registers, spec.surface_order)
    for (check_name, support), classical_bit in zip(
        spec.top_x_recovery_checks,
        classical_bits,
        strict=True,
    ):
        ancilla_name = check_name.lower()
        if ancilla_name not in registers.surface_x_ancillas:
            raise ValueError(f"registers do not include the {ancilla_name} ancilla.")
        spec.append_logical_measurement(
            circuit,
            "X",
            surface_qubits=surface_qubits,
            support=support,
            ancilla_qubit=registers.surface_x_ancillas[ancilla_name][0],
            classical_bits=[classical_bit],
            destructive=False,
        )
    if add_barriers:
        labels = " and ".join(spec.top_x_recovery_labels)
        circuit.barrier(label=f"recover surface {labels}")


def append_surface_logical_pauli_measurement(
    circuit: QuantumCircuit,
    *,
    spec: SurfaceCodeTransferSpec,
    registers: HiranoLabeledQRegs,
    classical_bit,
    basis: SurfaceReadoutBasis,
    destructive: bool,
    add_barriers: bool = True,
) -> None:
    """Measure a target logical Pauli on its standard support."""
    normalized_basis = basis.upper()
    if normalized_basis not in {"X", "Z"}:
        raise ValueError("basis must be 'X' or 'Z'.")

    support = spec.logical_x if normalized_basis == "X" else spec.logical_z
    kwargs: dict[str, object] = {
        "surface_qubits": surface_data_qubits(registers, spec.surface_order),
        "support": support,
        "classical_bits": classical_bit if destructive else [classical_bit],
        "destructive": destructive,
    }
    if not destructive:
        ancilla_name = f"logical_{normalized_basis.lower()}"
        if ancilla_name not in registers.logical_measurement_ancillas:
            raise ValueError(f"registers do not include the {ancilla_name} ancilla.")
        kwargs["ancilla_qubit"] = registers.logical_measurement_ancillas[ancilla_name][0]

    spec.append_logical_measurement(circuit, normalized_basis, **kwargs)
    if add_barriers:
        mode = "destructive " if destructive else ""
        circuit.barrier(
            label=f"{mode}logical {normalized_basis}_{''.join(support)}",
        )


def build_lattice_surgery_circuit(
    spec: SurfaceCodeTransferSpec,
    *,
    injected_state: InjectedStateName | tuple[complex, complex] = "H",
    include_cultivation: bool = True,
    merged_rounds: int = 3,
    include_merged_surface_syndromes: bool = True,
    include_steane_verification: bool = True,
    include_surface_qec: bool = False,
    surface_qec_rounds: int = 3,
    surface_readout_basis: SurfaceReadoutBasis | None = None,
    surface_readout_mode: SurfaceReadoutMode = "data",
    add_barriers: bool = True,
) -> QuantumCircuit:
    """Build the shared Steane-to-rotated-surface transfer protocol.

    The default sequence follows Section III A at the stabilizer-measurement
    level.  The two Steane check groups are part of merged rounds 1 and 2, and
    Steane demolition plus top-edge X recovery follows round 3.  The commuting
    checks are serialized in Qiskit; this function does not reproduce the
    paper implementation's hardware-specific superdense gate interleaving.

    ``include_surface_qec`` remains an explicit extension for experiments.  It
    is not enabled by the paper-facing default transfer.
    """
    if merged_rounds < 1:
        raise ValueError("merged_rounds must be at least 1.")
    if include_steane_verification and merged_rounds < len(STEANE_LS_VERIFICATION_ROUNDS):
        raise ValueError("include_steane_verification requires at least two merged rounds.")
    if surface_readout_mode not in {"data", "logical_pauli", "logical_pauli_indirect"}:
        raise ValueError(
            "surface_readout_mode must be 'data', 'logical_pauli', or 'logical_pauli_indirect'."
        )
    if include_surface_qec and surface_qec_rounds < 1:
        raise ValueError("surface_qec_rounds must be at least 1 when include_surface_qec=True.")

    qregs = build_transfer_qregs(
        spec,
        include_x_recovery=bool(spec.top_x_recovery_checks),
        include_merged_surface_syndromes=include_merged_surface_syndromes,
        include_surface_qec=include_surface_qec,
        surface_logical_measurement_basis=(
            surface_readout_basis
            if surface_readout_basis is not None
            and surface_readout_mode == "logical_pauli_indirect"
            else None
        ),
    )
    classical_specs = transfer_classical_register_specs(
        spec,
        include_cultivation=include_cultivation,
        merged_rounds=merged_rounds,
        include_merged_surface_syndromes=include_merged_surface_syndromes,
        include_steane_verification=include_steane_verification,
        include_surface_qec=include_surface_qec,
        surface_qec_rounds=surface_qec_rounds,
        surface_readout_basis=surface_readout_basis,
        surface_readout_mode=surface_readout_mode,
    )
    cregs = {
        register_spec.name: ClassicalRegister(len(register_spec.labels), register_spec.name)
        for register_spec in classical_specs
    }
    qreg_groups = (
        qregs.steane_data,
        qregs.steane_helpers,
        qregs.surface_data,
        qregs.surface_z_ancillas,
        qregs.surface_x_ancillas,
        qregs.merge_ancillas,
        qregs.logical_measurement_ancillas,
    )
    circuit = QuantumCircuit(
        *(register for group in qreg_groups for register in group.values()),
        *(cregs[register_spec.name] for register_spec in classical_specs),
    )

    # Prepare the accepted color-code input using the existing Steane front end.
    patch_qubits = hirano_labeled_patch_qubits(qregs)
    append_patch_steane_state_injection_encoder_crumble_sequences(
        circuit,
        patch_qubits=patch_qubits,
        injected_state=injected_state,
        helper_classical_bits=list(cregs["enc_helpers"]),
    )
    if include_cultivation:
        append_hadamard_axis_projection_round(
            circuit,
            patch_qubits=patch_qubits,
            classical_bits=list(cregs["cult"]),
            add_barriers=add_barriers,
        )

    # Project a product |+> state into the target surface code and record its Z frame.
    surface_qubits = surface_data_qubits(qregs, spec.surface_order)
    spec.prepare_plus(
        circuit,
        qubits=surface_qubits,
        add_barriers=add_barriers,
        barrier_label=f"prepare {spec.name} |+> product state",
    )
    spec.append_z_stabilizer_measurements(
        circuit,
        surface_qubits=surface_qubits,
        z_ancillas=[qregs.surface_z_ancillas[label][0] for label in spec.z_ancilla_labels],
        classical_bits=list(cregs["init_z"]),
        round_label="surface init",
        prepare_ancillas=False,
        add_barriers=add_barriers,
    )

    # Run merged-code rounds, including the two paper-defined Steane check groups.
    merged_zz_bits = list(cregs["ls_zz"])
    merged_z_bits = list(cregs["merged_surface_z"]) if include_merged_surface_syndromes else []
    merged_x_bits = list(cregs["merged_surface_x"]) if include_merged_surface_syndromes else []
    steane_verify_bits = list(cregs["steane_verify"]) if include_steane_verification else []
    steane_verify_start = 0
    for round_index in range(merged_rounds):
        zz_start = round_index * len(spec.merged_boundary_checks)
        append_boundary_zz_checks(
            circuit,
            registers=qregs,
            classical_bits=merged_zz_bits[zz_start : zz_start + len(spec.merged_boundary_checks)],
            merged_boundary_checks=spec.merged_boundary_checks,
            merged_rounds=1,
            add_barriers=False,
        )

        if include_merged_surface_syndromes:
            z_start = round_index * len(spec.z_labels)
            x_start = round_index * len(spec.merged_x_labels)
            append_merged_surface_syndrome_round(
                circuit,
                spec=spec,
                registers=qregs,
                surface_qubits=surface_qubits,
                z_classical_bits=merged_z_bits[z_start : z_start + len(spec.z_labels)],
                x_classical_bits=merged_x_bits[x_start : x_start + len(spec.merged_x_labels)],
                round_label=f"merged interval round {round_index + 1}",
                add_barriers=False,
            )

        if include_steane_verification and round_index < len(STEANE_LS_VERIFICATION_ROUNDS):
            round_width = len(STEANE_LS_VERIFICATION_ROUNDS[round_index])
            append_steane_lattice_surgery_verification_round(
                circuit,
                registers=qregs,
                classical_bits=steane_verify_bits[
                    steane_verify_start : steane_verify_start + round_width
                ],
                round_index=round_index,
                add_barriers=False,
            )
            steane_verify_start += round_width

        if add_barriers:
            circuit.barrier(label=f"merged interval round {round_index + 1}")

    # Complete teleportation by Steane demolition, then restore top-edge X checks.
    append_measure_steane_data_in_x_basis(
        circuit,
        registers=qregs,
        classical_bits=list(cregs["steane_x"]),
        add_barriers=add_barriers,
    )
    if spec.top_x_recovery_checks:
        append_surface_top_x_recovery_measurements(
            circuit,
            spec=spec,
            registers=qregs,
            classical_bits=list(cregs["surface_x_recovery"]),
            add_barriers=add_barriers,
        )

    # This fixed-patch QEC extension is not part of the paper-facing transfer default.
    if include_surface_qec:
        post_z_bits = list(cregs["surface_post_z"])
        post_x_bits = list(cregs["surface_post_x"])
        for round_index in range(surface_qec_rounds):
            z_start = round_index * len(spec.z_labels)
            x_start = round_index * len(spec.x_labels)
            spec.append_stabilizer_round(
                circuit,
                surface_qubits=surface_qubits,
                z_ancillas=[qregs.surface_z_ancillas[label][0] for label in spec.z_ancilla_labels],
                x_ancillas=[qregs.surface_x_ancillas[label][0] for label in spec.x_ancilla_labels],
                z_bits=post_z_bits[z_start : z_start + len(spec.z_labels)],
                x_bits=post_x_bits[x_start : x_start + len(spec.x_labels)],
                round_label=f"post-transfer surface round {round_index + 1}",
                prepare_z_ancillas=True,
                prepare_x_ancillas=True,
                add_barriers=add_barriers,
            )

    if surface_readout_basis is None:
        return circuit

    readout_register = cregs[
        (
            f"surface_{surface_readout_basis.lower()}"
            if surface_readout_mode == "data"
            else f"surface_logical_{surface_readout_basis.lower()}"
        )
    ]
    if surface_readout_mode == "data":
        append_surface_logical_readout(
            circuit,
            spec=spec,
            registers=qregs,
            classical_bits=list(readout_register),
            basis=surface_readout_basis,
            add_barriers=add_barriers,
        )
    elif surface_readout_mode == "logical_pauli":
        append_surface_logical_pauli_measurement(
            circuit,
            spec=spec,
            registers=qregs,
            classical_bit=list(readout_register),
            basis=surface_readout_basis,
            destructive=True,
            add_barriers=add_barriers,
        )
    else:
        append_surface_logical_pauli_measurement(
            circuit,
            spec=spec,
            registers=qregs,
            classical_bit=readout_register[0],
            basis=surface_readout_basis,
            destructive=False,
            add_barriers=add_barriers,
        )
    return circuit
