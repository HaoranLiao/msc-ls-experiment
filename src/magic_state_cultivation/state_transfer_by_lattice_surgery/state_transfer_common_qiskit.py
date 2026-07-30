"""Shared Qiskit helpers for Steane-to-rotated-surface state transfer.

The distance-specific modules (`state_transfer_to_d3_rotated_qiskit` and
`state_transfer_to_d5_rotated_qiskit`) define the target patch geometry and
which checks exist on that patch.  This module keeps the target-independent
parts in one place:

- Hirano/paper labels for the Steane/color-code data qubits.
- Mapping between those labels and the 11-site Crumble encoder patch.
- Mixed Steane-surface parity measurements used by lattice surgery.
- Classical-register layout shared by circuit construction and parsing.
- The target-independent caller-owned appender and register-owning builder.

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

from magic_state_cultivation.steane_code import magic_state_cultivation_qiskit as cultivation
from magic_state_cultivation.steane_code import state_injection_qiskit as steane
from magic_state_cultivation.surface_code.post_transfer import rotated_post_transfer
from magic_state_cultivation.utils.circuit_qiskit import (
    resolve_logical_measurement_terms,
)
from magic_state_cultivation.utils.normalize import normalize_pauli_char

SurfaceReadoutBasis = Literal["X", "Y", "Z"]
LATTICE_SURGERY_ROUNDS = 3
MergedBoundaryCheck = tuple[str, tuple[str, ...], tuple[str, ...]]
CircuitAppender = Callable[..., None]

HIRANO_STEANE_DATA_ORDER: tuple[str, ...] = ("0", "1", "2", "3", "4", "5", "6")
HIRANO_STEANE_GEOMETRIC_ORDER: tuple[str, ...] = ("6", "4", "0", "2", "1", "5", "3")
HIRANO_STEANE_LOGICAL_X: tuple[str, ...] = ("1", "4", "6")
HIRANO_STEANE_LOGICAL_Z: tuple[str, ...] = ("1", "5", "3")

STEANE_LS_VERIFICATION_STABILIZER_INDICES: dict[str, int] = {
    "Z0145": 1,
    "Z0235": 2,
    "X0235": 2,
    "Z0246": 0,
    "X0246": 0,
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
FINAL_X_VERIFICATION_CHECKS: dict[str, tuple[str, ...]] = {
    "X0235": ("0", "2", "3", "5"),
    "X0246": ("0", "2", "4", "6"),
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
      across initialization, merged-interval checks, and top-edge recovery.
    - `merge_ancillas`: ancillas for mixed Steane-surface `ZZ` checks.
    """

    steane_data: dict[str, QuantumRegister]
    steane_helpers: dict[str, QuantumRegister]
    surface_data: dict[str, QuantumRegister]
    surface_z_ancillas: dict[str, QuantumRegister]
    surface_x_ancillas: dict[str, QuantumRegister]
    merge_ancillas: dict[str, QuantumRegister]


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
class SurfaceCodeOperations:
    """Distance-specific surface-code circuit operations used by transfer.

    Grouping these callables keeps their ownership explicit at shared call
    sites without adding transfer-level wrappers around the surface modules.
    """

    prepare_product_plus: CircuitAppender
    append_scheduled_stabilizer_round: CircuitAppender
    append_logical_measurement: CircuitAppender


@dataclass(frozen=True)
class SurfaceCodeTransferSpec:
    """Distance-specific data and circuit operations for one transfer target.

    The d=3 and d=5 transfer modules differ only in patch geometry, logical
    supports, merged-boundary checks, and top-edge X checks.  Binding those
    values once avoids maintaining two nearly identical protocol builders.
    Distance-specific circuit callables are namespaced under
    ``surface_code_operations``.

    ``top_x_recovery_labels`` preserves measurement order using paper-style
    check labels such as ``X_ab``. The canonical supports and schedules remain
    owned by the distance-specific surface-code module.
    """

    name: str
    distance: int
    surface_order: tuple[str, ...]
    logical_x: tuple[str, ...]
    logical_z: tuple[str, ...]
    z_stabilizers: Mapping[str, tuple[str, ...]]
    x_stabilizers: Mapping[str, tuple[str, ...]]
    merged_boundary_checks: tuple[MergedBoundaryCheck, ...]
    top_x_recovery_labels: tuple[str, ...]
    surface_code_operations: SurfaceCodeOperations

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
    def z_ancilla_labels(self) -> tuple[str, ...]:
        """Qubit-register labels used for surface Z-check ancillas."""
        return tuple(label.lower() for label in self.z_labels)

    @property
    def x_ancilla_labels(self) -> tuple[str, ...]:
        """Qubit-register labels used for surface X-check ancillas."""
        return tuple(label.lower() for label in self.x_labels)


def build_transfer_qregs(
    spec: SurfaceCodeTransferSpec,
) -> HiranoLabeledQRegs:
    """Create all quantum registers required by the paper transfer."""
    return HiranoLabeledQRegs(
        steane_data={label: QuantumRegister(1, label) for label in HIRANO_STEANE_DATA_ORDER},
        steane_helpers={label: QuantumRegister(1, label) for label in STEANE_HELPER_PATCH_LABELS},
        surface_data={label: QuantumRegister(1, label) for label in spec.surface_order},
        surface_z_ancillas={label: QuantumRegister(1, label) for label in spec.z_ancilla_labels},
        surface_x_ancillas={label: QuantumRegister(1, label) for label in spec.x_ancilla_labels},
        merge_ancillas={
            f"zz_{name[1:].lower()}": QuantumRegister(1, f"zz_{name[1:].lower()}")
            for name, _, _ in spec.merged_boundary_checks
        },
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


def round_labels(labels: Sequence[str]) -> tuple[str, ...]:
    """Prefix repeated-round labels with ``roundN_`` for flat registers."""
    return tuple(
        f"round{round_index + 1}_{label}"
        for round_index in range(LATTICE_SURGERY_ROUNDS)
        for label in labels
    )


def transfer_classical_register_specs(
    spec: SurfaceCodeTransferSpec,
    *,
    include_cultivation: bool,
    surface_readout_basis: SurfaceReadoutBasis | None,
    include_post_transfer_stabilizer_rounds: bool = False,
) -> list[ClassicalRegisterSpec]:
    """Return the classical registers for the fixed paper transfer sequence."""
    specs: list[ClassicalRegisterSpec] = [
        ClassicalRegisterSpec(
            "enc_helpers",
            ("enc_h1", "enc_h2", "enc_h6", "enc_h7"),
        )
    ]
    specs.append(
        ClassicalRegisterSpec(
            "steane_qed",
            cultivation.POST_INJECTION_QED_STABILIZER_LABELS,
        )
    )
    if include_cultivation:
        specs.append(
            ClassicalRegisterSpec(
                "cult",
                ("root_x", "helper_h1", "helper_h2", "helper_h6", "helper_h7"),
            )
        )
    specs.extend(
        [
            ClassicalRegisterSpec("surface_init_z", spec.z_labels),
            ClassicalRegisterSpec(
                "ls_zz",
                round_labels(tuple(name for name, _, _ in spec.merged_boundary_checks)),
            ),
            ClassicalRegisterSpec("merged_surface_z", round_labels(spec.z_labels)),
            ClassicalRegisterSpec("merged_surface_x", round_labels(spec.merged_x_labels)),
            ClassicalRegisterSpec("steane_verify", STEANE_LS_VERIFICATION_LABELS),
            ClassicalRegisterSpec("steane_x", HIRANO_STEANE_DATA_ORDER),
            ClassicalRegisterSpec("surface_x_recovery", spec.top_x_recovery_labels),
        ]
    )
    if include_post_transfer_stabilizer_rounds:
        specs.extend(
            [
                ClassicalRegisterSpec(
                    "post_transfer_surface_z",
                    rotated_post_transfer.repeated_round_labels(
                        spec.z_labels,
                        rounds=spec.distance,
                    ),
                ),
                ClassicalRegisterSpec(
                    "post_transfer_surface_x",
                    rotated_post_transfer.repeated_round_labels(
                        spec.x_labels,
                        rounds=spec.distance,
                    ),
                ),
            ]
        )
    if surface_readout_basis is not None:
        surface_readout_basis = normalize_pauli_char(surface_readout_basis)
        measurement_terms, _ = resolve_logical_measurement_terms(
            surface_readout_basis,
            logical_x_support=spec.logical_x,
            logical_z_support=spec.logical_z,
        )
        selected_support = tuple(support_item for support_item, _ in measurement_terms)
        specs.append(
            ClassicalRegisterSpec(
                f"surface_logical_{surface_readout_basis.lower()}",
                tuple(selected_support),
            )
        )
    return specs


def append_boundary_zz_round(
    circuit: QuantumCircuit,
    *,
    registers: HiranoLabeledQRegs,
    classical_bits: Sequence,
    merged_boundary_checks: Sequence[tuple[str, tuple[str, ...], tuple[str, ...]]],
    add_barriers: bool = True,
) -> None:
    """Append one round of merged-boundary ``ZZ`` checks.

    Each check may involve Steane/color-code data labels, surface-code labels,
    or both.  The d=5 paper schedule uses `Z1a`, `Z35bc`, and `Zde`; the d=3
    adaptation uses the subset available on the smaller top boundary.

    Their parity is the logical `Z_L Z_L` measurement. A nonzero outcome becomes
    an X byproduct on the target surface-code Pauli frame. The protocol builder
    calls this helper three times and compares outcomes to detect measurement
    disagreement.
    """
    if len(classical_bits) != len(merged_boundary_checks):
        raise ValueError(
            f"classical_bits must contain {len(merged_boundary_checks)} merged-boundary bits."
        )

    for (check_name, steane_labels, surface_labels), classical_bit in zip(
        merged_boundary_checks,
        classical_bits,
        strict=True,
    ):
        data_qubits = [registers.steane_data[label][0] for label in steane_labels] + [
            registers.surface_data[label][0] for label in surface_labels
        ]
        ancilla = registers.merge_ancillas[f"zz_{check_name[1:].lower()}"][0]
        circuit.reset(ancilla)
        for qubit in data_qubits:
            circuit.cx(qubit, ancilla)
        circuit.measure(ancilla, classical_bit)
    if add_barriers:
        circuit.barrier(label="merged ZZ checks")


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

    steane.append_patch_logical_measurement(
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
        steane.append_patch_stabilizer_measurement(
            circuit,
            patch_qubits=patch_qubits,
            classical_bits=[classical_bit],
            basis=check_name[0].lower(),
            stabilizer_indices=(STEANE_LS_VERIFICATION_STABILIZER_INDICES[check_name],),
            add_barriers=False,
        )

    if add_barriers:
        circuit.barrier(label=f"Steane LS verification round {round_index + 1}")


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
    """Measure all merge-compatible surface checks in one four-step round.

    Every ordinary Z check is interleaved with the compatible X checks under
    their shared canonical schedule. The top-edge X checks listed by
    ``spec.top_x_recovery_labels`` anticommute with the merged-boundary Z
    checks, so they are withheld until the Steane patch has been demolished.
    """
    spec.surface_code_operations.append_scheduled_stabilizer_round(
        circuit,
        surface_qubits=surface_qubits,
        selected_stabilizer_names=spec.z_labels + spec.merged_x_labels,
        ancillas=[registers.surface_z_ancillas[label.lower()][0] for label in spec.z_labels]
        + [registers.surface_x_ancillas[label.lower()][0] for label in spec.merged_x_labels],
        classical_bits=[*z_classical_bits, *x_classical_bits],
        round_label=f"{round_label} compatible surface checks",
        reset_ancillas_initially=True,
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
    """Recover surface top-edge X checks in parallel after Steane demolition."""
    if len(classical_bits) < len(spec.top_x_recovery_labels):
        raise ValueError(
            "classical_bits must contain "
            f"{len(spec.top_x_recovery_labels)} top-edge X recovery bits."
        )

    spec.surface_code_operations.append_scheduled_stabilizer_round(
        circuit,
        surface_qubits=surface_data_qubits(registers, spec.surface_order),
        selected_stabilizer_names=spec.top_x_recovery_labels,
        ancillas=[
            registers.surface_x_ancillas[label.lower()][0] for label in spec.top_x_recovery_labels
        ],
        classical_bits=classical_bits,
        round_label="recover surface top edge",
        reset_ancillas_initially=False,
        add_barriers=add_barriers,
    )


def append_lattice_surgery_circuit(
    circuit: QuantumCircuit,
    *,
    spec: SurfaceCodeTransferSpec,
    qregs: HiranoLabeledQRegs,
    cregs: Mapping[str, ClassicalRegister],
    injected_state: steane.InjectedStateName = "T",
    include_cultivation: bool = True,
    include_post_transfer_stabilizer_rounds: bool = False,
    surface_readout_basis: SurfaceReadoutBasis | None = None,
    add_barriers: bool = True,
) -> None:
    """Append the complete transfer sequence to caller-owned registers.

    The circuit implements the fixed-patch lattice-surgery sequence from
    Section III A of Hirano et al., specialized by ``spec`` to a rotated d=3
    or d=5 target. It appends these stages in protocol order:

    1. Encode ``injected_state`` on the 11-site Steane patch with the routed
       Crumble encoder and measure its helper-qubit acceptance bits.
    2. Measure every Steane X and Z stabilizer and use the six
       outcomes for post-injection error detection and postselection.
    3. Optionally apply the Steane logical-H_XY cultivation projection.
    4. Prepare every target surface-data qubit in ``|+>`` and measure every
       surface Z stabilizer once, projecting the target into logical
       ``|+_L>`` and establishing the pre-merge Z-syndrome reference.
    5. Perform exactly three merged rounds. Every round measures the mixed
       boundary-Z checks, every compatible surface Z check, and every
       compatible surface X check. The paper-defined Steane verification
       groups are additionally measured in rounds 1 and 2.
    6. Demolish the Steane data in X after round 3 and recover the surface
       top-edge X checks that were withheld during the merge.
    7. Optionally append ``distance`` full surface-stabilizer rounds.
    8. Optionally append a destructive surface logical readout for analysis.

    Parameters
    ----------
    circuit
        Existing circuit containing every quantum and classical register used
        by the transfer.
    spec
        Target geometry, logical supports, stabilizer definitions,
        merged-boundary checks, withheld top-edge checks, and distance-specific
        surface-code circuit operations.
    qregs
        Caller-owned paper-labeled quantum registers returned by
        ``build_transfer_qregs(spec)``.
    cregs
        Caller-owned classical registers keyed by the names returned from
        ``transfer_classical_register_specs``.
    injected_state
        Named physical state passed to the Steane Crumble encoder. The default
        ``"T"`` prepares the H_XY-axis magic-state input used by cultivation.
    include_cultivation
        Whether to append the logical-H_XY projection and its classical
        postselection register. Disable this only for transfer plumbing tests
        or when the input has already been cultivated.
    include_post_transfer_stabilizer_rounds
        Whether to append ``spec.distance`` full surface X/Z stabilizer rounds
        after top-edge recovery and before any logical readout.
    surface_readout_basis
        Optional ``"X"``, ``"Y"``, or ``"Z"`` analysis basis. ``None`` ends
        the appended sequence immediately after top-edge recovery, matching
        the paper-facing transfer boundary. Y uses the union of the target's
        standard logical-X and logical-Z supports. This terminal destructive
        measurement is an analysis addition, not a lattice-surgery round.
    add_barriers
        Whether to add labeled barriers between conceptual protocol stages.
        Barriers improve circuit inspection but do not change the measured
        observables.

    Notes
    -----
    The three merged rounds contain temporal surface-code syndrome extraction,
    not in-circuit decoding. After circuit execution,
    ``parse_transfer_measurements`` establishes the known X-check preparation
    boundary, uses the pre-merge Z-stabilizer measurements as the Z-check
    reference, and decodes the resulting surface detector layers with
    ``TemporalRotatedDecoder``. Repeated merged-boundary changes are strict
    postselection failures, not decoder input. After demolition, the restored
    top-edge X values complete the surface stabilizer reference for future
    fixed-patch continuation. Corrections are tracked classically in a Pauli
    frame rather than applied as physical gates.

    The boundary, surface-syndrome, and Steane-verification measurement groups
    are appended sequentially within each merged round. The compatible surface
    X and Z checks share their canonical four-step schedule. This appender does
    not reproduce the paper's hardware-level superdense interleaving,
    circuit-level detector error model, or later code expansion. Optional
    fixed-patch stabilization is delegated to
    ``surface_code.post_transfer.rotated_post_transfer``.
    """
    # Prepare the accepted color-code input through the shared Steane front end.
    patch_qubits = hirano_labeled_patch_qubits(qregs)
    cultivation.append_steane_injection_and_cultivation(
        circuit,
        patch_qubits=patch_qubits,
        injected_state=injected_state,
        encoder_classical_bits=list(cregs["enc_helpers"]),
        qed_classical_bits=list(cregs["steane_qed"]),
        cultivation_classical_bits=([list(cregs["cult"])] if include_cultivation else []),
        cultivation_rounds=(1 if include_cultivation else 0),
        add_barriers=add_barriers,
    )

    # Prepare the target in logical |+_L>. The product |+> state fixes every X
    # check to +1; measuring every Z stabilizer projects into the surface-code
    # space and establishes the Z reference before lattice surgery begins.
    surface_qubits = surface_data_qubits(qregs, spec.surface_order)
    spec.surface_code_operations.prepare_product_plus(
        circuit,
        qubits=surface_qubits,
        add_barriers=add_barriers,
        barrier_label=f"prepare {spec.name} |+> product state",
    )
    spec.surface_code_operations.append_scheduled_stabilizer_round(
        circuit,
        surface_qubits=surface_qubits,
        selected_stabilizer_names=spec.z_labels,
        ancillas=[qregs.surface_z_ancillas[label.lower()][0] for label in spec.z_labels],
        classical_bits=list(cregs["surface_init_z"]),
        round_label=f"initialize {spec.name} |+_L>",
        reset_ancillas_initially=True,
        add_barriers=False,
    )
    if add_barriers:
        circuit.barrier(label=f"initialize {spec.name} |+_L>")

    # Run merged-code rounds, including the two paper-defined Steane check groups.
    merged_zz_bits = list(cregs["ls_zz"])
    merged_z_bits = list(cregs["merged_surface_z"])
    merged_x_bits = list(cregs["merged_surface_x"])
    steane_verify_bits = list(cregs["steane_verify"])
    steane_verify_start = 0
    for round_index in range(LATTICE_SURGERY_ROUNDS):
        zz_start = round_index * len(spec.merged_boundary_checks)
        append_boundary_zz_round(
            circuit,
            registers=qregs,
            classical_bits=merged_zz_bits[zz_start : zz_start + len(spec.merged_boundary_checks)],
            merged_boundary_checks=spec.merged_boundary_checks,
            add_barriers=False,
        )

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

        if round_index < len(STEANE_LS_VERIFICATION_ROUNDS):
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
    append_surface_top_x_recovery_measurements(
        circuit,
        spec=spec,
        registers=qregs,
        classical_bits=list(cregs["surface_x_recovery"]),
        add_barriers=add_barriers,
    )

    rotated_post_transfer.append_rotated_post_transfer(
        circuit,
        surface_qubits=surface_data_qubits(qregs, spec.surface_order),
        z_ancillas=[qregs.surface_z_ancillas[label.lower()][0] for label in spec.z_labels],
        x_ancillas=[qregs.surface_x_ancillas[label.lower()][0] for label in spec.x_labels],
        z_stabilizer_names=spec.z_labels,
        x_stabilizer_names=spec.x_labels,
        z_classical_bits=(
            list(cregs["post_transfer_surface_z"])
            if include_post_transfer_stabilizer_rounds
            else ()
        ),
        x_classical_bits=(
            list(cregs["post_transfer_surface_x"])
            if include_post_transfer_stabilizer_rounds
            else ()
        ),
        stabilizer_rounds=(spec.distance if include_post_transfer_stabilizer_rounds else 0),
        append_scheduled_stabilizer_round=(
            spec.surface_code_operations.append_scheduled_stabilizer_round
        ),
        logical_readout_basis=surface_readout_basis,
        logical_x=spec.logical_x,
        logical_z=spec.logical_z,
        logical_classical_bits=(
            list(cregs[f"surface_logical_{surface_readout_basis.lower()}"])
            if surface_readout_basis is not None
            else ()
        ),
        append_logical_measurement=(spec.surface_code_operations.append_logical_measurement),
        add_barriers=add_barriers,
    )


def build_lattice_surgery_circuit(
    spec: SurfaceCodeTransferSpec,
    *,
    injected_state: steane.InjectedStateName = "T",
    include_cultivation: bool = True,
    include_post_transfer_stabilizer_rounds: bool = False,
    surface_readout_basis: SurfaceReadoutBasis | None = None,
    add_barriers: bool = True,
) -> QuantumCircuit:
    """This is the authoritative transfer-register allocation and appending example."""
    # Quantum allocation uses one single-qubit qreg per physical site:
    # - 7 Steane data qregs and 4 Steane helper/routing qregs;
    # - distance**2 surface-data qregs (9 for d=3, 25 for d=5);
    # - one ancilla qreg per surface Z or X stabilizer
    #   (4 Z + 4 X for d=3, 12 Z + 12 X for d=5);
    # - one merge-ancilla qreg per merged-boundary check
    #   (2 for d=3, 3 for d=5).
    qregs = build_transfer_qregs(spec)

    # Classical allocation uses one creg per measurement family:
    # - enc_helpers: 4 state-injection encoder bits;
    # - steane_qed: 6 post-injection Steane stabilizer bits;
    # - cult: 5 cultivation bits, when cultivation is enabled;
    # - surface_init_z: one bit per surface Z stabilizer
    #   (4 for d=3, 12 for d=5);
    # - ls_zz: one bit per merge check in each of the 3 merge rounds
    #   (6 for d=3, 9 for d=5);
    # - merged_surface_z: one bit per Z stabilizer in each merge round
    #   (12 for d=3, 36 for d=5);
    # - merged_surface_x: one bit per compatible X stabilizer in each merge round
    #   (9 for d=3, 30 for d=5);
    # - steane_verify: 8 bits across the two Steane verification groups;
    # - steane_x: 7 Steane demolition bits;
    # - surface_x_recovery: one bit per withheld top-edge X check
    #   (1 for d=3, 2 for d=5);
    # - post_transfer_surface_z/x: distance rounds of full syndromes, if enabled
    #   (12 Z + 12 X bits for d=3, 60 Z + 60 X bits for d=5);
    # - surface_logical_x/y/z: one bit per measured physical qubit, if requested
    #   (Y uses the union of the logical-X and logical-Z supports).
    classical_specs = transfer_classical_register_specs(
        spec,
        include_cultivation=include_cultivation,
        surface_readout_basis=surface_readout_basis,
        include_post_transfer_stabilizer_rounds=include_post_transfer_stabilizer_rounds,
    )
    cregs = {
        register_spec.name: ClassicalRegister(
            len(register_spec.labels),
            register_spec.name,
        )
        for register_spec in classical_specs
    }
    qreg_groups = (
        qregs.steane_data,
        qregs.steane_helpers,
        qregs.surface_data,
        qregs.surface_z_ancillas,
        qregs.surface_x_ancillas,
        qregs.merge_ancillas,
    )
    circuit = QuantumCircuit(
        *(register for group in qreg_groups for register in group.values()),
        *(cregs[register_spec.name] for register_spec in classical_specs),
    )
    append_lattice_surgery_circuit(
        circuit,
        spec=spec,
        qregs=qregs,
        cregs=cregs,
        injected_state=injected_state,
        include_cultivation=include_cultivation,
        include_post_transfer_stabilizer_rounds=include_post_transfer_stabilizer_rounds,
        surface_readout_basis=surface_readout_basis,
        add_barriers=add_barriers,
    )
    return circuit
