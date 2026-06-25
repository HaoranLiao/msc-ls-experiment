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

High-level circuit order
------------------------
The default builder follows the complete distance-3 adaptation used in this
repo:

1. prepare the Steane/color-code input, optionally including H-axis cultivation
2. initialize the rotated d=3 surface patch as a logical ``|+_L>`` candidate
3. repeat the merged-boundary ``ZZ`` checks
4. measure Steane-side LS verification checks
5. demolish the Steane block in the ``X`` basis
6. recover the top-edge ``X_ab`` surface stabilizer and reconstruct
   ``X0145ab``
7. run post-transfer rotated d=3 stabilizer rounds
8. optionally append a final surface logical readout

The parser mirrors this order.  It exposes raw register chunks as well as
semantic dictionaries so notebook code and tests can inspect every stage.

Layout conventions
------------------
The public transfer circuit uses Hirano-style labels in its Qiskit registers.
The Steane/color-code part is drawn in the same orientation as Fig. 6:

        6
     4     2
        0
     1  5  3

The repo's 11-site Steane cultivation patch has the same data-qubit shape but
also includes four helper sites.  Square brackets show the Hirano data label
stored at each patch site:

             0[6]

        1[h1]   2[h2]

    3[4]   4[0]   5[2]

        6[h6]   7[h7]

    8[1]   9[5]   10[3]

The rotated distance-3 surface patch is the top-left 3-by-3 portion of the
paper's distance-5 target:

    a  b  c
    d  e  f
    g  h  i

During the merged-boundary interval, the Steane bottom edge ``1,5,3`` sits
above the surface top edge ``a,b,c``.  This distance-3 adaptation keeps the
merged checks ``Z1a`` and ``Z35bc``; the paper's extra ``Zde`` check belongs to
the distance-5 target and is not present here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

from research_qec_haoran.decoding.pauli_frame import PauliFrame
from research_qec_haoran.decoding.stabilizer_frame import StabilizerFrame
from research_qec_haoran.steane_code.magic_state_cultivation_qiskit import (
    append_hadamard_axis_projection_round,
)
from research_qec_haoran.steane_code.state_injection_qiskit import (
    InjectedStateName,
    append_patch_steane_state_injection_encoder_crumble_sequences,
)
from research_qec_haoran.surface_code.decoding.rotated_decoder import RotatedDecoder
from research_qec_haoran.surface_code.rotated_d3_qiskit import (
    DEFAULT_LOGICAL_X,
    DEFAULT_LOGICAL_Z,
    SURFACE_ORDER,
    X_STABILIZERS,
    Z_STABILIZERS,
    append_stabilizer_round,
    append_z_stabilizer_measurements,
    prepare_plus,
)
from research_qec_haoran.surface_code.rotated_d3_qiskit import (
    append_logical_measurement as append_surface_code_logical_measurement,
)
from research_qec_haoran.utils.normalize import parse_qiskit_bitstring

# ===========================================================================
# Public type aliases
# ===========================================================================

SurfaceReadoutBasis = Literal["X", "Z"]
SurfaceReadoutMode = Literal["data", "logical_pauli", "logical_pauli_indirect"]


# ===========================================================================
# Protocol labels and stabilizer/check definitions
# ===========================================================================

# The circuit uses one-qubit Qiskit registers so rendered wires match the paper
# labels directly.  These labels are not the same as the repo's internal
# 11-site Steane patch labels; the conversion is defined below.
HIRANO_STEANE_DATA_ORDER: tuple[str, ...] = ("0", "1", "2", "3", "4", "5", "6")
HIRANO_STEANE_GEOMETRIC_ORDER: tuple[str, ...] = ("6", "4", "0", "2", "1", "5", "3")

# Paper Fig. 6 uses ``X146`` as the destructive color-code logical-X
# measurement outcome for the transfer feedforward.
HIRANO_STEANE_LOGICAL_X: tuple[str, ...] = ("1", "4", "6")
HIRANO_STEANE_LOGICAL_Z: tuple[str, ...] = ("1", "5", "3")

TOP_BOUNDARY: tuple[str, ...] = ("a", "b", "c")
SURFACE_X_AB_LABELS: tuple[str, ...] = ("X_ab",)
SURFACE_INIT_Z_LABELS: tuple[str, ...] = ("Z_abde", "Z_cf", "Z_dg", "Z_efhi")
SURFACE_POST_Z_LABELS: tuple[str, ...] = tuple(Z_STABILIZERS)
SURFACE_POST_X_LABELS: tuple[str, ...] = tuple(X_STABILIZERS)
SURFACE_X_ANCILLA_LABELS: tuple[str, ...] = tuple(
    name.lower() for name in SURFACE_POST_X_LABELS
)

# These are the Steane/color-code checks measured during the merged-code
# interval in the distance-3 adaptation.  The paper's costly six-body
# ``X0145ab`` is intentionally not measured here; it is reconstructed after
# destructive Steane X readout and top-edge ``X_ab`` recovery.
STEANE_LS_VERIFICATION_CHECKS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("Z0145", "Z", ("0", "1", "4", "5")),
    ("Z0235", "Z", ("0", "2", "3", "5")),
    ("X0235", "X", ("0", "2", "3", "5")),
    ("Z0246", "Z", ("0", "2", "4", "6")),
    ("X0246", "X", ("0", "2", "4", "6")),
)
STEANE_LS_VERIFICATION_LABELS: tuple[str, ...] = tuple(
    name for name, _, _ in STEANE_LS_VERIFICATION_CHECKS
)

# Supports used after Steane demolition.  ``X0145`` is listed separately so it
# can be combined with the recovered surface ``X_ab`` bit into ``X0145ab``.
STEANE_X_RECONSTRUCTED_CHECKS: dict[str, tuple[str, ...]] = {
    "X0235": ("0", "2", "3", "5"),
    "X0246": ("0", "2", "4", "6"),
    "X0145": ("0", "1", "4", "5"),
}

# Local deterministic acceptance checks for the current d=3 adaptation.  These
# names intentionally match the paper-style checks exposed to notebooks.
LOCAL_STEANE_X_STABILIZER_CHECKS: dict[str, tuple[str, ...]] = {
    "X0246": ("0", "2", "4", "6"),
    "X0235": ("0", "2", "3", "5"),
}
LOCAL_X_AB_CORRELATION_CHECKS: dict[str, tuple[str, ...]] = {
    "X0145ab": ("0", "1", "4", "5", "X_ab"),
}


# ===========================================================================
# Label conversion between repo patch sites and paper/Hirano labels
# ===========================================================================

# Map the repo's 11-site cultivation patch labels onto Hirano's 7 color-code
# data labels.  The data rows match the paper's orientation:
#
#     Hirano data labels:             Repo patch sites:
#
#             6                              0
#          4     2                        3     5
#             0                              4
#          1  5  3                        8  9  10
#
# The full 11-site patch additionally has helper sites 1, 2, 6, and 7 between
# those rows.  These helpers are kept in separate one-qubit registers named
# h1, h2, h6, and h7.

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


# ===========================================================================
# Register description data classes
# ===========================================================================

@dataclass(frozen=True)
class HiranoLabeledQRegs:
    """Paper-labeled qubit registers used by the transfer circuit.

    Every data qubit is placed in its own one-qubit register so Qiskit drawings
    show labels such as ``0`` and ``a`` directly.  The dictionaries group those
    one-qubit registers by protocol role:

    - Steane data/helper registers hold the 11-site cultivation patch.
    - Steane verification ancillas measure color-code checks during surgery.
    - Surface data and ancillas hold the rotated d=3 target patch.
    - Merge ancillas measure boundary ``ZZ`` checks.
    - Logical-measurement ancillas are optional readout helpers.
    """

    steane_data: dict[str, QuantumRegister]
    steane_helpers: dict[str, QuantumRegister]
    steane_verification_ancillas: dict[str, QuantumRegister]
    surface_data: dict[str, QuantumRegister]
    surface_z_ancillas: dict[str, QuantumRegister]
    surface_x_ancillas: dict[str, QuantumRegister]
    merge_ancillas: dict[str, QuantumRegister]
    logical_measurement_ancillas: dict[str, QuantumRegister]


@dataclass(frozen=True)
class ClassicalRegisterSpec:
    """A human-readable description of one Qiskit classical register.

    Qiskit count keys are ordered by classical register, but the raw bitstrings
    do not retain semantic labels.  The parser rebuilds that semantic mapping
    from these specs.
    """

    name: str
    labels: tuple[str, ...]


def build_hirano_labeled_qregs(
    *,
    include_steane_verification: bool = False,
    include_noisy_verification: bool = False,
    include_surface_qec: bool = False,
    surface_logical_measurement_basis: SurfaceReadoutBasis | None = None,
) -> HiranoLabeledQRegs:
    """Create one-qubit Qiskit registers with Hirano-style labels.

    Parameters choose which optional ancilla groups are allocated.  This keeps
    compact diagnostic circuits small while the default full transfer circuit
    gets the ancillas it needs for verification and post-transfer QEC.
    """
    steane_data = {
        label: QuantumRegister(1, label)
        for label in HIRANO_STEANE_DATA_ORDER
    }
    steane_helpers = {
        label: QuantumRegister(1, label)
        for label in STEANE_HELPER_PATCH_LABELS
    }
    steane_verification_ancillas = (
        {
            f"steane_{label.lower()}": QuantumRegister(1, f"steane_{label.lower()}")
            for label in STEANE_LS_VERIFICATION_LABELS
        }
        if include_steane_verification
        else {}
    )
    surface_data = {
        label: QuantumRegister(1, label)
        for label in SURFACE_ORDER
    }
    surface_z_ancillas = {
        label: QuantumRegister(1, label)
        for label in SURFACE_Z_ANCILLA_LABELS
    }
    # ``x_ab`` is used both by the paper-style final X-check reconstruction and
    # by the full surface X-stabilizer schedule.  Build a set first so this
    # shared ancilla is allocated once when both features are enabled.
    surface_x_labels = set()
    if include_noisy_verification:
        surface_x_labels.add("x_ab")
    if include_surface_qec:
        surface_x_labels.update(SURFACE_X_ANCILLA_LABELS)
    surface_x_ancillas = {
        label: QuantumRegister(1, label)
        for label in SURFACE_X_ANCILLA_LABELS
        if label in surface_x_labels
    }
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
        steane_verification_ancillas=steane_verification_ancillas,
        surface_data=surface_data,
        surface_z_ancillas=surface_z_ancillas,
        surface_x_ancillas=surface_x_ancillas,
        merge_ancillas=merge_ancillas,
        logical_measurement_ancillas=logical_measurement_ancillas,
    )


# ===========================================================================
# Qubit and classical-bit mapping helpers
# ===========================================================================

def _hirano_labeled_patch_qubits(registers: HiranoLabeledQRegs) -> list:
    """Return the 11-site cultivation patch qubits in patch-label order.

    Existing Steane injection/cultivation helpers operate on the repo's
    11-site triangular patch indexing.  The transfer circuit exposes paper
    labels externally, so this function is the adapter between those views.
    """
    patch_qubits: list[object] = [None] * 11
    for site, label in GOOGLE_SITE_TO_HIRANO_LABEL.items():
        patch_qubits[site] = registers.steane_data[label][0]
    for label, site in STEANE_HELPER_LABEL_TO_GOOGLE_SITE.items():
        patch_qubits[site] = registers.steane_helpers[label][0]
    return patch_qubits


def _surface_data_qubits(registers: HiranoLabeledQRegs) -> list:
    """Return the rotated-d3 surface data qubits in ``a..i`` order.

    The rotated surface-code helpers use ``SURFACE_ORDER`` as their positional
    API.  The transfer circuit stores the same qubits by label in dictionaries.
    """
    return [registers.surface_data[label][0] for label in SURFACE_ORDER]


def _parse_register_chunk(chunk: str, labels: tuple[str, ...]) -> dict[str, int]:
    """Map one Qiskit classical-register chunk back to semantic bit labels.

    Qiskit prints each classical register with most-significant bit first, so
    parsing by label requires reversing the chunk before zipping with labels.
    """
    return parse_qiskit_bitstring(chunk, labels)


def _parity(bits: dict[str, int], labels: tuple[str, ...]) -> int:
    """Return the mod-2 parity over a labeled bit dictionary."""
    return sum(bits[label] for label in labels) % 2


def _merged_zz_labels(merged_rounds: int) -> tuple[str, ...]:
    """Return labels for all repeated merged-boundary check outcomes."""
    return tuple(
        f"round{round_index + 1}_{check_name}"
        for round_index in range(merged_rounds)
        for check_name, _, _ in MERGED_BOUNDARY_CHECKS
    )


def _round_labels(rounds: int, labels: tuple[str, ...]) -> tuple[str, ...]:
    """Prefix repeated-round labels with ``roundN_`` for flat registers."""
    return tuple(
        f"round{round_index + 1}_{label}"
        for round_index in range(rounds)
        for label in labels
    )


def _surface_logical_readout_label(basis: SurfaceReadoutBasis) -> str:
    """Return the combined one-bit logical-readout label for indirect mode."""
    support = DEFAULT_LOGICAL_X if basis.upper() == "X" else DEFAULT_LOGICAL_Z
    return f"{basis.upper()}_{''.join(support)}"


# ===========================================================================
# Classical-register layout
# ===========================================================================

def _classical_register_specs(
    *,
    include_cultivation: bool,
    merged_rounds: int,
    noisy: bool,
    include_steane_verification: bool,
    include_surface_qec: bool,
    surface_qec_rounds: int,
    surface_readout_basis: SurfaceReadoutBasis | None,
    surface_readout_mode: SurfaceReadoutMode,
) -> list[ClassicalRegisterSpec]:
    """Return the classical-register layout for a circuit configuration.

    Builder and parser both call this helper.  Keeping the register order in
    one place prevents count-key parsing bugs when optional stages are enabled
    or disabled.
    """
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
        ]
    )
    if include_steane_verification:
        specs.append(ClassicalRegisterSpec("steane_verify", STEANE_LS_VERIFICATION_LABELS))
    specs.append(ClassicalRegisterSpec("steane_x", HIRANO_STEANE_DATA_ORDER))
    if noisy or include_steane_verification:
        specs.append(ClassicalRegisterSpec("surface_x_recovery", SURFACE_X_AB_LABELS))
    if include_surface_qec:
        specs.extend(
            [
                ClassicalRegisterSpec(
                    "surface_post_z",
                    _round_labels(surface_qec_rounds, SURFACE_POST_Z_LABELS),
                ),
                ClassicalRegisterSpec(
                    "surface_post_x",
                    _round_labels(surface_qec_rounds, SURFACE_POST_X_LABELS),
                ),
            ]
        )
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


# ===========================================================================
# Generic Pauli-parity measurement primitives
# ===========================================================================

def append_measure_z_parity(
    circuit: QuantumCircuit,
    *,
    data_qubits,
    ancilla,
    classical_bit,
) -> None:
    """Measure the Z-basis parity of data qubits with an ancilla.

    The ancilla is reset internally because the same one-qubit register can be
    reused across repeated check rounds.  Without the reset, repeated parity
    measurements would accumulate the previous round's result.
    """
    circuit.reset(ancilla)
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
    """Measure the X-basis parity of data qubits with an ancilla.

    This is the usual X-check pattern: prepare/rotate the ancilla into the
    X-basis interaction frame, fan out CNOTs from ancilla to data, rotate back,
    and measure the ancilla in Z.  The initial reset makes repeated use safe.
    """
    circuit.reset(ancilla)
    circuit.h(ancilla)
    for qubit in data_qubits:
        circuit.cx(ancilla, qubit)
    circuit.h(ancilla)
    circuit.measure(ancilla, classical_bit)


# ===========================================================================
# Protocol circuit fragments
# ===========================================================================

def append_boundary_zz_checks(
    circuit: QuantumCircuit,
    *,
    registers: HiranoLabeledQRegs,
    classical_bits: list,
    merged_rounds: int = 1,
    add_barriers: bool = True,
) -> None:
    """Append repeated merged-boundary ``ZZ`` checks for the d=3 adaptation.

    The paper's distance-5 target has three merged checks: ``Z1a``, ``Z35bc``,
    and ``Zde``.  The current d=3 target has only the available top-boundary
    checks ``Z1a`` and ``Z35bc``.  Repetition lets the parser detect unstable
    merged-check histories in the same spirit as the paper's three rounds.
    """
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
    """Demolish the Steane block by measuring all seven data qubits in ``X``.

    These destructive physical measurements serve two purposes:

    - their ``X146`` parity is the logical-X transfer feedforward bit
    - their stabilizer parities reconstruct final color-code X checks
    """
    if len(classical_bits) != len(HIRANO_STEANE_DATA_ORDER):
        raise ValueError("classical_bits must contain seven Steane-data bits.")

    for label, classical_bit in zip(HIRANO_STEANE_DATA_ORDER, classical_bits, strict=False):
        qubit = registers.steane_data[label][0]
        circuit.h(qubit)
        circuit.measure(qubit, classical_bit)

    if add_barriers:
        circuit.barrier(label="measure Steane data in X")


def append_steane_lattice_surgery_verification(
    circuit: QuantumCircuit,
    *,
    registers: HiranoLabeledQRegs,
    classical_bits: list,
    add_barriers: bool = True,
) -> None:
    """Measure the d=3 color-code checks used for LS postselection.

    This is the distance-3 specialization described around Fig. 6:
    measure ``Z0145``, ``Z0235``, ``X0235``, ``Z0246``, and ``X0246`` during
    the merged-code interval.  The skipped check ``X0145`` is recovered later
    as ``X0145ab`` after measuring the surface top-edge ``X_ab`` check.
    """
    if len(classical_bits) != len(STEANE_LS_VERIFICATION_CHECKS):
        raise ValueError(
            f"classical_bits must contain {len(STEANE_LS_VERIFICATION_CHECKS)} bits."
        )

    for (check_name, basis, support), classical_bit in zip(
        STEANE_LS_VERIFICATION_CHECKS, classical_bits, strict=True
    ):
        ancilla_name = f"steane_{check_name.lower()}"
        if ancilla_name not in registers.steane_verification_ancillas:
            raise ValueError(f"registers do not include {ancilla_name}.")
        data_qubits = [registers.steane_data[label][0] for label in support]
        if basis == "Z":
            append_measure_z_parity(
                circuit,
                data_qubits=data_qubits,
                ancilla=registers.steane_verification_ancillas[ancilla_name][0],
                classical_bit=classical_bit,
            )
        else:
            append_measure_x_parity(
                circuit,
                data_qubits=data_qubits,
                ancilla=registers.steane_verification_ancillas[ancilla_name][0],
                classical_bit=classical_bit,
            )

    if add_barriers:
        circuit.barrier(label="Steane LS verification")


def append_surface_logical_readout(
    circuit: QuantumCircuit,
    *,
    registers: HiranoLabeledQRegs,
    classical_bits: list,
    basis: SurfaceReadoutBasis,
    add_barriers: bool = True,
) -> None:
    """Measure all rotated-d3 data qubits in ``basis`` for logical readout.

    This is the broadest destructive readout mode.  The parser later computes
    the logical parity from the default weight-3 surface-code representative.
    """
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
    """Recover the top-boundary ``X_ab`` stabilizer used by postselection.

    After Steane demolition, ``X0145`` is not directly available as a
    non-destructive check.  Combining the destructive Steane parity ``X0145``
    with this recovered ``X_ab`` bit yields the paper-style ``X0145ab`` check.
    """
    if "x_ab" not in registers.surface_x_ancillas:
        raise ValueError("registers do not include the x_ab recovery ancilla.")
    append_surface_code_logical_measurement(
        circuit,
        "X",
        surface_qubits=_surface_data_qubits(registers),
        support=("a", "b"),
        ancilla_qubit=registers.surface_x_ancillas["x_ab"][0],
        classical_bits=[classical_bit],
        destructive=False,
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
    """Measure the weight-3 rotated-d3 logical Pauli directly with an ancilla.

    This non-destructive mode is useful for notebook diagnostics when the user
    wants a single combined logical bit without demolishing the whole surface
    patch.
    """
    normalized_basis = basis.upper()
    if normalized_basis not in {"X", "Z"}:
        raise ValueError("basis must be 'X' or 'Z'.")

    ancilla_name = f"logical_{normalized_basis.lower()}"
    if ancilla_name not in registers.logical_measurement_ancillas:
        raise ValueError(f"registers do not include the {ancilla_name} measurement ancilla.")

    support = DEFAULT_LOGICAL_X if normalized_basis == "X" else DEFAULT_LOGICAL_Z
    append_surface_code_logical_measurement(
        circuit,
        normalized_basis,
        surface_qubits=_surface_data_qubits(registers),
        support=support,
        ancilla_qubit=registers.logical_measurement_ancillas[ancilla_name][0],
        classical_bits=[classical_bit],
        destructive=False,
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

    append_surface_code_logical_measurement(
        circuit,
        normalized_basis,
        surface_qubits=_surface_data_qubits(registers),
        support=support,
        classical_bits=classical_bits,
        destructive=True,
    )

    if add_barriers:
        circuit.barrier(label=f"destructive logical {normalized_basis}_{''.join(support)}")


def build_steane_to_rotated_d3_lattice_surgery_circuit(
    *,
    injected_state: InjectedStateName | tuple[complex, complex] = "H",
    include_cultivation: bool = True,
    merged_rounds: int = 3,
    noisy: bool = False,
    include_steane_verification: bool = True,
    include_surface_qec: bool = True,
    surface_qec_rounds: int = 3,
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
        three repeated rounds for distance-3 fault detection.
    noisy
        Deprecated compatibility flag for explicitly including parser-side
        reconstructed X-checks.  The complete d=3 path also recovers ``X_ab``
        whenever ``include_steane_verification`` is ``True``.
    include_steane_verification
        If ``True``, measure the d=3 Steane/color-code checks used for
        lattice-surgery postselection and recover ``X0145ab`` via ``X_ab``.
    include_surface_qec
        If ``True``, append post-transfer rotated-d3 stabilizer rounds.
    surface_qec_rounds
        Number of post-transfer rotated-d3 stabilizer rounds.  The first
        X-stabilizer round establishes the reference because these outcomes
        are not all deterministic immediately after surgery.
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
    if include_surface_qec and surface_qec_rounds < 1:
        raise ValueError("surface_qec_rounds must be at least 1 when include_surface_qec=True.")

    # Allocate registers first.  Optional ancilla/register groups are controlled
    # by the same flags used later for appending protocol stages.
    qregs = build_hirano_labeled_qregs(
        include_steane_verification=include_steane_verification,
        include_noisy_verification=noisy or include_steane_verification,
        include_surface_qec=include_surface_qec,
        surface_logical_measurement_basis=(
            surface_readout_basis
            if (
                surface_readout_basis is not None
                and surface_readout_mode == "logical_pauli_indirect"
            )
            else None
        ),
    )
    classical_specs = _classical_register_specs(
        include_cultivation=include_cultivation,
        merged_rounds=merged_rounds,
        noisy=noisy,
        include_steane_verification=include_steane_verification,
        include_surface_qec=include_surface_qec,
        surface_qec_rounds=surface_qec_rounds,
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
        + list(qregs.steane_verification_ancillas.values())
        + list(qregs.surface_data.values())
        + list(qregs.surface_z_ancillas.values())
        + list(qregs.surface_x_ancillas.values())
        + list(qregs.merge_ancillas.values())
        + list(qregs.logical_measurement_ancillas.values())
    )
    classical_registers = [classical_registers_by_name[spec.name] for spec in classical_specs]

    circuit = QuantumCircuit(*qubit_registers, *classical_registers)
    patch_qubits = _hirano_labeled_patch_qubits(qregs)

    # Stage 1: prepare the input logical state on the Steane/color-code patch.
    # The imported helper expects the repo's 11-site patch ordering, not the
    # Hirano labels, so ``patch_qubits`` is the key adapter.
    append_patch_steane_state_injection_encoder_crumble_sequences(
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

    # Stage 2: initialize a rotated d=3 surface patch.  Starting from physical
    # |+> states and measuring Z checks prepares the logical |+_L> sector up to
    # the recorded Z-stabilizer reference.
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

    # Stage 3: perform the merged-boundary ZZ measurements that realize the
    # logical Z_L Z_L teleportation measurement in this d=3 adaptation.
    append_boundary_zz_checks(
        circuit,
        registers=qregs,
        classical_bits=list(classical_registers_by_name["ls_zz"]),
        merged_rounds=merged_rounds,
        add_barriers=add_barriers,
    )

    # Stage 4: measure Steane-side checks that should be trivial in the
    # noiseless accepted branch.  The skipped X0145 part is handled after
    # demolition by combining it with recovered X_ab.
    if include_steane_verification:
        append_steane_lattice_surgery_verification(
            circuit,
            registers=qregs,
            classical_bits=list(classical_registers_by_name["steane_verify"]),
            add_barriers=add_barriers,
        )

    # Stage 5: demolish the Steane block.  These physical X measurements supply
    # both the logical X feedforward and final Steane X-check parities.
    append_measure_steane_data_in_x_basis(
        circuit,
        registers=qregs,
        classical_bits=list(classical_registers_by_name["steane_x"]),
        add_barriers=add_barriers,
    )

    # Stage 6: recover X_ab so the final mixed check X0145ab can be evaluated.
    if noisy or include_steane_verification:
        append_surface_x_ab_recovery_measurement(
            circuit,
            registers=qregs,
            classical_bit=classical_registers_by_name["surface_x_recovery"][0],
            add_barriers=add_barriers,
        )

    # Stage 7: run surface-code stabilization after transfer.  The first
    # post-transfer X-check round establishes a reference rather than a
    # deterministic syndrome; see ``surface_qec_deltas`` for the matching
    # parser-side convention.
    if include_surface_qec:
        post_z_bits = list(classical_registers_by_name["surface_post_z"])
        post_x_bits = list(classical_registers_by_name["surface_post_x"])
        for round_index in range(surface_qec_rounds):
            z_start = round_index * len(SURFACE_POST_Z_LABELS)
            x_start = round_index * len(SURFACE_POST_X_LABELS)
            append_stabilizer_round(
                circuit,
                surface_qubits=surface_qubits,
                z_ancillas=[
                    qregs.surface_z_ancillas[label][0]
                    for label in SURFACE_Z_ANCILLA_LABELS
                ],
                x_ancillas=[
                    qregs.surface_x_ancillas[label][0]
                    for label in SURFACE_X_ANCILLA_LABELS
                ],
                z_bits=post_z_bits[z_start:z_start + len(SURFACE_POST_Z_LABELS)],
                x_bits=post_x_bits[x_start:x_start + len(SURFACE_POST_X_LABELS)],
                round_label=f"post-transfer surface round {round_index + 1}",
                prepare_z_ancillas=True,
                prepare_x_ancillas=True,
                add_barriers=add_barriers,
            )

    # Stage 8: optional final readout.  Tests use this to verify the logical
    # transfer on basis states; notebooks use it for expectation estimates.
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


# ===========================================================================
# Count-key parser
# ===========================================================================

def parse_transfer_measurements(
    bitstring: str,
    *,
    include_cultivation: bool = True,
    merged_rounds: int = 3,
    noisy: bool = False,
    include_steane_verification: bool = True,
    include_surface_qec: bool = True,
    surface_qec_rounds: int = 3,
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
    include_steane_verification
        Whether the circuit included the Steane/color-code LS verification
        register and recovered ``X_ab`` top-boundary check.
    include_surface_qec
        Whether the circuit included a post-transfer full surface-code
        stabilizer round.
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
        include_steane_verification=include_steane_verification,
        include_surface_qec=include_surface_qec,
        surface_qec_rounds=surface_qec_rounds,
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

    # Final readouts are parsed first because Qiskit prints classical registers
    # in reverse order.  The register specs still define the authoritative
    # semantic labels, so downstream code does not need to know bit positions.
    if surface_readout_basis is not None:
        if surface_readout_mode == "data":
            parsed["surface_readout"] = register_bits(f"surface_{surface_readout_basis.lower()}")
        else:  # logical_pauli or logical_pauli_indirect
            parsed["surface_logical_readout"] = register_bits(
                f"surface_logical_{surface_readout_basis.lower()}"
            )

    # Surface QEC is recorded as flat classical registers but exposed as a list
    # of per-round dictionaries.  That representation is easier to feed into
    # StabilizerFrame and RotatedDecoder.
    if include_surface_qec:
        post_z_bits = register_bits("surface_post_z")
        post_x_bits = register_bits("surface_post_x")
        parsed["surface_post_z_rounds"] = [
            {
                label: post_z_bits[f"round{round_index + 1}_{label}"]
                for label in SURFACE_POST_Z_LABELS
            }
            for round_index in range(surface_qec_rounds)
        ]
        parsed["surface_post_x_rounds"] = [
            {
                label: post_x_bits[f"round{round_index + 1}_{label}"]
                for label in SURFACE_POST_X_LABELS
            }
            for round_index in range(surface_qec_rounds)
        ]
        parsed["surface_post_z"] = parsed["surface_post_z_rounds"][-1]
        parsed["surface_post_x"] = parsed["surface_post_x_rounds"][-1]

    # Surface X_ab may be present because the user explicitly asked for noisy
    # diagnostics or because the complete Steane verification path needs it to
    # reconstruct X0145ab.
    if noisy or include_steane_verification:
        parsed["surface_x_recovery"] = register_bits("surface_x_recovery")

    parsed["steane_x"] = register_bits("steane_x")

    if include_steane_verification:
        parsed["steane_verification"] = register_bits("steane_verify")

    # Merged-boundary checks are likewise exposed as one dictionary per repeated
    # round, even though they live in a flat classical register.
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

    # Derived fields are intentionally computed here so notebooks/tests can
    # inspect a single parsed measurement object without reimplementing protocol
    # rules in multiple places.
    if include_cultivation:
        parsed["cultivation"] = register_bits("cult")

    if noisy or include_steane_verification:
        parsed["reconstructed_x_checks"] = reconstructed_x_checks(parsed)
        parsed["local_x_verification_checks"] = local_x_verification_checks(parsed)
        parsed["x_verification_succeeds"] = x_verification_succeeds(parsed)
    if include_steane_verification:
        parsed["steane_verification_succeeds"] = steane_verification_succeeds(parsed)
    if include_surface_qec:
        parsed["surface_qec_deltas"] = surface_qec_deltas(parsed)
        recoveries = surface_qec_recoveries(parsed)
        parsed["surface_qec_recoveries"] = recoveries
        parsed["surface_qec_syndrome_is_trivial"] = all(
            recovery.is_trivial for recovery in recoveries
        )

    return parsed


# ===========================================================================
# Derived transfer outcomes and postselection predicates
# ===========================================================================

def merged_boundary_logical_zz_outcome(
    measurements: dict[str, object],
) -> int:
    """Return the logical ``Z_L Z_L`` parity inferred from merged checks.

    The d=3 adaptation infers the logical parity as the parity of all
    merged-boundary checks in a round.  Repeated rounds must agree before this
    value is considered valid.
    """
    round_parities = [
        sum(round_bits.values()) % 2
        for round_bits in measurements["merged_zz_rounds"]
    ]
    if len(set(round_parities)) != 1:
        raise ValueError(
            "Merged-boundary rounds disagree; this noiseless demonstrator expects a stable parity."
        )
    return round_parities[0]


def merged_boundary_checks_stable(measurements: dict[str, object]) -> bool:
    """Return whether every repeated merged-boundary check is time-stable.

    This is the measurement-error detection rule for the repeated boundary
    checks.  It is separate from ``merged_boundary_logical_zz_outcome`` so code
    can ask "would this shot be accepted?" without raising.
    """
    rounds = measurements["merged_zz_rounds"]
    return all(
        len({round_bits[check_name] for round_bits in rounds}) == 1
        for check_name, _, _ in MERGED_BOUNDARY_CHECKS
    )


def steane_logical_x_outcome(measurements: dict[str, object]) -> int:
    """Return the Steane logical-``X`` outcome from paper support ``X146``.

    This bit controls the logical-Z Pauli-frame update on the transferred
    surface-code magic state.
    """
    steane_x = measurements["steane_x"]
    return sum(steane_x[label] for label in HIRANO_STEANE_LOGICAL_X) % 2


def reconstructed_x_checks(measurements: dict[str, object]) -> dict[str, int]:
    """Reconstruct paper-style final X checks from demolition data.

    ``X0235`` and ``X0246`` are direct Steane demolition parities.  ``X0145ab``
    is reconstructed by XORing the Steane ``X0145`` demolition parity with the
    recovered surface ``X_ab`` stabilizer outcome.
    """
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
    """Return deterministic final X checks for this d=3 adaptation."""
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
    """Return whether all reconstructed final X checks have trivial syndrome."""
    return all(bit == 0 for bit in local_x_verification_checks(measurements).values())


def steane_verification_succeeds(measurements: dict[str, object]) -> bool:
    """Return whether non-destructive Steane LS checks are all trivial.

    In the fault-detecting path, any nonzero color-code syndrome measured
    during the merged-code interval rejects the transfer attempt.
    """
    verification = measurements.get("steane_verification")
    if verification is None:
        return True
    return all(bit == 0 for bit in verification.values())


# ===========================================================================
# Post-transfer surface-code QEC interpretation
# ===========================================================================

def surface_qec_deltas(measurements: dict[str, object]) -> list[dict[str, dict[str, int]]]:
    """Return change-detected post-transfer surface-code stabilizer deltas.

    The first post-transfer X-stabilizer round is treated as a reference,
    matching the paper's note that newly introduced X-stabilizer outcomes are
    not all deterministic immediately after lattice surgery.

    Z-stabilizer deltas compare against the initialization Z-check round
    because the surface patch was prepared through those checks before the
    merge.  X-stabilizer deltas compare against the first post-transfer X round
    because the transferred state has not previously supplied a full X-check
    reference.
    """
    init_z = measurements["surface_init_z"]
    post_z_rounds = measurements["surface_post_z_rounds"]
    post_x_rounds = measurements["surface_post_x_rounds"]

    z_frame = StabilizerFrame()
    z_frame.record_many(init_z)
    x_frame = StabilizerFrame()
    x_frame.record_many(post_x_rounds[0])

    deltas: list[dict[str, dict[str, int]]] = []
    for round_index, (post_z, post_x) in enumerate(zip(post_z_rounds, post_x_rounds, strict=True)):
        # Z checks already have a preparation reference from ``surface_init_z``.
        z_deltas = z_frame.compare_many(post_z, update=True)
        if round_index == 0:
            # The first X round establishes the post-transfer reference.  Treat
            # its delta as zero rather than decoding it as a physical error.
            x_deltas = {label: 0 for label in post_x}
        else:
            x_deltas = x_frame.compare_many(post_x, update=True)
        deltas.append({"z": z_deltas, "x": x_deltas})
    return deltas


def surface_qec_recoveries(measurements: dict[str, object]):
    """Decode all post-transfer rotated-d3 surface-code stabilizer deltas.

    The returned recovery objects describe physical corrections inferred by the
    rotated-surface matching decoder.  ``transfer_pauli_frame`` uses the same
    decoder to update logical byproducts instead of applying gates.
    """
    deltas_by_round = surface_qec_deltas(measurements)
    decoder = RotatedDecoder(
        data_qubits=SURFACE_ORDER,
        z_stabilizers=Z_STABILIZERS,
        x_stabilizers=X_STABILIZERS,
        logical_x=DEFAULT_LOGICAL_X,
        logical_z=DEFAULT_LOGICAL_Z,
    )
    return [
        decoder.decode(z_deltas=deltas["z"], x_deltas=deltas["x"])
        for deltas in deltas_by_round
    ]


def surface_qec_recovery(measurements: dict[str, object]):
    """Return the final decoded post-transfer surface-code recovery.

    This is a convenience accessor for one-shot diagnostics.  Full frame
    interpretation should generally use every recovery round.
    """
    return surface_qec_recoveries(measurements)[-1]


def transfer_acceptance_succeeds(measurements: dict[str, object]) -> bool:
    """Return whether the fault-detecting d=3 transfer checks accept.

    This predicate covers the transfer-local rejection rules implemented in
    this module: cultivation bits (if present), stable repeated merged checks,
    Steane LS verification checks, and final reconstructed X checks.  Surface
    QEC decoding is tracked separately because nontrivial surface syndromes can
    be corrected in the Pauli frame rather than causing hard rejection.
    """
    return (
        cultivation_succeeds(measurements)
        and merged_boundary_checks_stable(measurements)
        and steane_verification_succeeds(measurements)
        and x_verification_succeeds(measurements)
    )


def transfer_pauli_frame(
    measurements: dict[str, object],
    *,
    apply_surface_qec: bool = True,
) -> PauliFrame:
    """Build the target Pauli frame from transfer and surface-QEC outcomes.

    The lattice-surgery teleportation contributes two logical byproducts:

    - merged ``Z_L Z_L`` parity toggles a logical X byproduct
    - Steane destructive ``X146`` parity toggles a logical Z byproduct

    If post-transfer surface QEC data is present, decoded physical recoveries
    are also folded into the same Pauli frame.
    """
    frame = PauliFrame()
    frame.register("surface")

    # Teleportation byproducts from the lattice-surgery measurement pattern.
    frame.update_x("surface", bool(merged_boundary_logical_zz_outcome(measurements)))
    frame.update_z("surface", bool(steane_logical_x_outcome(measurements)))

    # Surface-code syndrome rounds are correction data, not hard rejection
    # criteria.  Convert their decoded recoveries into logical frame updates.
    if apply_surface_qec and "surface_post_z" in measurements:
        deltas_by_round = surface_qec_deltas(measurements)
        decoder = RotatedDecoder(
            data_qubits=SURFACE_ORDER,
            z_stabilizers=Z_STABILIZERS,
            x_stabilizers=X_STABILIZERS,
            logical_x=DEFAULT_LOGICAL_X,
            logical_z=DEFAULT_LOGICAL_Z,
        )
        for deltas in deltas_by_round:
            decoder.update_pauli_frame(
                frame,
                "surface",
                z_deltas=deltas["z"],
                x_deltas=deltas["x"],
            )
    return frame


# ===========================================================================
# Surface logical readout interpretation
# ===========================================================================

def raw_surface_logical_readout(
    measurements: dict[str, object],
    *,
    basis: SurfaceReadoutBasis,
) -> int:
    """Return the uncorrected surface logical readout from measurement bits.

    Supports three readout styles:

    - all-data destructive readout, where the logical bit is a support parity
    - logical-Pauli destructive readout, where only support qubits are measured
    - logical-Pauli indirect readout, where one combined ancilla bit is present
    """
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
    """Return the surface logical readout corrected by the transfer Pauli frame.

    This is the endpoint used by the logical-transfer tests: build a frame from
    all transfer/QEC data, compute the raw surface logical bit, and interpret it
    through the frame.
    """
    frame = transfer_pauli_frame(measurements)
    raw_bit = raw_surface_logical_readout(measurements, basis=basis)
    if basis.upper() == "X":
        return frame.interpret_x_readout("surface", raw_bit)
    return frame.interpret_z_readout("surface", raw_bit)


def cultivation_succeeds(measurements: dict[str, object]) -> bool:
    """Return whether the optional cultivation postselection bits are all zero.

    Transfer-only circuits omit the cultivation register; in that case the
    predicate returns ``True`` so it composes naturally with transfer acceptance.
    """
    cultivation = measurements.get("cultivation")
    if cultivation is None:
        return True
    return sum(cultivation.values()) == 0
