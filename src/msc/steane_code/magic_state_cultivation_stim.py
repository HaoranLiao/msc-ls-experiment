"""Stim logical-Pauli projection and logical-Y cultivation on the Steane patch.

This module parallels :mod:`magic_state_cultivation_qiskit` for the 11-site
patch, and provides the parallel 14-site path. The fold schedules, patch
sites, measurement counts, and optional-CX filtering are shared via
:mod:`magic_state_cultivation_common`. Both patches replace the non-Clifford
transversal rotation with ``S``/``S_DAG``, so Stim performs logical-Y
cultivation (proxy for logical ``H_XY``) and cultivates the logical Pauli-Y
eigenstate. The 14-site round also supports direct logical-X and logical-Z
diagnostic projections through identity and ``H`` basis changes.

Each 11-site round appends five measurements to Stim's record in this order:

1. folded root at site 7;
2. helper sites 1, 2, 6, and 7, where site 7 is the repeated-root check and
   the other three are fixed-zero checks.

Each 14-site round appends six measurements in this order:

1. folded root at site 10;
2. helper sites 8-12, where site 10 is the repeated-root check and the other
   four are fixed-zero checks.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import stim

from msc.steane_code.magic_state_cultivation_common import (
    PATCH_11_SITE_CULTIVATION_ROOT_CHECK_INDEX,
    PATCH_11_SITE_FOLD_LAYERS,
    PATCH_11_SITE_FOLDED_MEASUREMENT_SITE,
    PATCH_14_SITE_FOLD_LAYERS,
    PATCH_14_SITE_FOLDED_MEASUREMENT_SITE,
    Patch14SiteEncoderName,
    filter_11_site_patch_fold_layer,
    measured_14_site_cultivation_helpers,
)
from msc.steane_code.state_injection_common import (
    NUM_QUBITS_PATCH_11_SITES,
    NUM_QUBITS_PATCH_14_SITES,
    PATCH_11_SITE_DATA_SITES,
    PATCH_11_SITE_HELPER_SITES,
    PATCH_14_SITE_DATA_SITES,
    PATCH_14_SITE_HELPER_SITES,
    resolve_14_site_patch_stabilizer_checks,
)
from msc.steane_code.state_injection_stim import (
    DEFAULT_STIM_11_SITE_PATCH_QUBITS,
    DEFAULT_STIM_14_SITE_PATCH_QUBITS,
    StimInjectedStateName,
    append_11_site_patch_rosenfeld_encoder,
    append_11_site_patch_stabilizer_measurement,
    append_14_site_patch_abstract_encoder,
    append_14_site_patch_abstract_routed_encoder,
    append_14_site_patch_hirano_encoder,
    append_14_site_patch_superdense_stabilizer_measurement,
)
from msc.utils import circuit_stim
from msc.utils.normalize import normalize_pauli_char, resolve_patch_qubits

ProjectionAxis = Literal["X", "Y", "Z"]


def measure_in_x_basis(circuit: stim.Circuit, qubit: int) -> None:
    """Measure ``qubit`` in the X basis."""
    circuit.append("MX", [qubit])


def reset_to_plus(circuit: stim.Circuit, qubit: int) -> None:
    """Reset a qubit and prepare ``|+>``."""
    circuit.append("RX", [qubit])


def append_cnot_layer_from_site_labels(
    circuit: stim.Circuit,
    patch_qubits: Sequence[int],
    cnot_layer: Sequence[tuple[int, int]],
) -> None:
    """Append one CNOT layer using patch site labels."""
    targets: list[int] = []
    for control_site, target_site in cnot_layer:
        targets.extend(
            (
                patch_qubits[control_site],
                patch_qubits[target_site],
            )
        )
    if targets:
        circuit.append("CX", targets)


# =============================================================================
# Fold / unfold network
# =============================================================================


def append_11_site_patch_fold_network(
    circuit: stim.Circuit,
    patch_qubits: Sequence[int],
    *,
    add_ticks: bool = True,
    skip_cx_1_4: bool = False,
    skip_cx: tuple[int, int] | None = None,
) -> None:
    """Append the forward 4-3-2-1 fold network for the 11-site patch."""
    for layer in PATCH_11_SITE_FOLD_LAYERS:
        append_cnot_layer_from_site_labels(
            circuit,
            patch_qubits,
            filter_11_site_patch_fold_layer(
                layer,
                skip_cx_1_4=skip_cx_1_4,
                skip_cx=skip_cx,
            ),
        )
        if add_ticks:
            circuit.append("TICK", [])


def append_11_site_patch_unfold_network(
    circuit: stim.Circuit,
    patch_qubits: Sequence[int],
    *,
    add_ticks: bool = True,
    skip_cx_1_4: bool = False,
    skip_cx: tuple[int, int] | None = None,
) -> None:
    """Append the inverse 11-site fold network in fully reversed order."""
    for layer in reversed(PATCH_11_SITE_FOLD_LAYERS):
        append_cnot_layer_from_site_labels(
            circuit,
            patch_qubits,
            list(
                reversed(
                    filter_11_site_patch_fold_layer(
                        layer,
                        skip_cx_1_4=skip_cx_1_4,
                        skip_cx=skip_cx,
                    )
                )
            ),
        )
        if add_ticks:
            circuit.append("TICK", [])


def append_14_site_patch_fold_network(
    circuit: stim.Circuit,
    patch_qubits: Sequence[int],
    *,
    add_ticks: bool = True,
) -> None:
    """Append the forward 4-4-2-1 H10-root fold network.

    The active helper ancillas H8-H12 fold the seven data sites onto H10.
    ``patch_qubits`` must index at least sites 0-12; site 13 (D2') is unused.
    """
    for layer in PATCH_14_SITE_FOLD_LAYERS:
        append_cnot_layer_from_site_labels(circuit, patch_qubits, layer)
        if add_ticks:
            circuit.append("TICK", [])


def append_14_site_patch_unfold_network(
    circuit: stim.Circuit,
    patch_qubits: Sequence[int],
    *,
    add_ticks: bool = True,
) -> None:
    """Append the inverse 14-site fold network in fully reversed order."""
    for layer in reversed(PATCH_14_SITE_FOLD_LAYERS):
        append_cnot_layer_from_site_labels(circuit, patch_qubits, list(reversed(layer)))
        if add_ticks:
            circuit.append("TICK", [])


# =============================================================================
# Logical projection rounds (fold, measure, unfold, measure)
# =============================================================================


def append_11_site_patch_y_projection_round(
    circuit: stim.Circuit,
    *,
    patch_qubits: Sequence[int] = DEFAULT_STIM_11_SITE_PATCH_QUBITS,
    add_ticks: bool = True,
    skip_cx_1_4: bool = False,
    skip_cx: tuple[int, int] | None = None,
    running_kt: bool = False,
) -> tuple[int, ...]:
    """Append one detector-annotated logical-Y cultivation round on the 11-site patch.

    Stim replaces the non-Clifford transversal rotation with ``S``/``S_DAG``
    while preserving the fold network, so this is logical-Y cultivation (a
    proxy for either supported magic-state axis in KT). Three helper flags
    (sites 1, 2, 6) are always postselected to zero, and the site-7 root check
    agrees with the folded root. Cultivation also postselects the folded root to
    zero; kickback tomography instead retains it as a logical observable.
    Set ``running_kt=False`` for cultivation and ``True`` for KT.

    Parameters
    ----------
    skip_cx_1_4
        Whether to omit ``CX 1 -> 4`` from both fold and unfold.
    skip_cx
        Optional ``(control_site, target_site)`` gate to omit. This takes
        precedence over ``skip_cx_1_4``.
    running_kt
        Whether the round supplies a KT logical readout. Cultivation
        postselects the folded root to zero; KT retains either result.

    Returns
    -------
    tuple[int, ...]
        Absolute measurement-record positions for the folded root followed by
        the four helper results.
    """
    patch_qubits = resolve_patch_qubits(
        patch_qubits,
        expected_count=NUM_QUBITS_PATCH_11_SITES,
    )
    first_projection_position = circuit.num_measurements

    for helper_site in PATCH_11_SITE_HELPER_SITES:
        reset_to_plus(circuit, patch_qubits[helper_site])
    if add_ticks:
        circuit.append("TICK", [])

    circuit.append("S", [patch_qubits[site] for site in PATCH_11_SITE_DATA_SITES])
    if add_ticks:
        circuit.append("TICK", [])

    append_11_site_patch_fold_network(
        circuit,
        patch_qubits,
        add_ticks=add_ticks,
        skip_cx_1_4=skip_cx_1_4,
        skip_cx=skip_cx,
    )

    measure_in_x_basis(circuit, patch_qubits[PATCH_11_SITE_FOLDED_MEASUREMENT_SITE])
    if add_ticks:
        circuit.append("TICK", [])

    reset_to_plus(circuit, patch_qubits[PATCH_11_SITE_FOLDED_MEASUREMENT_SITE])
    if add_ticks:
        circuit.append("TICK", [])

    append_11_site_patch_unfold_network(
        circuit,
        patch_qubits,
        add_ticks=add_ticks,
        skip_cx_1_4=skip_cx_1_4,
        skip_cx=skip_cx,
    )

    for helper_site in PATCH_11_SITE_HELPER_SITES:
        measure_in_x_basis(circuit, patch_qubits[helper_site])
    if add_ticks:
        circuit.append("TICK", [])

    circuit.append(
        "S_DAG",
        [patch_qubits[site] for site in PATCH_11_SITE_DATA_SITES],
    )
    if add_ticks:
        circuit.append("TICK", [])

    # circuit.num_measurements is the running length of Stim's single global
    # record. Reading it at the top and again here brackets exactly the
    # measurements this call contributed, in append order.
    measurement_positions = tuple(range(first_projection_position, circuit.num_measurements))
    folded_root = measurement_positions[0]
    root_check = measurement_positions[PATCH_11_SITE_CULTIVATION_ROOT_CHECK_INDEX]
    flag_positions = (
        measurement_positions[1:PATCH_11_SITE_CULTIVATION_ROOT_CHECK_INDEX]
        + measurement_positions[PATCH_11_SITE_CULTIVATION_ROOT_CHECK_INDEX + 1 :]
    )
    if not running_kt:
        circuit_stim.append_detector(circuit, measurement_positions=(folded_root,))
    for position in flag_positions:
        circuit_stim.append_detector(circuit, measurement_positions=(position,))
    circuit_stim.append_detector(
        circuit,
        measurement_positions=(folded_root, root_check),
    )
    return measurement_positions


def append_14_site_patch_projection_round(
    circuit: stim.Circuit,
    *,
    patch_qubits: Sequence[int] = DEFAULT_STIM_14_SITE_PATCH_QUBITS,
    projection_axis: ProjectionAxis = "Y",
    measure_flags: bool = True,
    add_ticks: bool = True,
    running_kt: bool = False,
) -> tuple[int, ...]:
    """Append one detector-annotated logical-Pauli projection on the 14-site patch.

    Reset the active helpers H8-H12 to ``|+>``, rotate the requested logical
    Pauli onto X, fold onto H10, measure and reset H10 in X, unfold, read out
    the helper checks, then undo the data rotation. X needs no rotation, Y uses
    transversal ``S``/``S_DAG``, and Z uses transversal ``H``/``H``. H7 is
    unused. H10's second result must agree with the folded root.

    Parameters
    ----------
    projection_axis
        Logical Pauli eigenstate projected to +1. ``"Y"`` is Stim's Clifford
        cultivation proxy; ``"X"`` and ``"Z"`` are diagnostic projections.
    measure_flags
        Whether to read the four fixed-+1 ancilla flags H8, H9, H11, and H12
        (each an extra postselection detector). When false, only H10 is
        re-measured as the root check, so the round keeps the folded root and
        its root check and omits the four ancilla-flag measurements.
    running_kt
        Whether to retain both folded-root branches for KT. The root is reset
        and checked after unfolding in either mode, but Stim's detector
        annotations differ: cultivation pins the folded root to +1, whereas KT
        omits that detector and retains both root outcomes.

    Returns
    -------
    tuple[int, ...]
        Absolute record positions for the folded H10 root followed by the
        measured helper readouts (H8-H12 with flags, else H10 alone). The H10
        readout is the root check.
    """
    patch_qubits = resolve_patch_qubits(
        patch_qubits,
        expected_count=NUM_QUBITS_PATCH_14_SITES,
    )
    projection_axis = normalize_pauli_char(projection_axis)
    if projection_axis == "X":
        before_fold_gate = after_unfold_gate = None
    elif projection_axis == "Y":
        before_fold_gate, after_unfold_gate = "S", "S_DAG"
    elif projection_axis == "Z":
        before_fold_gate = after_unfold_gate = "H"
    else:
        raise ValueError("projection_axis must be 'X', 'Y', or 'Z'.")
    measured_helpers = measured_14_site_cultivation_helpers(measure_flags=measure_flags)
    first_projection_position = circuit.num_measurements
    data_qubits = [patch_qubits[site] for site in PATCH_14_SITE_DATA_SITES]

    for helper_site in PATCH_14_SITE_HELPER_SITES[1:]:
        reset_to_plus(circuit, patch_qubits[helper_site])
    if add_ticks:
        circuit.append("TICK", [])

    if before_fold_gate is not None:
        circuit.append(before_fold_gate, data_qubits)
        if add_ticks:
            circuit.append("TICK", [])

    append_14_site_patch_fold_network(circuit, patch_qubits, add_ticks=add_ticks)

    measure_in_x_basis(circuit, patch_qubits[PATCH_14_SITE_FOLDED_MEASUREMENT_SITE])
    if add_ticks:
        circuit.append("TICK", [])

    reset_to_plus(circuit, patch_qubits[PATCH_14_SITE_FOLDED_MEASUREMENT_SITE])
    if add_ticks:
        circuit.append("TICK", [])

    append_14_site_patch_unfold_network(circuit, patch_qubits, add_ticks=add_ticks)

    for helper_site in measured_helpers:
        measure_in_x_basis(circuit, patch_qubits[helper_site])
    if add_ticks:
        circuit.append("TICK", [])

    if after_unfold_gate is not None:
        circuit.append(after_unfold_gate, data_qubits)
        if add_ticks:
            circuit.append("TICK", [])

    # circuit.num_measurements is the running length of Stim's single global
    # record. Reading it at the top and again here brackets exactly the
    # measurements this call contributed, in append order.
    measurement_positions = tuple(range(first_projection_position, circuit.num_measurements))
    folded_root = measurement_positions[0]
    root_check_index = 1 + measured_helpers.index(PATCH_14_SITE_FOLDED_MEASUREMENT_SITE)
    root_check = measurement_positions[root_check_index]
    flag_positions = (
        measurement_positions[1:root_check_index] + measurement_positions[root_check_index + 1 :]
    )
    if not running_kt:
        circuit_stim.append_detector(circuit, measurement_positions=(folded_root,))
    for position in flag_positions:
        circuit_stim.append_detector(circuit, measurement_positions=(position,))
    circuit_stim.append_detector(
        circuit,
        measurement_positions=(folded_root, root_check),
    )
    return measurement_positions


# =============================================================================
# Injection + cultivation front doors (append)
# =============================================================================


def append_11_site_patch_injection_and_cultivation(
    circuit: stim.Circuit,
    *,
    add_qed_round: bool,
    skip_cx_1_4: bool,
    patch_qubits: Sequence[int] = DEFAULT_STIM_11_SITE_PATCH_QUBITS,
    injected_state: StimInjectedStateName = "S",
    cultivation_rounds: int = 1,
    measure_encoder_flags: bool = False,
    add_ticks: bool = True,
    skip_cx: tuple[int, int] | None = None,
) -> None:
    """Append 11-site injection, optional QED, and zero or more cultivation rounds.

    When ``add_qed_round`` is true, all three X and three Z Steane
    stabilizers are measured after injection and before cultivation (the
    papers call this a QEC cycle, but here the six outcomes are used only for
    postselection, not recovery). When ``measure_encoder_flags`` is true, the
    record order is: four encoder flags, then six QED outcomes, then five
    outcomes per cultivation round (the folded root plus its four helper
    flags). Set ``cultivation_rounds=0`` to stop after injection and optional
    QED. Every measured encoder-flag, QED, and cultivation outcome --
    including every helper flag -- is annotated as a postselection detector.
    """
    if cultivation_rounds < 0:
        raise ValueError("cultivation_rounds must be non-negative.")

    append_11_site_patch_rosenfeld_encoder(
        circuit,
        patch_qubits=patch_qubits,
        injected_state=injected_state,
        measure_flags=measure_encoder_flags,
        add_ticks=add_ticks,
    )
    if add_ticks:
        circuit.append("TICK", [])

    if add_qed_round:
        append_11_site_patch_stabilizer_measurement(
            circuit,
            patch_qubits=patch_qubits,
            basis="BOTH",
            add_ticks=add_ticks,
        )
        if add_ticks:
            circuit.append("TICK", [])

    for _ in range(cultivation_rounds):
        append_11_site_patch_y_projection_round(
            circuit,
            patch_qubits=patch_qubits,
            add_ticks=add_ticks,
            skip_cx_1_4=skip_cx_1_4,
            skip_cx=skip_cx,
        )


def append_14_site_patch_injection_and_cultivation(
    circuit: stim.Circuit,
    *,
    add_qed_round: bool,
    patch_qubits: Sequence[int] = DEFAULT_STIM_14_SITE_PATCH_QUBITS,
    injected_state: StimInjectedStateName = "S",
    projection_axis: ProjectionAxis = "Y",
    encoder: Patch14SiteEncoderName = "abstract_routed",
    cultivation_rounds: int = 1,
    measure_encoder_flags: bool = False,
    measure_cultivation_flags: bool = True,
    post_injection_qed_checks: Sequence[str] | None = None,
    add_ticks: bool = True,
) -> None:
    """Append selectable 14-site injection, optional QED, and projection rounds.

    The default ``abstract_routed`` encoder accepts arbitrary Stim-supported
    inputs. ``hirano`` selects the specialized logical-Y encoder. When
    ``add_qed_round`` is true, ``post_injection_qed_checks`` selects the QED
    checks: ``None`` measures all six, ``()`` skips extraction, and names such
    as ``"X_0145"`` select individual checks. All selected faces share one
    superdense schedule; single-basis faces retain their Bell-partner flags.
    Face 0246 also retains the terminal D2-copy flag.

    The Y projection is the cultivation proxy; X/Z are diagnostic Pauli
    projections. The record starts with optional encoder flags (five for
    abstract-routed, six for Hirano), followed by selected QED readouts (seven
    by default), then the folded H10 root and helper readouts per cultivation
    round. With
    ``measure_cultivation_flags`` true each round reads H8-H12 (six outcomes,
    H10 being the root check); false reads only the folded root and its H10
    root check (two outcomes), omitting the four ancilla flags. Set
    ``cultivation_rounds=0`` to stop after injection and optional QED. Every measured encoder-flag,
    QED, and cultivation outcome -- including every helper flag -- is
    annotated as a postselection detector.
    """
    if cultivation_rounds < 0:
        raise ValueError("cultivation_rounds must be non-negative.")

    if encoder == "abstract":
        append_14_site_patch_abstract_encoder(
            circuit,
            patch_qubits=patch_qubits,
            injected_state=injected_state,
            add_ticks=add_ticks,
        )
    elif encoder == "abstract_routed":
        append_14_site_patch_abstract_routed_encoder(
            circuit,
            patch_qubits=patch_qubits,
            injected_state=injected_state,
            measure_flags=measure_encoder_flags,
            add_ticks=add_ticks,
        )
    elif encoder == "hirano":
        append_14_site_patch_hirano_encoder(
            circuit,
            patch_qubits=patch_qubits,
            injected_state=injected_state,
            measure_flags=measure_encoder_flags,
            add_ticks=add_ticks,
        )
    else:
        raise ValueError("encoder must be 'abstract', 'abstract_routed', or 'hirano'.")
    if add_ticks:
        circuit.append("TICK", [])

    if add_qed_round:
        stabilizer_bases, stabilizer_indices = resolve_14_site_patch_stabilizer_checks(
            post_injection_qed_checks
        )
        if stabilizer_indices:
            append_14_site_patch_superdense_stabilizer_measurement(
                circuit,
                patch_qubits=patch_qubits,
                basis=stabilizer_bases,
                stabilizer_indices=stabilizer_indices,
                add_ticks=add_ticks,
            )
        if add_ticks:
            circuit.append("TICK", [])

    for _ in range(cultivation_rounds):
        append_14_site_patch_projection_round(
            circuit,
            patch_qubits=patch_qubits,
            projection_axis=projection_axis,
            measure_flags=measure_cultivation_flags,
            add_ticks=add_ticks,
        )


# =============================================================================
# Circuit builders (top-level orchestrators)
# =============================================================================


def build_11_site_patch_injection_and_cultivation_circuit(
    injected_state: StimInjectedStateName = "S",
    *,
    add_qed_round: bool,
    skip_cx_1_4: bool,
    cultivation_rounds: int = 1,
    add_ticks: bool = True,
    skip_cx: tuple[int, int] | None = None,
) -> stim.Circuit:
    """Authoritative 11-site Stim qubit-layout and appending example.

    Circuit-owning counterpart to
    :func:`append_11_site_patch_steane_injection_and_cultivation`. Stim has no
    registers, so the builder uses qubit indices ``0..10`` and appends
    measurements directly to Stim's record.

    Measurement-record layout: four encoder-flag entries, then (if
    ``add_qed_round``) six Steane QED entries, then five entries per
    cultivation round (the folded root plus its four helper flags). Every
    entry is emitted as a postselection detector.

    Returns
    -------
    stim.Circuit
        New circuit containing injection and all requested cultivation rounds.
    """
    if cultivation_rounds < 0:
        raise ValueError("cultivation_rounds must be non-negative.")

    # Quantum layout: Stim qubit indices 0..10 represent patch sites 0..10.
    circuit = stim.Circuit()

    # Measurement-record layout:
    # - 4 encoder-flag entries;
    # - 6 post-injection Steane QED entries, when enabled;
    # - 5 entries for each cultivation round.
    append_11_site_patch_injection_and_cultivation(
        circuit,
        patch_qubits=DEFAULT_STIM_11_SITE_PATCH_QUBITS,
        injected_state=injected_state,
        cultivation_rounds=cultivation_rounds,
        measure_encoder_flags=True,
        add_qed_round=add_qed_round,
        add_ticks=add_ticks,
        skip_cx_1_4=skip_cx_1_4,
        skip_cx=skip_cx,
    )
    return circuit


def build_14_site_patch_injection_and_cultivation_circuit(
    injected_state: StimInjectedStateName = "S",
    *,
    add_qed_round: bool,
    projection_axis: ProjectionAxis = "Y",
    encoder: Patch14SiteEncoderName = "abstract_routed",
    cultivation_rounds: int = 1,
    measure_cultivation_flags: bool = True,
    add_ticks: bool = True,
) -> stim.Circuit:
    """Build a 14-site injection, QED, and logical-Pauli projection circuit.

    Circuit-owning counterpart to
    :func:`append_14_site_patch_injection_and_cultivation`. Stim has no
    registers, so the builder uses qubit indices ``0..13`` (site 13 is D2',
    used only by the superdense QED round) and appends measurements directly
    to Stim's record.

    Measurement-record layout: the selected encoder's flag entries, then (if
    ``add_qed_round``) seven Steane QED entries (six stabilizers plus the
    terminal D2' flag), then per cultivation round the folded H10 root and its
    helper readouts. ``measure_cultivation_flags`` true reads H8-H12 (six
    entries, H10 the root check); false reads only the folded root and its H10
    root check (two entries). Every entry is emitted as a postselection detector.

    Returns
    -------
    stim.Circuit
        New circuit containing injection and all requested cultivation rounds.
    """
    if cultivation_rounds < 0:
        raise ValueError("cultivation_rounds must be non-negative.")

    # Quantum layout: Stim qubit indices 0..13 represent patch sites 0..13.
    circuit = stim.Circuit()

    # Measurement-record layout:
    # - selected encoder's flag entries;
    # - 7 post-injection Steane QED entries, when enabled;
    # - 6 (flags on) or 2 (flags off) entries for each cultivation round.
    append_14_site_patch_injection_and_cultivation(
        circuit,
        patch_qubits=DEFAULT_STIM_14_SITE_PATCH_QUBITS,
        injected_state=injected_state,
        projection_axis=projection_axis,
        encoder=encoder,
        cultivation_rounds=cultivation_rounds,
        measure_encoder_flags=True,
        measure_cultivation_flags=measure_cultivation_flags,
        add_qed_round=add_qed_round,
        add_ticks=add_ticks,
    )
    return circuit


# =============================================================================
# Postselection
# =============================================================================


def postselect_success_shots(
    shots: np.ndarray,
    postselection_bit_positions: Sequence[int] | None = None,
) -> np.ndarray:
    """Return sampled shots whose selected error-detection bits are all zero.

    If ``postselection_bit_positions`` is omitted, every measurement column is
    included in the all-zero condition. Otherwise, only the selected zero-based
    Stim measurement-record columns are checked.
    """
    if shots.ndim != 2:
        raise ValueError("shots must be a two-dimensional array.")

    positions = (
        tuple(range(shots.shape[1]))
        if postselection_bit_positions is None
        else tuple(postselection_bit_positions)
    )
    if any(position < 0 or position >= shots.shape[1] for position in positions):
        raise ValueError("postselection_bit_positions contains an out-of-range position.")
    if not positions:
        return shots.copy()

    return shots[~shots[:, positions].any(axis=1)]
