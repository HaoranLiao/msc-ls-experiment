"""Kickback tomography helpers for the Steane-code cultivation patch.

This module builds Fig. 3-style kickback tomography (KT) circuits from
Rosenfeld et al. using the repo's 11-site and 14-site Steane patch primitives.
The KT measurement reuses cultivation rounds as repeated logical H_XY- or
H_XZ-axis measurements: one optional preparation round, one tomography
measurement round, and one consistency-check round.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import get_args

import numpy as np
import yaml
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

from msc.steane_code.kickback_tomography_common import (
    KT_STABILIZER_BITS_PER_CYCLE,
    FinalStabilizerMode,
    KickbackCultivationRoundBits,
    KickbackTomographyBitLayout,
    KickbackTomographyPatchSites,
    KickbackTomographyPatchSpec,
    KickbackTomographySummary,
    cultivation_round_bits_from_positions,
    final_stabilizer_bit_count,
    normalize_kickback_tomography_state,
    normalize_qed_cycle_indices,
    resolve_kickback_tomography_patch_spec,
)
from msc.steane_code.magic_state_cultivation_common import CultivatedStateName
from msc.steane_code.magic_state_cultivation_qiskit import (
    append_11_site_patch_magic_state_projection_round,
    append_14_site_patch_magic_state_projection_round,
)
from msc.steane_code.state_injection_qiskit import (
    InjectedStateName,
    append_11_site_patch_rosenfeld_encoder,
    append_11_site_patch_stabilizer_measurement,
    append_14_site_patch_abstract_routed_encoder,
    append_14_site_patch_superdense_stabilizer_measurement,
)
from msc.utils.normalize import qiskit_bit_value

# =============================================================================
# Circuit configuration
# =============================================================================

KICKBACK_TOMOGRAPHY_CONFIG_PATH = Path(__file__).with_name("kickback_tomography_config.yaml")

# Fig. 3 injection-angle sweep in [0, 2 pi): pi/12 spacing, refined to pi/24 within
# [pi/12, 5 pi/12] and to pi/48 within [pi/6, pi/3], so 32 points resolve the curve's
# shape while concentrating around the cultivated state at pi/4.
KT_SWEEP_INJECTION_ANGLES: tuple[float, ...] = tuple(
    np.pi * multiple / 48
    for multiple in sorted({*range(0, 96, 4), *range(6, 19, 4), *range(9, 16, 2)})
)


@dataclass(frozen=True)
class KickbackTomographyConfig:
    """Validated structural settings loaded from a KT circuit YAML file.

    ``injection_angle_sweep`` selects the Fig. 3 sweep over
    :data:`KT_SWEEP_INJECTION_ANGLES` on the cultivated state's axis, with
    ``injected_state`` null; otherwise ``injected_state`` names the single KT
    point's injected state.
    """

    patch_sites: KickbackTomographyPatchSites
    cultivated_state: CultivatedStateName
    injected_state: InjectedStateName | None
    injection_angle_sweep: bool
    include_prep_cultivation: bool
    measure_encoder_flags: bool
    qed_cycle_indices: tuple[int, ...]
    final_stabilizer_basis: FinalStabilizerMode
    add_barriers: bool

    def __post_init__(self) -> None:
        if type(self.patch_sites) is not int or self.patch_sites not in (11, 14):
            raise ValueError("patch_sites must be 11 or 14.")
        normalize_kickback_tomography_state(self.cultivated_state)
        injectable_states = get_args(InjectedStateName)
        if self.injected_state is not None and self.injected_state not in injectable_states:
            raise ValueError(f"injected_state must be null or one of {injectable_states}.")
        for name in (
            "injection_angle_sweep",
            "include_prep_cultivation",
            "measure_encoder_flags",
            "add_barriers",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be true or false.")
        if self.injection_angle_sweep == (self.injected_state is not None):
            raise ValueError("injected_state must be null for a sweep and set otherwise.")
        normalize_qed_cycle_indices(self.qed_cycle_indices)
        final_stabilizer_bit_count(self.final_stabilizer_basis)


def load_kickback_tomography_config(
    circuit_config_path: str | Path = KICKBACK_TOMOGRAPHY_CONFIG_PATH,
) -> KickbackTomographyConfig:
    """Load and validate one kickback-tomography circuit YAML configuration."""
    circuit_config_path = Path(circuit_config_path)
    try:
        raw_config = yaml.safe_load(circuit_config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ValueError(f"invalid kickback-tomography YAML in {circuit_config_path}") from error
    if not isinstance(raw_config, dict):
        raise ValueError(f"kickback-tomography config {circuit_config_path} must be a mapping.")

    expected_fields = set(KickbackTomographyConfig.__annotations__)
    if set(raw_config) != expected_fields:
        raise ValueError(
            f"kickback-tomography config {circuit_config_path} fields must be "
            f"{sorted(expected_fields)}."
        )
    return KickbackTomographyConfig(
        **{**raw_config, "qed_cycle_indices": tuple(raw_config["qed_cycle_indices"])}
    )


def resolve_kickback_tomography_config(
    *,
    circuit_config_path: str | Path = KICKBACK_TOMOGRAPHY_CONFIG_PATH,
    patch_sites: KickbackTomographyPatchSites | None = None,
    cultivated_state: CultivatedStateName | None = None,
    injected_state: InjectedStateName | None = None,
    include_prep_cultivation: bool | None = None,
    measure_encoder_flags: bool | None = None,
    qed_cycle_indices: Iterable[int] | None = None,
    final_stabilizer_basis: FinalStabilizerMode | None = None,
    add_barriers: bool | None = None,
) -> KickbackTomographyConfig:
    """Load the KT circuit YAML and apply per-call overrides.

    An ``injected_state`` override selects a single KT point, so it also turns
    off the YAML's ``injection_angle_sweep``.
    """
    config = load_kickback_tomography_config(circuit_config_path)
    overrides = {
        name: value
        for name, value in (
            ("patch_sites", patch_sites),
            ("cultivated_state", cultivated_state),
            ("injected_state", injected_state),
            ("include_prep_cultivation", include_prep_cultivation),
            ("measure_encoder_flags", measure_encoder_flags),
            ("qed_cycle_indices", None if qed_cycle_indices is None else tuple(qed_cycle_indices)),
            ("final_stabilizer_basis", final_stabilizer_basis),
            ("add_barriers", add_barriers),
        )
        if value is not None
    }
    if injected_state is not None:
        overrides["injection_angle_sweep"] = False
    return replace(config, **overrides)


def build_kickback_tomography_experiments(
    *,
    circuit_config_path: str | Path = KICKBACK_TOMOGRAPHY_CONFIG_PATH,
) -> tuple[np.ndarray | None, list[tuple[QuantumCircuit, KickbackTomographyBitLayout]]]:
    """Build every KT point the YAML requests: one injected state or an angle sweep.

    A sweep injects :data:`KT_SWEEP_INJECTION_ANGLES` on the cultivated state's
    axis (polar for ``"Ty"``, azimuth for ``"T"``) and returns them with the
    experiments, in submission order; a single point returns ``None`` for the
    angles.
    """
    config = load_kickback_tomography_config(circuit_config_path)
    if not config.injection_angle_sweep:
        return None, [build_kickback_tomography_experiment(circuit_config_path=circuit_config_path)]
    if config.cultivated_state == "Ty":
        angle_keyword = "injection_polar_angle"
    elif config.cultivated_state == "T":
        angle_keyword = "injection_azimuth_angle"
    else:  # pragma: no cover - validated by KickbackTomographyConfig.
        raise ValueError("cultivated_state must be 'T' or 'Ty'.")
    return np.array(KT_SWEEP_INJECTION_ANGLES), [
        build_kickback_tomography_experiment(
            circuit_config_path=circuit_config_path,
            **{angle_keyword: angle},
        )
        for angle in KT_SWEEP_INJECTION_ANGLES
    ]


def build_kickback_tomography_experiment(
    injected_state: InjectedStateName | None = None,
    *,
    circuit_config_path: str | Path = KICKBACK_TOMOGRAPHY_CONFIG_PATH,
    patch_sites: KickbackTomographyPatchSites | None = None,
    cultivated_state: CultivatedStateName | None = None,
    injection_azimuth_angle: float | None = None,
    injection_polar_angle: float | None = None,
    include_prep_cultivation: bool | None = None,
    measure_encoder_flags: bool | None = None,
    qed_cycle_indices: Iterable[int] | None = None,
    final_stabilizer_basis: FinalStabilizerMode | None = None,
    add_barriers: bool | None = None,
) -> tuple[QuantumCircuit, KickbackTomographyBitLayout]:
    """Build a T/Ty kickback-tomography experiment, defaulting to Ty on 14 sites.

    Structural settings load from ``circuit_config_path``
    (:data:`KICKBACK_TOMOGRAPHY_CONFIG_PATH` by default). Every non-``None``
    keyword overrides the corresponding YAML field; ``injection_azimuth_angle``
    and ``injection_polar_angle`` inject one sweep angle and replace the YAML
    injection state, but cannot accompany an explicit ``injected_state``
    argument. Without an angle, a sweep YAML builds its cultivated-state point;
    :func:`build_kickback_tomography_experiments` builds the whole sweep.

    The shared protocol order is

    ``encode -> QED1 -> [prepare cultivation] -> QED2 -> KT measure
    -> QED3 -> KT check -> QED4 -> final stabilizer check``.

    ``cultivated_state="T"`` measures ``H_XY = (X + Y) / sqrt(2)``;
    ``"Ty"`` measures ``H_XZ = (X + Z) / sqrt(2)``.

    The three injection inputs are those of :func:`prepare_injection_input`,
    except that all ``None`` here injects the selected ``cultivated_state``. Pair
    each Fig. 3 angle sweep with its own axis — ``injection_azimuth_angle`` with
    ``"T"``, ``injection_polar_angle`` with ``"Ty"`` — so the swept plane holds
    the cultivated axis and the injection-only error runs the full
    ``sin^2(angle / 2 - pi / 8)``.

    The 11-site path uses the Rosenfeld encoder, and the 14-site path uses the
    abstract-routed encoder. When ``measure_encoder_flags`` is true, their
    available flag qubits are measured after encoding and postselected to zero.

    ``qed_cycle_indices`` uses the paper's one-based numbering. The default
    schedule is ``KT_DEFAULT_QED_CYCLE_INDICES = (1, 3)``; pass
    ``KT_FULL_QED_CYCLE_INDICES = (1, 2, 3, 4)`` for every Fig. 3 QED cycle.
    QED1 is the protocol's post-injection error-detection round. An empty tuple
    selects the "without QED cycles" variant. These rounds only
    measure error-detection syndromes. KT postselects on them and does not apply
    corrections or Pauli-frame updates.

    ``final_stabilizer_basis="X"`` matches the final detector block drawn in
    Fig. 3. ``"Z"`` selects the equivalent CSS basis, ``"BOTH"`` measures all
    six stabilizers, and ``"NONE"`` omits the final check. A 14-site X- or
    Z-only final block also records and postselects the three default
    Bell-partner flags.
    """
    config = resolve_kickback_tomography_config(
        circuit_config_path=circuit_config_path,
        patch_sites=patch_sites,
        cultivated_state=cultivated_state,
        injected_state=injected_state,
        include_prep_cultivation=include_prep_cultivation,
        measure_encoder_flags=measure_encoder_flags,
        qed_cycle_indices=qed_cycle_indices,
        final_stabilizer_basis=final_stabilizer_basis,
        add_barriers=add_barriers,
    )
    cultivated_state = config.cultivated_state
    if injection_azimuth_angle is None and injection_polar_angle is None:
        # A sweep YAML has no injected state; without an angle, inject the cultivated state.
        injected_state = config.injected_state or cultivated_state
    include_prep_cultivation = config.include_prep_cultivation
    final_stabilizer_basis = config.final_stabilizer_basis
    add_barriers = config.add_barriers
    spec = resolve_kickback_tomography_patch_spec(config.patch_sites)
    qed_indices = normalize_qed_cycle_indices(config.qed_cycle_indices)

    if (cultivated_state == "T" and injection_polar_angle is not None) or (
        cultivated_state == "Ty" and injection_azimuth_angle is not None
    ):
        raise ValueError("Injection sweep angle must match the cultivated-state axis.")

    patch = QuantumRegister(spec.patch_sites, "patch")
    encoder_flags_creg = (
        ClassicalRegister(spec.encoder_flag_count, "enc_flags")
        if config.measure_encoder_flags and spec.encoder_flag_count
        else None
    )
    prep_creg = (
        ClassicalRegister(spec.cultivation_measurement_count, "prep_cult")
        if include_prep_cultivation
        else None
    )
    qed_cregs = {
        index: ClassicalRegister(spec.qed_measurement_count, f"qed_{index}")
        for index in qed_indices
    }
    kt_measure_creg = ClassicalRegister(
        spec.cultivation_measurement_count,
        "kt_measure",
    )
    kt_check_creg = ClassicalRegister(
        spec.cultivation_measurement_count,
        "kt_check",
    )
    final_bit_count = final_stabilizer_bit_count(final_stabilizer_basis)
    final_creg = ClassicalRegister(final_bit_count, "final_stab") if final_bit_count else None
    final_bell_partner_creg = (
        ClassicalRegister(KT_STABILIZER_BITS_PER_CYCLE // 2, "final_bell_partner_flags")
        if spec.patch_sites == 14 and final_stabilizer_basis in ("X", "Z")
        else None
    )
    final_d2_copy_creg = (
        ClassicalRegister(1, "final_d2_copy_flag")
        if spec.patch_sites == 14 and final_stabilizer_basis != "NONE"
        else None
    )

    classical_registers: list[ClassicalRegister] = []
    if encoder_flags_creg is not None:
        classical_registers.append(encoder_flags_creg)
    if prep_creg is not None:
        classical_registers.append(prep_creg)
    classical_registers.extend(qed_cregs[index] for index in qed_indices)
    classical_registers.extend((kt_measure_creg, kt_check_creg))
    if final_creg is not None:
        classical_registers.append(final_creg)
    if final_bell_partner_creg is not None:
        classical_registers.append(final_bell_partner_creg)
    if final_d2_copy_creg is not None:
        classical_registers.append(final_d2_copy_creg)

    circuit = QuantumCircuit(patch, *classical_registers)
    patch_qubits = list(patch)

    _append_patch_encoder(
        circuit,
        spec,
        patch_qubits=patch_qubits,
        injected_state=injected_state,
        injection_azimuth_angle=injection_azimuth_angle,
        injection_polar_angle=injection_polar_angle,
        flag_bits=(list(encoder_flags_creg) if encoder_flags_creg is not None else None),
        add_barriers=add_barriers,
    )

    def append_qed_cycle(index: int) -> None:
        _append_patch_stabilizer_measurement(
            circuit,
            spec,
            patch_qubits=patch_qubits,
            stabilizer_bits=list(qed_cregs[index])[:KT_STABILIZER_BITS_PER_CYCLE],
            d2_copy_flag_bit=(qed_cregs[index][-1] if spec.patch_sites == 14 else None),
            basis="BOTH",
            add_barriers=add_barriers,
        )

    if 1 in qed_cregs:
        append_qed_cycle(1)

    if include_prep_cultivation:
        assert prep_creg is not None
        _append_patch_projection_round(
            circuit,
            spec,
            patch_qubits=patch_qubits,
            bits=list(prep_creg),
            cultivated_state=cultivated_state,
            add_barriers=add_barriers,
        )

    if 2 in qed_cregs:
        append_qed_cycle(2)

    _append_patch_projection_round(
        circuit,
        spec,
        patch_qubits=patch_qubits,
        bits=list(kt_measure_creg),
        cultivated_state=cultivated_state,
        add_barriers=add_barriers,
    )

    if 3 in qed_cregs:
        append_qed_cycle(3)

    _append_patch_projection_round(
        circuit,
        spec,
        patch_qubits=patch_qubits,
        bits=list(kt_check_creg),
        cultivated_state=cultivated_state,
        add_barriers=add_barriers,
    )

    if 4 in qed_cregs:
        append_qed_cycle(4)

    if final_creg is not None:
        _append_patch_stabilizer_measurement(
            circuit,
            spec,
            patch_qubits=patch_qubits,
            stabilizer_bits=list(final_creg),
            bell_partner_flag_bits=(
                list(final_bell_partner_creg) if final_bell_partner_creg is not None else None
            ),
            d2_copy_flag_bit=(final_d2_copy_creg[0] if final_d2_copy_creg is not None else None),
            basis=final_stabilizer_basis,
            add_barriers=add_barriers,
        )

    qed_positions = tuple(
        tuple(circuit.find_bit(bit).index for bit in qed_cregs[index]) for index in qed_indices
    )
    preparation_bits = (
        cultivation_round_bits_from_positions(
            tuple(circuit.find_bit(bit).index for bit in prep_creg),
            root_check_index=spec.cultivation_root_check_index,
        )
        if prep_creg is not None
        else None
    )
    encoder_flag_positions = (
        tuple(circuit.find_bit(bit).index for bit in encoder_flags_creg)
        if encoder_flags_creg is not None
        else ()
    )
    bit_layout = KickbackTomographyBitLayout(
        preparation_round=preparation_bits,
        tomography_measure_round=cultivation_round_bits_from_positions(
            tuple(circuit.find_bit(bit).index for bit in kt_measure_creg),
            root_check_index=spec.cultivation_root_check_index,
        ),
        tomography_check_round=cultivation_round_bits_from_positions(
            tuple(circuit.find_bit(bit).index for bit in kt_check_creg),
            root_check_index=spec.cultivation_root_check_index,
        ),
        qed_cycles=qed_positions,
        qed_cycle_indices=qed_indices,
        final_stabilizers=(
            tuple(circuit.find_bit(bit).index for bit in final_creg)
            if final_creg is not None
            else ()
        ),
        final_bell_partner_flags=(
            tuple(circuit.find_bit(bit).index for bit in final_bell_partner_creg)
            if final_bell_partner_creg is not None
            else ()
        ),
        final_d2_copy_flag=(
            circuit.find_bit(final_d2_copy_creg[0]).index
            if final_d2_copy_creg is not None
            else None
        ),
        encoder_flags=encoder_flag_positions,
    )
    circuit.metadata = {
        **(circuit.metadata or {}),
        "kickback_tomography_bit_layout": {
            "patch_sites": spec.patch_sites,
            "preparation_round": (
                bit_layout.preparation_round.all_positions if bit_layout.preparation_round else None
            ),
            "tomography_measure_round": bit_layout.tomography_measure_round.all_positions,
            "tomography_check_round": bit_layout.tomography_check_round.all_positions,
            "qed_cycles": bit_layout.qed_cycles,
            "final_stabilizers": bit_layout.final_stabilizers,
            "final_bell_partner_flags": bit_layout.final_bell_partner_flags,
            "final_d2_copy_flag": bit_layout.final_d2_copy_flag,
            "encoder_flags": bit_layout.encoder_flags,
            "qed_cycle_indices": qed_indices,
            "cultivated_state": cultivated_state,
            "final_stabilizer_basis": final_stabilizer_basis,
        },
    }
    return circuit, bit_layout


def _append_patch_encoder(
    circuit: QuantumCircuit,
    spec: KickbackTomographyPatchSpec,
    *,
    patch_qubits: list,
    injected_state: InjectedStateName | None,
    injection_azimuth_angle: float | None,
    injection_polar_angle: float | None,
    flag_bits: list | None,
    add_barriers: bool,
) -> None:
    """Append the selected patch encoder and optional flag measurements."""
    if spec.patch_sites == 11:
        append_11_site_patch_rosenfeld_encoder(
            circuit,
            patch_qubits=patch_qubits,
            injected_state=injected_state,
            injection_azimuth_angle=injection_azimuth_angle,
            injection_polar_angle=injection_polar_angle,
            flag_bits=flag_bits,
            add_barriers=add_barriers,
        )
    elif spec.patch_sites == 14:
        append_14_site_patch_abstract_routed_encoder(
            circuit,
            patch_qubits=patch_qubits,
            injected_state=injected_state,
            injection_azimuth_angle=injection_azimuth_angle,
            injection_polar_angle=injection_polar_angle,
            flag_bits=flag_bits,
            add_barriers=add_barriers,
        )
    else:
        raise ValueError(f"Unsupported KT patch_sites: {spec.patch_sites}.")


def _append_patch_stabilizer_measurement(
    circuit: QuantumCircuit,
    spec: KickbackTomographyPatchSpec,
    *,
    patch_qubits: list,
    stabilizer_bits: list,
    bell_partner_flag_bits: list | None = None,
    d2_copy_flag_bit=None,
    basis: FinalStabilizerMode,
    add_barriers: bool,
) -> None:
    """Append one full-QED or final stabilizer measurement on the patch.

    ``stabilizer_bits`` stores only face-parity results. The two
    optional destinations store the 14-site Bell-partner and D2-copy flags.
    """
    if spec.patch_sites == 11:
        append_11_site_patch_stabilizer_measurement(
            circuit,
            patch_qubits=patch_qubits,
            bits=stabilizer_bits,
            basis=basis,
            add_barriers=add_barriers,
        )
    elif spec.patch_sites == 14:
        append_14_site_patch_superdense_stabilizer_measurement(
            circuit,
            patch_qubits=patch_qubits,
            stabilizer_bits=stabilizer_bits,
            bell_partner_flag_bits=bell_partner_flag_bits,
            d2_copy_flag_bit=d2_copy_flag_bit,
            basis=basis,
            add_barriers=add_barriers,
        )
    else:
        raise ValueError(f"Unsupported KT patch_sites: {spec.patch_sites}.")


def _append_patch_projection_round(
    circuit: QuantumCircuit,
    spec: KickbackTomographyPatchSpec,
    *,
    patch_qubits: list,
    bits: list,
    cultivated_state: CultivatedStateName,
    add_barriers: bool,
) -> None:
    """Append one cultivation-derived magic-state-axis measurement on the patch."""
    if spec.patch_sites == 11:
        append_11_site_patch_magic_state_projection_round(
            circuit,
            patch_qubits=patch_qubits,
            bits=bits,
            cultivated_state=cultivated_state,
            add_barriers=add_barriers,
        )
    elif spec.patch_sites == 14:
        append_14_site_patch_magic_state_projection_round(
            circuit,
            patch_qubits=patch_qubits,
            bits=bits,
            cultivated_state=cultivated_state,
            add_barriers=add_barriers,
        )
    else:
        raise ValueError(f"Unsupported KT patch_sites: {spec.patch_sites}.")


# =============================================================================
# Counts parsing and KT estimators
# =============================================================================


def _interpret_cultivation_round(
    bitstring: str,
    round_bits: KickbackCultivationRoundBits,
) -> tuple[int, int | None]:
    """Return one round's folded root and its optional repeated-root check.

    The check is the root re-measured after unfolding. Both reads are reset
    before measurement, so each raw bit reports its own outcome directly.
    """
    root = qiskit_bit_value(bitstring, round_bits.folded_root)
    if round_bits.folded_root_check is None:
        return root, None
    return root, qiskit_bit_value(bitstring, round_bits.folded_root_check)


def analyze_kickback_tomography_counts(
    counts: dict[str, int],
    bit_layout: KickbackTomographyBitLayout,
    *,
    retain_measure_check_disagreement: bool = True,
) -> KickbackTomographySummary:
    """Analyze Qiskit counts by postselecting and classifying the two KT logical readouts.

    Fixed-zero detector bits reject first. Each folded root must also agree
    with its repeated-root check (the root re-measured after unfolding),
    otherwise the shot is discarded. Shots whose two inter-round roots disagree
    are retained by default and reported separately. Agreement at zero is a
    logical success; agreement at one is a logical error.
    """

    total_shots = sum(counts.values())
    retained_shots = 0
    logical_success_shots = 0
    logical_error_shots = 0
    measure_check_disagreement_shots = 0

    for bitstring, count in counts.items():
        # Encoder flags, preparation, QED syndromes, KT flags, and the final
        # stabilizer check are fixed-zero postselection detectors.
        if any(qiskit_bit_value(bitstring, position) for position in bit_layout.detector_positions):
            continue

        # Each folded root must agree with its repeated-root check.
        measurement_root, measurement_root_check = _interpret_cultivation_round(
            bitstring,
            bit_layout.tomography_measure_round,
        )
        check_root, check_root_check = _interpret_cultivation_round(
            bitstring,
            bit_layout.tomography_check_round,
        )
        if (measurement_root_check is not None and measurement_root_check != measurement_root) or (
            check_root_check is not None and check_root_check != check_root
        ):
            continue

        # Root disagreement is diagnostic, not a logical error. The caller may
        # retain it in the denominator or turn it into another rejection rule.
        if measurement_root != check_root:
            if retain_measure_check_disagreement:
                retained_shots += count
                measure_check_disagreement_shots += count
            continue

        # Equal roots are the repeated KT logical outcome.
        retained_shots += count
        if measurement_root:
            logical_error_shots += count
        else:
            logical_success_shots += count

    return KickbackTomographySummary(
        total_shots=total_shots,
        retained_shots=retained_shots,
        rejected_shots=total_shots - retained_shots,
        logical_success_shots=logical_success_shots,
        logical_error_shots=logical_error_shots,
        measure_check_disagreement_shots=measure_check_disagreement_shots,
    )
