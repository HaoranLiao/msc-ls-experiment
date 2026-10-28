"""Circuit-level decoding of saved Qiskit tomography and LER counts with calibrated Stim models.

This module is the repository's explicit Qiskit/Stim boundary exception: only
Qiskit analysis entry points import it, never Stim modules or package
initializers. It converts the submitted, transpiled circuit of one X/Z
tomography point, either a standalone tomography basis or one memory duration
of an LER sweep, into a detector-annotated Stim circuit whose noise sits at the
physical gate locations with the calibrated error rates, so matching weights
include hook-error signatures and device nonuniformity.

Scope: 14-site abstract-routed Steane cultivation transferred to the rotated
d=5 surface code with either merge schedule, destructive X or Z full-data
readout, and any number of post-transfer rounds. Each decoder models one
fixed physical circuit, so a twirled job is decoded one twirl variant at a
time; the circuit must have an identity routing permutation. The decoder of
Google's grafting circuits (``google/decoder/``) reuses the public wire
alignment, frame verification, and calibrated suffix conversion.

Terminology: the Steane input (injection, QED, and cultivation) is replaced by
a noiseless Stim stand-in; each Steane wire's Steane cut is where the input ends
in the transpiled circuit; the suffix is every later physical operation, which
is converted with calibrated noise.

Call structure::

    CircuitLevelDecoder(submitted_circuit, calibration=..., tomography_config=...,
                        measurement_bit_layout=..., readout_basis=...,
                        persistent_surface_x_post_selection=...,
                        complementary_gap=...)
    ├── _build_calibrated_stim_model(...) -> _CalibratedStimModel
    │   ├── rebuild the logical circuit and require the saved bit layout
    │   ├── align_wire_operations(...)        # match anchors and group gate blocks
    │   ├── verify_frames_and_locate_steane_cuts(...)  # verify local frames
    │   ├── noiseless Stim stand-in for the Steane input
    │   ├── append_calibrated_suffix(...)      # physical suffix with calibrated noise
    │   └── shared labeled detectors, logical observable, optional gap detector
    └── DetectorErrorModelShotAnalyzer

    decoder.decode_counts(counts)              # without the gap detector
    decoder.decode_counts(counts, thresholds)  # with it: one decode, every cutoff
    └── (accepted logical-zero shots, accepted logical-one shots), per threshold

Not modeled: noise before the Steane cut, correlated readout/reset errors,
leakage, crosstalk, coherent errors, readout asymmetry, calibration drift, and
dynamical-decoupling suppression beyond the Pauli-twirled idle channel.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import stim
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import CZGate, HGate, SXGate, XGate
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGOpNode
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.transpiler import InstructionProperties, PassManager
from qiskit.transpiler.passes import RemoveBarriers, RemoveResetInZeroState

from msc.noise.fault_tolerance_stim import DetectorErrorModelShotAnalyzer
from msc.state_transfer import common
from msc.state_transfer import helper_stim as transfer_stim_helper
from msc.state_transfer import to_d5_rotated_qiskit as d5
from msc.steane_code import magic_state_cultivation_stim as cultivation_stim
from msc.steane_code.state_injection_common import NUM_QUBITS_PATCH_14_SITES
from msc.utils import circuit_stim
from msc.utils.normalize import normalize_pauli_char, qiskit_bit_values

if TYPE_CHECKING:
    from msc.experiment.results_io import DeviceCalibration
    from msc.state_transfer.tomography_qiskit import TyTomographyConfig

# Families measured by injection, post-injection QED, and cultivation. The
# noiseless Stim stand-in replaces every operation that produced them.
_STEANE_INPUT_FAMILIES: tuple[str, ...] = ("enc_flags", "steane_qed", "cult")
_UNITARY_TOLERANCE = 1e-8
_BATCH_SIZE = 10_000


# =============================================================================
# Decoder
# =============================================================================


class CircuitLevelDecoder:
    """Decode one saved X/Z tomography point with a calibrated circuit-level model.

    ``tomography_config`` fixes the point's post-transfer round count, so each
    LER memory duration builds its own decoder from its own submitted circuit
    and calibration.

    Construction builds the calibrated Stim model once, then compiles its
    detector postselection, decoder, and measurement-to-detection conversion.
    The stand-in's logical value is known, so the analyzer's accepted
    observable-disagreement bit is the corrected logical readout, not a
    hardware Ty-state failure label.

    ``decoder`` selects matching (the default) or Tesseract with Google's
    search settings and the analysis priority-queue limit
    (``TESSERACT_PQLIMIT``), the decoder of Google's grafting experiment, so
    the two protocols can be compared under one decoder.

    With ``complementary_gap``, the model's final gap detector is never
    hard-postselected; each shot is decoded once in both logical sectors, and
    every requested threshold rejects the shots whose sector-weight gap is
    below it.

    Attributes: ``circuit`` is the annotated calibrated Stim model,
    ``record_clbits`` maps each Stim record to its Qiskit classical bit,
    ``postselection_detector_indices`` are the hard-postselected detectors,
    ``steane_readout_count`` the leading readouts of the noiseless stand-in,
    ``analyzer`` owns postselection and decoding, ``reference_logical_bit``
    is the observable parity of the noiseless reference sample, and
    ``complementary_gap`` records whether the model has the gap detector.
    """

    def __init__(
        self,
        submitted_circuit: QuantumCircuit,
        *,
        calibration: DeviceCalibration,
        tomography_config: TyTomographyConfig,
        measurement_bit_layout: Mapping[str, Sequence[str]],
        readout_basis: common.SurfaceReadoutBasis,
        persistent_surface_x_post_selection: bool,
        complementary_gap: bool,
        decoder: Literal["matching", "tesseract"] = "matching",
    ) -> None:
        model = _build_calibrated_stim_model(
            submitted_circuit,
            calibration=calibration,
            tomography_config=tomography_config,
            measurement_bit_layout=measurement_bit_layout,
            readout_basis=readout_basis,
            add_complementary_gap_detector=complementary_gap,
        )
        self.complementary_gap = complementary_gap
        self.circuit = model.circuit
        self.record_clbits = model.record_clbits
        self.reference_logical_bit = model.reference_logical_bit
        self.steane_readout_count = model.steane_readout_count
        self.postselection_detector_indices = common.transfer_postselection_detector_indices(
            model.detector_roles,
            full_post_selection=tomography_config.full_post_selection,
            persistent_surface_x_post_selection=persistent_surface_x_post_selection,
        )
        self.analyzer = DetectorErrorModelShotAnalyzer(
            model.circuit,
            postselection_detector_indices=self.postselection_detector_indices,
            # Any threshold enables two-sector decoding; cutoffs apply per call.
            complementary_gap_postselection_threshold=0.0 if complementary_gap else None,
            decoder=decoder,
        )
        self._converter = model.circuit.compile_m2d_converter()

    def decode_counts(
        self,
        counts: Mapping[str, int],
        complementary_gap_thresholds: Sequence[float] | None = None,
    ) -> tuple[int, int] | tuple[tuple[int, int], ...]:
        """Return accepted logical-zero and logical-one shot counts.

        Without the gap detector, the decoder takes no thresholds and returns
        one pair. With it, ``complementary_gap_thresholds`` is required and one
        pair is returned per threshold from a single two-sector decoding of
        every shot. Unique count keys are decoded in batches and weighted by
        multiplicity; Qiskit keys list classical bit 0 last.
        """
        if (complementary_gap_thresholds is not None) != self.complementary_gap:
            raise ValueError(
                "complementary_gap_thresholds must be given exactly when the decoder "
                "has the complementary-gap detector."
            )
        return decode_count_keys(
            counts,
            analyzer=self.analyzer,
            converter=self._converter,
            record_clbits=self.record_clbits,
            reference_logical_bit=self.reference_logical_bit,
            complementary_gap_thresholds=complementary_gap_thresholds,
        )


def decode_count_keys(
    counts: Mapping[str, int],
    *,
    analyzer: DetectorErrorModelShotAnalyzer,
    converter: stim.CompiledMeasurementsToDetectionEventsConverter,
    record_clbits: Sequence[int],
    reference_logical_bit: int,
    complementary_gap_thresholds: Sequence[float] | None,
) -> tuple[int, int] | tuple[tuple[int, int], ...]:
    """Decode weighted hardware count keys against one calibrated model.

    Shared by this decoder and Google's grafting decoder. Unique keys are
    converted in batches to the model's detection events, postselected and
    decoded by ``analyzer``, and their accepted logical-zero and logical-one
    shots are weighted by multiplicity: one pair without thresholds, else one
    pair per threshold from a single two-sector decoding of every shot.
    """
    keys = list(counts)
    multiplicities = np.fromiter(counts.values(), dtype=np.int64, count=len(keys))
    record_clbits = list(record_clbits)
    shots_by_threshold = 0
    for start in range(0, len(keys), _BATCH_SIZE):
        # Hardware keys follow Qiskit bit order; Stim needs its record order.
        clbits = qiskit_bit_values(keys[start : start + _BATCH_SIZE])
        if clbits.shape[1] != len(record_clbits):
            raise ValueError(f"every count key must contain {len(record_clbits)} bits.")
        detection_events, observable_flips = converter.convert(
            measurements=clbits[:, record_clbits],
            separate_observables=True,
        )
        # One accepted-shot row per threshold.
        if complementary_gap_thresholds is None:
            is_accepted_shot, is_logical_error_shot = (
                analyzer.postselect_and_decode_samples(detection_events, observable_flips)
            )
            is_accepted_shot = is_accepted_shot[np.newaxis]
        else:
            is_accepted_shot, is_logical_error_shot, complementary_gaps = (
                analyzer.decode_complementary_gap_samples(
                    detection_events, observable_flips
                )
            )
            # Gaps are nan for hard-rejected shots, so every cutoff rejects them.
            is_accepted_shot = is_accepted_shot & (
                complementary_gaps >= np.asarray(complementary_gap_thresholds)[:, np.newaxis]
            )
        # The analyzer reports disagreement with the noiseless reference, so
        # undo that offset to recover the logical outcome.
        is_logical_one_shot = is_logical_error_shot ^ bool(reference_logical_bit)
        batch_multiplicities = multiplicities[start : start + _BATCH_SIZE]
        shots_by_threshold = shots_by_threshold + np.stack(
            (
                (is_accepted_shot & ~is_logical_one_shot) @ batch_multiplicities,
                (is_accepted_shot & is_logical_one_shot) @ batch_multiplicities,
            ),
            axis=1,
        )
    shots = tuple((int(zero), int(one)) for zero, one in shots_by_threshold)
    return shots[0] if complementary_gap_thresholds is None else shots


# =============================================================================
# Layout selection: hard-postselected circuit qubits
# =============================================================================


def layout_postselected_qubits(
    logical_circuits_by_basis: Mapping[str, QuantumCircuit],
    build_decoder: Callable[[QuantumCircuit, DeviceCalibration, str], Any],
) -> frozenset[int]:
    """Hard-postselected qubits of one layout point's tomography circuits, for layout selection.

    No device layout exists yet, so each basis circuit is submitted to a
    generic all-to-all device and ``build_decoder(submission, calibration,
    basis)`` builds its calibrated model (this module's decoder or Google's
    Tesseract decoder). The bases share one layout, so their sets are united.
    """
    backend, calibration = generic_calibrated_device(
        next(iter(logical_circuits_by_basis.values())).num_qubits
    )
    postselected_qubits: set[int] = set()
    for basis, logical_circuit in logical_circuits_by_basis.items():
        submission = submit_to_generic_device(logical_circuit, backend)
        decoder = build_decoder(submission, calibration, basis)
        postselected_qubits |= postselected_circuit_qubits(
            submission,
            decoder.circuit,
            decoder.postselection_detector_indices,
            decoder.steane_readout_count,
        )
    return frozenset(postselected_qubits)


def generic_calibrated_device(num_qubits: int) -> tuple[GenericBackendV2, DeviceCalibration]:
    """A generic all-to-all device, with resets timed like readouts, and its calibration.

    Lets a logical circuit be submitted locally on its own qubit indices, so its
    calibrated model exists before any device layout: for the postselected
    qubits of layout selection and for decoder tests.
    """
    # Imported here: results_io imports the analysis entry points that import this bridge.
    from msc.experiment.results_io import DeviceCalibration

    backend = GenericBackendV2(
        num_qubits,
        basis_gates=["cz", "sx", "x", "rz"],
        coupling_map=[[a, b] for a in range(num_qubits) for b in range(num_qubits) if a != b],
        seed=1,
    )
    target = backend.target
    for qubit in range(num_qubits):
        target.update_instruction_properties(
            "reset",
            (qubit,),
            InstructionProperties(duration=target["measure"][(qubit,)].duration),
        )
    calibration = DeviceCalibration(
        dt=target.dt,
        relaxation_times=tuple((qubit.t1, qubit.t2) for qubit in target.qubit_properties),
        errors={
            (name, qubits): properties.error
            for name in target.operation_names
            for qubits, properties in target[name].items()
        },
        durations={
            (name, qubits): properties.duration
            for name in target.operation_names
            for qubits, properties in target[name].items()
        },
    )
    return backend, calibration


def submit_to_generic_device(
    logical_circuit: QuantumCircuit, backend: GenericBackendV2
) -> QuantumCircuit:
    """Transpile like a submission on the identity layout: the submission-input rewrite, then delays.

    Hardware preprocessing drops resets on fresh wires, as the conversion
    expects; barriers only constrain scheduling and would misalign the wire
    operations, so they are dropped too; the calibrated suffix conversion
    requires the scheduled delays.
    """
    from msc.experiment.runner import prepare_submission_input

    return transpile(
        PassManager([RemoveBarriers(), RemoveResetInZeroState()]).run(
            prepare_submission_input(
                logical_circuit, backend.target, align_steane_and_merge_readouts=False
            )
        ),
        backend,
        initial_layout=list(range(logical_circuit.num_qubits)),
        optimization_level=0,
        scheduling_method="asap",
        seed_transpiler=1,
    )


def postselected_circuit_qubits(
    submitted_circuit: QuantumCircuit,
    model_circuit: stim.Circuit,
    postselection_detector_indices: Sequence[int],
    steane_readout_count: int,
) -> frozenset[int]:
    """Circuit qubits whose faults can flip a hard-postselected detector.

    Layout selection weights these qubits' operations up: a fault on them
    discards the shot, while one elsewhere feeds the decoder. The Steane stage
    is a noiseless stand-in in the model, so its qubits are read off the
    submitted circuit: every qubit of a two-qubit gate or readout preceding a
    readout into the first ``steane_readout_count`` classical bits, which both
    decoders assert the stand-in owns. Beyond it, they are the
    qubits of the circuit error locations behind the detector-error-model
    errors that touch a postselected detector. Device wires map back to circuit
    qubits through the submitted layout.
    """
    dag = circuit_to_dag(submitted_circuit)
    steane_readouts = [
        node
        for node in dag.op_nodes()
        if node.op.name.startswith("measure")
        and submitted_circuit.find_bit(node.cargs[0]).index < steane_readout_count
    ]
    steane_nodes = set(steane_readouts).union(*(dag.ancestors(node) for node in steane_readouts))
    wires = {
        submitted_circuit.find_bit(qubit).index
        for node in steane_nodes
        if isinstance(node, DAGOpNode)
        and (node.op.num_qubits == 2 or node.op.name.startswith("measure"))
        for qubit in node.qargs
    }
    postselected = set(postselection_detector_indices)
    postselected_errors = stim.DetectorErrorModel()
    for instruction in model_circuit.detector_error_model(decompose_errors=False).flattened():
        if instruction.type == "error" and any(
            target.is_relative_detector_id() and target.val in postselected
            for target in instruction.targets_copy()
        ):
            postselected_errors.append(instruction)
    for explanation in model_circuit.explain_detector_error_model_errors(
        dem_filter=postselected_errors, reduce_to_one_representative_error=False
    ):
        for location in explanation.circuit_error_locations:
            wires.update(target.gate_target.value for target in location.flipped_pauli_product)
            if location.flipped_measurement is not None:
                wires.update(
                    target.gate_target.value
                    for target in location.flipped_measurement.observable
                )
    layout = submitted_circuit.layout
    physical_qubits = (
        list(range(submitted_circuit.num_qubits))
        if layout is None
        else layout.initial_index_layout(filter_ancillas=True)
    )
    return frozenset(physical_qubits.index(wire) for wire in wires)


# =============================================================================
# Calibrated Stim model
# =============================================================================


@dataclass(frozen=True)
class _CalibratedStimModel:
    """The detector-annotated Stim circuit of one tomography point and its record contract.

    ``record_clbits[r]`` is the Qiskit classical bit of Stim measurement record
    ``r``. ``detector_roles`` lists the postselection role of every detector in
    circuit order, except an optional final complementary-gap detector, which
    has no role and is never hard-postselected. ``steane_readout_count`` is how
    many leading readouts the noiseless stand-in owns. ``reference_logical_bit``
    is the observable parity of the noiseless reference sample.
    """

    circuit: stim.Circuit
    record_clbits: tuple[int, ...]
    detector_roles: tuple[common.TransferDetectorRole, ...]
    steane_readout_count: int
    reference_logical_bit: int


def _build_calibrated_stim_model(
    submitted_circuit: QuantumCircuit,
    *,
    calibration: DeviceCalibration,
    tomography_config: TyTomographyConfig,
    measurement_bit_layout: Mapping[str, Sequence[str]],
    readout_basis: common.SurfaceReadoutBasis,
    add_complementary_gap_detector: bool,
) -> _CalibratedStimModel:
    """Convert one submitted tomography circuit into a calibrated, annotated Stim model.

    The Steane input is replaced by the noiseless Stim builder: encoded ``|+>``
    with an X projection for X readout, encoded ``|0>`` with a Z projection for
    Z readout. Its records keep their labels and stay hard-postselected. Every
    suffix operation is converted in order, including surface and interface
    operations that overlap cultivation in time. The optional gap detector
    contains exactly the observable's readout and transfer-frame records.
    """
    # Fail fast on inputs this conversion cannot follow; the layout checked
    # here is also what later maps logical qubits onto device wires.
    config = tomography_config
    readout_basis = normalize_pauli_char(readout_basis)
    if readout_basis not in ("X", "Z"):
        raise ValueError("circuit-level decoding supports X or Z full-data readout.")
    if (config.target_distance, config.patch_sites) != (5, NUM_QUBITS_PATCH_14_SITES):
        raise ValueError("circuit-level decoding supports only 14-site transfer to d=5.")
    layout = submitted_circuit.layout
    if layout is None or layout.routing_permutation() != list(range(submitted_circuit.num_qubits)):
        raise ValueError("the submitted circuit needs a layout without routing permutation.")

    # The saved bit layout is what gives hardware bits a protocol meaning. Rebuild
    # the logical circuit from the config and require the same labeled
    # records, so every later step addresses measurements by label, not bit.
    spec = d5.TRANSFER_SPEC
    post_transfer_stabilizer_rounds = config.resolved_post_transfer_stabilizer_rounds
    bit_layout_options = dict(
        patch_sites=config.patch_sites,
        merge_schedule=config.merge_schedule,
        measure_encoder_flags=config.measure_encoder_flags,
        measure_cultivation_flags=config.measure_cultivation_flags,
        post_injection_qed_checks=config.resolved_post_injection_qed_checks,
    )
    logical_circuit = d5.build_steane_to_rotated_d5_lattice_surgery_circuit(
        **bit_layout_options,
        surface_readout_basis=readout_basis,
        full_post_selection=config.full_post_selection,
        post_transfer_stabilizer_rounds=post_transfer_stabilizer_rounds,
        add_barriers=False,
    )
    bit_layout = common.transfer_measurement_labels(
        spec,
        **bit_layout_options,
        include_cultivation=True,
        surface_readout_basis=readout_basis,
        post_transfer_stabilizer_rounds=post_transfer_stabilizer_rounds,
        full_data_readout=True,
    )
    if [(family, tuple(labels)) for family, labels in measurement_bit_layout.items()] != list(
        bit_layout.items()
    ) or submitted_circuit.num_clbits != logical_circuit.num_clbits:
        raise ValueError("the saved measurement bit layout does not match the rebuilt protocol.")
    clbit_records = [(family, label) for family, labels in bit_layout.items() for label in labels]
    clbit_by_record = {record: clbit for clbit, record in enumerate(clbit_records)}

    # Decide what the noiseless stand-in replaces: the logical Steane input
    # before the first surface or interface operation. It must own exactly the
    # injection, QED, and cultivation records, so nothing the transfer needs
    # is lost and nothing replaced leaks into the noisy suffix.
    steane_input_length = next(
        index
        for index, instruction in enumerate(logical_circuit.data)
        if any(
            logical_circuit.find_bit(qubit).index >= NUM_QUBITS_PATCH_14_SITES
            for qubit in instruction.qubits
        )
    )
    steane_input_records = {
        record for record in clbit_records if record[0] in _STEANE_INPUT_FAMILIES
    }
    if {
        clbit_records[logical_circuit.find_bit(clbit).index]
        for instruction in logical_circuit.data[:steane_input_length]
        for clbit in instruction.clbits
    } != steane_input_records:
        raise ValueError("the Steane input does not own exactly its QED/cultivation records.")

    # Locate the Steane cut in the transpiled circuit. Transpilation reshuffles
    # single-qubit gates, so the cut and the gate frame there are found by
    # aligning logical and physical operations wire by wire.
    physical_by_virtual = layout.initial_index_layout()[: logical_circuit.num_qubits]
    steane_wires = physical_by_virtual[:NUM_QUBITS_PATCH_14_SITES]
    logical_wires, physical_wires = align_wire_operations(
        logical_circuit, submitted_circuit, physical_by_virtual
    )
    steane_cuts = verify_frames_and_locate_steane_cuts(
        logical_wires,
        physical_wires,
        steane_wires=steane_wires,
        steane_input_length=steane_input_length,
    )

    # Model the Steane input noiselessly with a Stim-simulable state. A
    # readout-basis eigenstate stands in for Ty because decoding needs only a
    # known ideal logical value, which the reference sample supplies below.
    circuit = stim.Circuit()
    cultivation_stim.append_14_site_patch_injection_and_cultivation(
        circuit,
        add_qed_round=True,
        patch_qubits=steane_wires,
        injected_state="+" if readout_basis == "X" else "0",
        projection_axis=readout_basis,
        encoder="abstract_routed",
        measure_encoder_flags=config.measure_encoder_flags,
        measure_cultivation_flags=config.measure_cultivation_flags,
        post_injection_qed_checks=config.resolved_post_injection_qed_checks,
        add_ticks=False,
    )

    # The stand-in must be a drop-in replacement, with the same records and
    # wires, so its measurements take over the hardware bits they replace.
    standin_layout = transfer_stim_helper.transfer_measurement_layout(
        spec,
        include_cultivation=True,
        **bit_layout_options,
    )
    standin_records = [
        record
        for _, record in sorted(
            (position, (family, label))
            for family in _STEANE_INPUT_FAMILIES
            if family in standin_layout.labels
            for label, position in standin_layout.by_label(family).items()
        )
    ]
    standin_wires = {
        target.value
        for instruction in circuit
        if instruction.name != "QUBIT_COORDS"
        for target in instruction.targets_copy()
        if target.is_qubit_target
    }
    if (
        circuit.num_measurements != len(standin_records)
        or set(standin_records) != steane_input_records
        or standin_wires != set(steane_cuts)
    ):
        raise RuntimeError("the Steane stand-in does not replace exactly the Steane input.")

    # Convert the suffix with calibrated noise. Joining the stand-in and suffix
    # record streams tells decoding which hardware bit feeds each Stim
    # measurement, and lets the label-based detectors below find their
    # measurements.
    standin_detector_count = circuit.num_detectors
    record_clbits = (
        *(clbit_by_record[record] for record in standin_records),
        *append_calibrated_suffix(
            circuit,
            submitted_circuit,
            steane_cuts=steane_cuts,
            active_wires=physical_by_virtual,
            calibration=calibration,
        ),
    )
    if sorted(record_clbits) != list(range(len(clbit_records))):
        raise RuntimeError("converted records do not cover every classical bit once.")
    if sorted(record_clbits[: len(standin_records)]) != list(range(len(standin_records))):
        raise RuntimeError("the Steane stand-in does not own the leading classical bits.")
    record_by_label = {clbit_records[clbit]: record for record, clbit in enumerate(record_clbits)}

    # Annotate with the same label-based detectors and transfer-frame-corrected
    # observable as the logical transfer, so circuit-level and phenomenological
    # decoding postselect and read out the same parities.
    detectors = (
        *(
            (((family, label),), "protocol")
            for family in common.STEANE_POSTSELECTION_FAMILIES
            if family not in _STEANE_INPUT_FAMILIES
            for label in bit_layout.get(family, ())
        ),
        *common.transfer_detectors(spec),
        *common.post_transfer_detectors(
            spec,
            post_transfer_stabilizer_rounds=post_transfer_stabilizer_rounds,
        ),
        *common.readout_boundary_detectors(
            spec,
            surface_readout_basis=readout_basis,
            post_transfer_stabilizer_rounds=post_transfer_stabilizer_rounds,
        ),
    )
    for records, _ in detectors:
        circuit_stim.append_detector(
            circuit,
            measurement_positions=tuple(record_by_label[record] for record in records),
        )
    logical_support = spec.logical_x if readout_basis == "X" else spec.logical_z
    observable_positions = tuple(
        record_by_label[record]
        for record in (
            *((f"surface_data_{readout_basis.lower()}", label) for label in logical_support),
            *common.transfer_logical_frame_records(spec, readout_basis),
        )
    )
    circuit_stim.append_observable_include(circuit, measurement_positions=observable_positions)

    # Sanity-check the conversion: a frame or sign mistake shows up as a
    # detector that fires without noise. The observable's noiseless parity is
    # the stand-in's known logical value, against which shots are decoded.
    reference_sample = circuit.reference_sample()
    if any(
        sum(reference_sample[record_by_label[record]] for record in records) % 2
        for records, _ in detectors
    ):
        raise RuntimeError("a converted detector has a nonzero ideal parity.")
    if add_complementary_gap_detector:
        circuit_stim.append_detector(circuit, measurement_positions=observable_positions)
    return _CalibratedStimModel(
        circuit=circuit,
        record_clbits=record_clbits,
        steane_readout_count=len(steane_input_records),
        detector_roles=("protocol",) * standin_detector_count
        + tuple(role for _, role in detectors),
        reference_logical_bit=int(sum(reference_sample[list(observable_positions)]) % 2),
    )


# =============================================================================
# Wire alignment
# =============================================================================


@dataclass(frozen=True)
class _WireOperation:
    """One single-qubit gate (``unitary``) or anchor on one wire.

    Anchors are CZs keyed by partner, Z measurements keyed by classical bit,
    and resets; ``reset_after_measure`` follows a measurement directly.
    ``position`` indexes the wire's operation list, so a composite
    ``measure_reset_2`` supplies two anchors at one position.
    """

    anchor: tuple[object, ...] | None
    unitary: np.ndarray | None
    instruction_index: int
    position: int


@dataclass
class _AnchoredWireOperations:
    """Keep one wire's anchors and intervening gate blocks in circuit order.

    ``blocks[i]`` precedes ``anchors[i]``; the final block follows the last
    anchor. ``operation_count`` counts the circuit instructions on this wire, so a
    composite measurement/reset still occupies one position despite supplying two
    anchors.
    """

    anchors: list[_WireOperation]
    blocks: list[list[_WireOperation]]
    operation_count: int = 0


def align_wire_operations(
    logical_circuit: QuantumCircuit,
    submitted_circuit: QuantumCircuit,
    physical_by_virtual: Sequence[int],
) -> tuple[dict[int, _AnchoredWireOperations], dict[int, _AnchoredWireOperations]]:
    """Pair each wire's logical and physical operations after matching their anchors.

    Logical CX becomes H-CZ-H on its target and logical CZ stays a CZ, a
    leading reset on a fresh wire is dropped, and a measurement followed by a
    reset matches either a composite measurement/reset or separate operations.
    Each logical two-qubit gate must pair with the same physical CZ from both
    wires. Device wires carrying only delays are unused and excluded.
    """
    # Every wire doing real work must be a layout wire; otherwise the per-wire
    # comparison below would silently miss operations.
    if {
        submitted_circuit.find_bit(qubit).index
        for instruction in submitted_circuit.data
        if instruction.operation.name not in ("barrier", "delay")
        for qubit in instruction.qubits
    } != set(physical_by_virtual):
        raise ValueError("the submitted circuit's active wires differ from its initial layout.")
    logical_operations = _split_wire_operations(logical_circuit, physical_by_virtual, logical=True)
    physical_operations = _split_wire_operations(
        submitted_circuit,
        range(submitted_circuit.num_qubits),
        logical=False,
    )

    # Anchors are what transpilation cannot move or merge away, so they must
    # agree one to one on every wire. A CX spans two wires, so both must match
    # it to the same physical CZ.
    physical_cz_by_logical_cx: dict[int, int] = {}
    for wire in physical_by_virtual:
        logical_anchors = logical_operations[wire].anchors
        physical_anchors = physical_operations[wire].anchors
        if [op.anchor for op in logical_anchors] != [op.anchor for op in physical_anchors]:
            raise ValueError(f"wire {wire} anchors differ between logical and submitted circuits.")
        for logical_anchor, physical_anchor in zip(logical_anchors, physical_anchors, strict=True):
            if (
                logical_anchor.anchor[0] == "cz"
                and physical_cz_by_logical_cx.setdefault(
                    logical_anchor.instruction_index,
                    physical_anchor.instruction_index,
                )
                != physical_anchor.instruction_index
            ):
                raise ValueError("a logical CX matches different physical CZs on its wires.")
    return logical_operations, {wire: physical_operations[wire] for wire in physical_by_virtual}


def _split_wire_operations(
    circuit: QuantumCircuit,
    wire_labels: Sequence[int],
    *,
    logical: bool,
) -> dict[int, _AnchoredWireOperations]:
    """Split a circuit into ordered single-qubit-gate and anchor operations per physical wire.

    ``wire_labels`` maps each circuit qubit index to its physical wire. Logical
    circuits contain CX or CZ, measurements, resets, and named single-qubit gates;
    submitted circuits contain the backend's native CZ, ``rz``, ``sx``, ``x``,
    measurements, resets, and delays. Delays are identity gates here because
    only their duration matters, which the conversion reads separately.
    """
    # Track each wire's operation positions, which conversion later compares
    # with the Steane cut, and its previous operation, which decides how a
    # reset is anchored.
    operations_by_wire = {wire: _AnchoredWireOperations([], [[]]) for wire in wire_labels}
    wire_by_qubit = dict(zip(circuit.qubits, wire_labels, strict=True))
    clbit_indices = {clbit: index for index, clbit in enumerate(circuit.clbits)}
    previous_names: dict[int, str | None] = dict.fromkeys(wire_labels)
    for instruction_index, instruction in enumerate(circuit.data):
        operation = instruction.operation
        name = operation.name
        if name == "barrier":
            continue
        wires = [wire_by_qubit[qubit] for qubit in instruction.qubits]
        clbits = tuple(clbit_indices[clbit] for clbit in instruction.clbits)
        for qubit_position, wire in enumerate(wires):
            wire_operations = operations_by_wire[wire]
            position = wire_operations.operation_count
            wire_operations.operation_count += 1
            previous_name = previous_names[wire]
            previous_names[wire] = name
            # Anchors (CZ, measurement, reset) must match exactly; every other
            # operation becomes a matrix compared only up to the local frame.
            if name == "cz" or (logical and name == "cx"):
                entries = [(("cz", wires[1 - qubit_position]), None)]
                if name == "cx" and qubit_position == 1:
                    entries = [(None, _HADAMARD), *entries, (None, _HADAMARD)]
            elif name in ("measure", "measure_2"):
                entries = [(("measure", clbits), None)]
            elif name == "measure_reset_2" and not logical:
                entries = [(("measure", clbits), None), (("reset_after_measure",), None)]
            elif name == "reset":
                if previous_name is None and logical:
                    continue  # A fresh wire already starts in |0>.
                is_after_measure = previous_name in ("measure", "measure_2", "measure_reset_2")
                entries = [(("reset_after_measure",) if is_after_measure else ("reset",), None)]
            elif name == "delay" and not logical:
                entries = [(None, _IDENTITY)]
            elif len(wires) == 1 and (logical or name in ("rz", "sx", "x")):
                entries = [(None, np.asarray(operation.to_matrix(), dtype=complex))]
            else:
                raise ValueError(f"unsupported {'logical' if logical else 'submitted'} {name!r}.")
            # Split each wire into blocks at its anchors. Between consecutive anchors,
            # logical and physical gates may differ individually but must agree as a
            # block up to the local frame.
            for anchor, unitary in entries:
                wire_operation = _WireOperation(anchor, unitary, instruction_index, position)
                if anchor is None:
                    wire_operations.blocks[-1].append(wire_operation)
                else:
                    wire_operations.anchors.append(wire_operation)
                    wire_operations.blocks.append([])
    return {
        wire: operations
        for wire, operations in operations_by_wire.items()
        if operations.anchors or operations.blocks[0]
    }


# =============================================================================
# Local frames and the Steane cut
# =============================================================================


@dataclass(frozen=True)
class _SteaneCut:
    """Where the suffix starts on one Steane wire and the local frame it starts in.

    At operation ``first_suffix_position`` the physical state is
    ``Z(initial_phase) X**initial_x_offset`` applied to the stand-in state, so
    conversion emits the X offset noiselessly and carries the Z rotation as
    pending. ``reset_only`` marks a composite measurement/reset whose
    measurement belongs to the replaced Steane input.
    """

    first_suffix_position: int
    initial_phase: float
    initial_x_offset: int
    reset_only: bool


def verify_frames_and_locate_steane_cuts(
    logical_wires: Mapping[int, _AnchoredWireOperations],
    physical_wires: Mapping[int, _AnchoredWireOperations],
    *,
    steane_wires: Sequence[int],
    steane_input_length: int,
) -> dict[int, _SteaneCut]:
    """Verify transpiled == logical up to local frames and locate each Steane cut.

    Transpilation may move virtual Z rotations and resynthesize a maximal run
    of consecutive CZs on one pair together with neighboring single-qubit gates,
    so the physical and logical circuits agree only up to a local frame ``F`` on
    each wire (physical = ``F`` · logical). Walking all wires jointly in logical
    order, each single-qubit block updates ``F``, each CZ run must map
    ``F_a ⊗ F_b`` to another product frame, and each Z measurement requires a
    diagonal frame so its reported bit agrees. Fresh and reset wires have
    ``F = I``.

    On a Steane wire, the block containing the cut is split at its last ``sx``
    pulse: those pulses implement the full logical residual before the cut,
    including the Ty rotations and their Clifford parts, and are removed. The
    frame there must be a Z rotation times an optional X, so the remaining
    delays, DD pulses, and suffix gates convert to Clifford operations. A
    composite measurement/reset whose measurement ends the input keeps only its
    reset. Returns one Steane cut per Steane wire that the input touches.

    Comparing a whole CZ run as one unit is kept only to decode jobs compiled
    before commit 38b31de, whose transpiler resynthesized two-qubit blocks. Jobs
    compiled since keep every CZ, and comparing each CZ on its own also
    verifies them.
    """
    # Express each Steane wire's cut in the same block coordinates, so the
    # frame walk below knows which block to split.
    cut_locations: dict[int, tuple[int, int]] = {}
    for wire in steane_wires:
        logical_wire = logical_wires[wire]
        input_anchor_count = sum(
            anchor.instruction_index < steane_input_length for anchor in logical_wire.anchors
        )
        input_gate_count = sum(
            gate.instruction_index < steane_input_length
            for gate in logical_wire.blocks[input_anchor_count]
        )
        if input_anchor_count or input_gate_count:
            cut_locations[wire] = input_anchor_count, input_gate_count

    # Walk state: each wire's running local frame, and the Steane cuts recorded
    # as the walk passes them.
    frames = dict.fromkeys(logical_wires, _IDENTITY)
    steane_cuts: dict[int, _SteaneCut] = {}

    def advance_block(wire: int, block_index: int) -> None:
        """Update ``F`` to ``U_physical · F · U_logical^dagger``, splitting a cut block first."""
        logical_block = logical_wires[wire].blocks[block_index]
        physical_block = physical_wires[wire].blocks[block_index]
        cut_location = cut_locations.get(wire)
        if cut_location is not None and cut_location[0] == block_index:
            steane_cuts[wire] = _split_block_at_steane_cut(
                frames[wire],
                replaced_logical_gates=logical_block[: cut_location[1]],
                physical_wire=physical_wires[wire],
                block_index=block_index,
            )
        frames[wire] = (
            _block_unitary(physical_block) @ frames[wire] @ _block_unitary(logical_block).conj().T
        )

    # Visit anchors in logical order, because a CZ couples two wires' frames and
    # both must be current. A CZ run is compared as one unit, so its later
    # anchors are skipped when the loop reaches them.
    anchor_events: dict[int, list[tuple[int, int]]] = {}
    for wire, logical_wire in logical_wires.items():
        for block_index, anchor in enumerate(logical_wire.anchors):
            anchor_events.setdefault(anchor.instruction_index, []).append((wire, block_index))
    consumed_anchors = dict.fromkeys(logical_wires, 0)
    for instruction_index in sorted(anchor_events):
        participants = anchor_events[instruction_index]
        if any(consumed_anchors[wire] > block_index for wire, block_index in participants):
            continue  # Already compared inside a CZ run.
        for wire, block_index in participants:
            advance_block(wire, block_index)
        first_wire, first_block_index = participants[0]
        anchor_kind = logical_wires[first_wire].anchors[first_block_index].anchor[0]
        if anchor_kind == "cz":
            # Kept only for jobs compiled before 38b31de, whose transpiler resynthesized
            # a whole run of CZs on one pair: extend over the maximal run.
            (wire_a, block_a), (wire_b, block_b) = participants
            run_length = 1
            while (
                block_a + run_length < len(logical_wires[wire_a].anchors)
                and block_b + run_length < len(logical_wires[wire_b].anchors)
                and logical_wires[wire_a].anchors[block_a + run_length].instruction_index
                == logical_wires[wire_b].anchors[block_b + run_length].instruction_index
            ):
                run_length += 1
            if any(
                block_index < cut_locations[wire][0] < block_index + run_length
                for wire, block_index in ((wire_a, block_a), (wire_b, block_b))
                if wire in cut_locations
            ):
                raise ValueError("a resynthesized CZ run crosses the Steane cut.")
            # The physical run must equal the logical one up to local frames
            # on either side; an entangling remainder means they differ.
            runs = []
            for circuit_wires in (logical_wires, physical_wires):
                run = _CZ
                for offset in range(1, run_length):
                    run_block = np.kron(
                        _block_unitary(circuit_wires[wire_a].blocks[block_a + offset]),
                        _block_unitary(circuit_wires[wire_b].blocks[block_b + offset]),
                    )
                    run = _CZ @ run_block @ run
                runs.append(run)
            logical_run, physical_run = runs
            frames[wire_a], frames[wire_b] = _product_factors(
                physical_run @ np.kron(frames[wire_a], frames[wire_b]) @ logical_run.conj().T,
                context=f"CZ run on wires {wire_a} and {wire_b}",
            )
            consumed_anchors[wire_a] = block_a + run_length
            consumed_anchors[wire_b] = block_b + run_length
        elif anchor_kind in ("measure", "reset", "reset_after_measure"):
            ((wire, block_index),) = participants
            if anchor_kind == "measure" and not _is_diagonal(frames[wire]):
                raise ValueError(f"a measurement on wire {wire} differs from the logical one.")
            frames[wire] = _IDENTITY
            consumed_anchors[wire] = block_index + 1
        else:  # pragma: no cover - anchors are built by _split_wire_operations.
            raise ValueError(f"unsupported anchor kind {anchor_kind!r}.")
    # Close out the gates after each wire's last anchor, where a cut can lie.
    for wire, logical_wire in logical_wires.items():
        advance_block(wire, len(logical_wire.anchors))
    return steane_cuts


def _split_block_at_steane_cut(
    frame: np.ndarray,
    *,
    replaced_logical_gates: Sequence[_WireOperation],
    physical_wire: _AnchoredWireOperations,
    block_index: int,
) -> _SteaneCut:
    """Split the physical block containing one wire's logical Steane cut.

    ``frame`` is the local frame entering the block, and
    ``replaced_logical_gates`` are the block's logical gates before the cut.
    The previous and next anchors identify a composite measurement/reset that
    straddles the cut.
    """
    physical_block = physical_wire.blocks[block_index]
    anchors = physical_wire.anchors
    previous_physical_anchor = anchors[block_index - 1] if block_index else None
    next_physical_anchor = anchors[block_index] if block_index < len(anchors) else None
    # The cut separates a composite measurement from its reset: only the
    # reset belongs to the suffix.
    if (
        not physical_block
        and previous_physical_anchor is not None
        and next_physical_anchor is not None
        and previous_physical_anchor.anchor[0] == "measure"
        and next_physical_anchor.anchor == ("reset_after_measure",)
        and previous_physical_anchor.position == next_physical_anchor.position
    ):
        return _SteaneCut(next_physical_anchor.position, 0.0, 0, True)

    # Drop the pulses that implement the replaced logical gates, through the
    # block's last sx. The leftover frame must be Stim-convertible: a Z rotation
    # that conversion defers, times an optional X that it emits.
    head_length = 0
    if replaced_logical_gates:
        head_length = max(
            (
                index + 1
                for index, operation in enumerate(physical_block)
                if np.allclose(operation.unitary, _SQRT_X, atol=_UNITARY_TOLERANCE)
            ),
            default=0,
        )
    cut_frame = (
        _block_unitary(physical_block[:head_length])
        @ frame
        @ _block_unitary(replaced_logical_gates).conj().T
    )
    if _is_diagonal(cut_frame):
        initial_x_offset = 0
        z_frame = cut_frame
    elif _is_diagonal(cut_frame @ _PAULI_X):
        initial_x_offset = 1
        z_frame = cut_frame @ _PAULI_X
    else:
        raise ValueError("the frame at the Steane cut is not a Z rotation times an X.")

    # Conversion resumes at the first physical operation not dropped above.
    if head_length < len(physical_block):
        first_position = physical_block[head_length].position
    elif next_physical_anchor is not None:
        first_position = next_physical_anchor.position
    else:
        raise ValueError("a Steane wire has no suffix after its cut.")
    return _SteaneCut(
        first_suffix_position=first_position,
        initial_phase=float(np.angle(z_frame[1, 1] / z_frame[0, 0])),
        initial_x_offset=initial_x_offset,
        reset_only=False,
    )


def _product_factors(unitary: np.ndarray, *, context: str) -> tuple[np.ndarray, np.ndarray]:
    """Factor a two-qubit unitary as ``A ⊗ B``, rejecting entangling remainders."""
    realigned = unitary.reshape(2, 2, 2, 2).transpose(0, 2, 1, 3).reshape(4, 4)
    left, singular_values, right = np.linalg.svd(realigned)
    if singular_values[1] > _UNITARY_TOLERANCE * singular_values[0]:
        raise ValueError(f"the submitted {context} differs from the logical one.")
    scale = math.sqrt(singular_values[0])
    return (left[:, 0] * scale).reshape(2, 2), (right[0] * scale).reshape(2, 2)


def _block_unitary(operations: Sequence[_WireOperation]) -> np.ndarray:
    """Multiply single-qubit gates in circuit order (later gates on the left)."""
    unitary = _IDENTITY
    for operation in operations:
        unitary = operation.unitary @ unitary
    return unitary


def _is_diagonal(unitary: np.ndarray) -> bool:
    """Test off-diagonal entries relative to the matrix scale, since frames are unnormalized."""
    scale = np.abs(unitary).max()
    return max(abs(unitary[0, 1]), abs(unitary[1, 0])) < _UNITARY_TOLERANCE * scale


_IDENTITY = np.eye(2, dtype=complex)
_PAULI_X = XGate().to_matrix()
_HADAMARD = HGate().to_matrix()
_SQRT_X = SXGate().to_matrix()
_CZ = CZGate().to_matrix()


# =============================================================================
# Calibrated suffix conversion
# =============================================================================


def append_calibrated_suffix(
    circuit: stim.Circuit,
    submitted_circuit: QuantumCircuit,
    *,
    steane_cuts: Mapping[int, _SteaneCut],
    active_wires: Sequence[int],
    calibration: DeviceCalibration,
) -> list[int]:
    """Append the physical suffix with calibrated noise; return its measurement bits.

    Virtual ``rz`` angles accumulate through CZ and delays; ``x`` negates the
    pending angle; before ``sx`` the pending angle must be a Clifford Z rotation,
    which is emitted first. Z measurement and reset discard the pending angle.
    Deferred Z rotations commute with every noise channel below.

    A calibrated average gate infidelity ``r`` becomes ``DEPOLARIZE1(3r/2)``
    after ``sx``/``x`` and ``DEPOLARIZE2(5r/4)`` after ``cz``; it already
    includes gate relaxation. Measurements flip symmetrically at their readout
    error. Resets use an independent reset-error proxy at the paired readout
    error, not a derived feedforward model: ``M(p) R X_ERROR(p)`` for
    ``measure_reset_2`` and ``R X_ERROR(p)`` for ``reset``.

    Delays get the Pauli-twirled T1/T2 idle channel as independent X, Y, and Z
    flips, except on a wire still in its prepared ``|0>``, which amplitude
    damping and dephasing leave unchanged. Wire clocks must agree at every CZ,
    so every idle interval is an explicit delay.
    """
    # Running state per wire: the deferred virtual-Z angle, whether idle noise
    # applies yet, and the position and clock that locate the Steane cut and
    # check CZ synchronization. Converted measurements' hardware bits are
    # collected in Stim record order.
    lines: list[str] = []
    pending_phases: dict[int, float] = {}
    ground_states: dict[int, bool] = {}
    wire_positions = dict.fromkeys(active_wires, 0)
    wire_clocks = dict.fromkeys(active_wires, 0.0)
    qubit_indices = {qubit: index for index, qubit in enumerate(submitted_circuit.qubits)}
    clbit_indices = {clbit: index for index, clbit in enumerate(submitted_circuit.clbits)}
    suffix_clbits: list[int] = []
    for instruction in submitted_circuit.data:
        operation = instruction.operation
        name = operation.name
        wires = tuple(qubit_indices[qubit] for qubit in instruction.qubits)
        if name == "barrier" or any(wire not in wire_positions for wire in wires):
            continue  # Unused device wires carry only delays.

        # Advance timing even through the skipped Steane input, so suffix
        # operations stay aligned with the Steane cuts and the device schedule.
        if name != "delay":
            duration = calibration.operation_duration(name, wires)
        elif operation.unit == "dt":
            duration = float(operation.params[0]) * calibration.dt
        else:
            raise ValueError(f"unsupported delay unit {operation.unit!r}.")
        if max(wire_clocks[wire] for wire in wires) - min(wire_clocks[wire] for wire in wires) > (
            1e-12
        ):
            raise ValueError(f"{name}{wires} starts at different times on its wires.")
        positions = {wire: wire_positions[wire] for wire in wires}
        for wire in wires:
            wire_clocks[wire] += duration
            wire_positions[wire] += 1

        # Skip the replaced Steane input and start each suffix wire in its frame.
        in_suffix = [
            wire not in steane_cuts or positions[wire] >= steane_cuts[wire].first_suffix_position
            for wire in wires
        ]
        if not any(in_suffix):
            continue
        if not all(in_suffix):
            raise ValueError(f"{name}{wires} crosses the Steane cut.")
        for wire in wires:
            if wire not in pending_phases:
                steane_cut = steane_cuts.get(wire)
                if steane_cut is not None and steane_cut.initial_x_offset:
                    lines.append(f"X {wire}")  # Noiseless local-frame correction.
                pending_phases[wire] = 0.0 if steane_cut is None else steane_cut.initial_phase
                ground_states[wire] = steane_cut is None or steane_cut.reset_only

        wire = wires[0]
        if name == "rz":
            pending_phases[wire] += float(operation.params[0])
        elif name == "sx":
            # sx does not commute with the deferred Z rotation, so emit it first;
            # it must be Clifford for Stim.
            quarter_turns = pending_phases[wire] / (math.pi / 2)
            if abs(quarter_turns - round(quarter_turns)) > 1e-9:
                raise ValueError(f"a suffix sx on wire {wire} follows a non-Clifford rotation.")
            z_rotation = (None, "S", "Z", "S_DAG")[round(quarter_turns) % 4]
            if z_rotation is not None:
                lines.append(f"{z_rotation} {wire}")
            lines.append(f"SQRT_X {wire}")
            lines.append(f"DEPOLARIZE1({1.5 * _gate_error(calibration, name, wires)!r}) {wire}")
            pending_phases[wire] = 0.0
            ground_states[wire] = False
        elif name == "x":
            lines.append(f"X {wire}")
            lines.append(f"DEPOLARIZE1({1.5 * _gate_error(calibration, name, wires)!r}) {wire}")
            pending_phases[wire] = -pending_phases[wire]
            ground_states[wire] = False
        elif name == "cz":
            targets = " ".join(map(str, wires))
            lines.append(f"CZ {targets}")
            lines.append(f"DEPOLARIZE2({1.25 * _gate_error(calibration, name, wires)!r}) {targets}")
            for cz_wire in wires:
                ground_states[cz_wire] = False
        elif name in ("measure", "measure_2", "measure_reset_2", "reset"):
            # A composite straddling the Steane cut keeps only its reset; its
            # measurement belongs to the stand-in.
            readout_error = calibration.operation_error(name, wires)
            if not 0 <= readout_error <= 0.5:
                raise ValueError(f"{name}{wires} readout error {readout_error} exceeds 1/2.")
            is_reset_only = (
                wire in steane_cuts
                and steane_cuts[wire].reset_only
                and positions[wire] == steane_cuts[wire].first_suffix_position
            )
            if name != "reset" and not is_reset_only:
                lines.append(f"M({readout_error!r}) {wire}")
                suffix_clbits.extend(clbit_indices[clbit] for clbit in instruction.clbits)
            resets = name in ("measure_reset_2", "reset")
            if resets:
                lines.append(f"R {wire}")
                lines.append(f"X_ERROR({readout_error!r}) {wire}")
            pending_phases[wire] = 0.0
            ground_states[wire] = resets
        elif name == "delay":
            if not ground_states[wire]:
                for channel, probability in zip(
                    ("X_ERROR", "Y_ERROR", "Z_ERROR"),
                    _idle_pauli_error_probabilities(duration, *calibration.relaxation_times[wire]),
                    strict=True,
                ):
                    lines.append(f"{channel}({probability!r}) {wire}")
        else:
            raise ValueError(f"unsupported submitted operation {name!r}.")
    # Building from text avoids Stim's per-instruction Python append overhead.
    circuit += stim.Circuit("\n".join(lines))
    return suffix_clbits


def _gate_error(calibration: DeviceCalibration, name: str, wires: tuple[int, ...]) -> float:
    """Return a gate infidelity whose depolarizing channel is physical."""
    error = calibration.operation_error(name, wires)
    if not 0 <= error <= (0.5 if len(wires) == 1 else 0.75):
        raise ValueError(f"{name}{wires} infidelity {error} exceeds the depolarizing range.")
    return error


def _idle_pauli_error_probabilities(
    duration: float,
    t1: float | None,
    t2: float | None,
) -> tuple[float, float, float]:
    """Return independent X, Y, Z flips equal to the Pauli-twirled idle channel.

    The twirled amplitude-damping/dephasing channel has
    ``pX = pY = (1 - exp(-t/T1))/4`` and
    ``pZ = (1 + exp(-t/T1) - 2 exp(-t/T2))/4``. Matching its Pauli-transfer
    eigenvalues with independent flips gives ``qX = qY = (1 - exp(-t/(2 T1)))/2``
    and ``qZ = (1 - exp(-t/T2 + t/(2 T1)))/2``, nonnegative exactly when
    ``T2 <= 2 T1``. Separately fitted calibrations can report ``T2 > 2 T1``
    (for example q65 of ibm_phoenix on 2026-10-19); such a T2 is clamped to the
    pure-T1 limit ``2 T1``, leaving no dephasing beyond relaxation.
    """
    if t1 is None or t2 is None or not (0 < t1 < math.inf and 0 < t2 < math.inf):
        raise ValueError("idle relaxation requires finite positive T1 and T2.")
    t2 = min(t2, 2 * t1)
    transverse_flip = (1 - math.exp(-duration / (2 * t1))) / 2
    dephasing_flip = (1 - math.exp(-duration / t2 + duration / (2 * t1))) / 2
    return transverse_flip, transverse_flip, dephasing_flip
