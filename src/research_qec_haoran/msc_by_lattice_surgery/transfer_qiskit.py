from __future__ import annotations

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

from research_qec_haoran.msc_by_lattice_surgery.color_code_mapping_qiskit import (
    google_order_qubits_from_paper_order,
)
from research_qec_haoran.msc_by_lattice_surgery.common_qiskit import (
    CircuitBuildResult,
    measure_x_parity,
    measure_z_parity,
)
from research_qec_haoran.steane_code.magic_state_cultivation import append_steane_state_injection_encoder
from research_qec_haoran.surface_code.rotated_d3_layout_qiskit import (
    rotated_d3_surface_label_map,
)
from research_qec_haoran.surface_code.rotated_d3_qiskit import (
    append_prepare_plus_state,
    append_rotated_d3_z_stabilizer_measurements,
    append_surface_d3_stabilizer_round,
)


D3_TRANSFER_FINAL_PARITY_FORMULAS: dict[str, tuple[str, ...]] = {
    "X_0145ab": (
        "color_x_0_final",
        "color_x_1_final",
        "color_x_4_final",
        "color_x_5_final",
        "surface_x_ab_final",
    ),
    "X_0235": (
        "color_x_0_final",
        "color_x_2_final",
        "color_x_3_final",
        "color_x_5_final",
    ),
    "X_0246": (
        "color_x_0_final",
        "color_x_2_final",
        "color_x_4_final",
        "color_x_6_final",
    ),
    "X_146": (
        "color_x_1_final",
        "color_x_4_final",
        "color_x_6_final",
    ),
}


def build_d3_msc_ls_measurement_labels(
    *,
    include_surface_stabilizer_rounds: bool = True,
) -> tuple[str, ...]:
    """Return the classical-bit labels for the current d=3 transfer builder."""
    labels: list[str] = []
    labels.extend(("surface_init_z_abde", "surface_init_z_cf", "surface_init_z_dg", "surface_init_z_efhi"))

    for round_index in range(1, 4):
        labels.append(f"merge_z_1a_r{round_index}")
        labels.append(f"merge_z_35bc_r{round_index}")
        if include_surface_stabilizer_rounds:
            for name in ("z_abde", "z_cf", "z_dg", "z_efhi", "x_ab", "x_bcef", "x_degh", "x_hi"):
                labels.append(f"surface_{name}_r{round_index}")
        if round_index == 1:
            labels.extend(
                (
                    "color_z_0145_r1",
                    "color_z_0235_r1",
                    "color_x_0235_r1",
                    "color_z_0246_r1",
                    "color_x_0246_r1",
                )
            )
        if round_index == 2:
            labels.extend(
                (
                    "color_z_0145_r2",
                    "color_z_0235_r2",
                    "color_z_0246_r2",
                )
            )

    labels.append("surface_x_ab_final")
    labels.extend(f"color_x_{paper_index}_final" for paper_index in range(7))
    return tuple(labels)


def derive_d3_msc_ls_named_parities(measurements: dict[str, int]) -> dict[str, int]:
    """Derive the paper-named X parities from raw labeled measurement bits."""
    return {
        name: sum(measurements[label] for label in support) % 2
        for name, support in D3_TRANSFER_FINAL_PARITY_FORMULAS.items()
    }


def build_d3_msc_ls_transfer_circuit(
    *,
    injected_state: str = "H",
    include_surface_stabilizer_rounds: bool = True,
    add_barriers: bool = True,
) -> CircuitBuildResult:
    """Build a first-pass MSC-LS transfer circuit onto a rotated d=3 patch.

    The paper's detailed lattice-surgery figure is shown for
    ``d_intermediate = 5``. This builder keeps the same operator logic but
    truncates the top edge to the d=3 target labels ``a,b,c``.
    """
    labels = build_d3_msc_ls_measurement_labels(
        include_surface_stabilizer_rounds=include_surface_stabilizer_rounds,
    )

    color = QuantumRegister(7, "color")
    surface = QuantumRegister(9, "surface")
    merge_ancilla = QuantumRegister(1, "merge_anc")
    surface_z_ancillas = QuantumRegister(4, "surface_z_anc")
    surface_x_ancillas = QuantumRegister(4, "surface_x_anc")
    classical = ClassicalRegister(len(labels), "m")
    circuit = QuantumCircuit(color, surface, merge_ancilla, surface_z_ancillas, surface_x_ancillas, classical)

    label_to_surface_qubit = rotated_d3_surface_label_map(list(surface))
    google_order_color_qubits = google_order_qubits_from_paper_order(list(color))

    append_steane_state_injection_encoder(
        circuit,
        steane_qubits=google_order_color_qubits,
        injected_state=injected_state,
    )

    if add_barriers:
        circuit.barrier(label="accepted cultivated color state")

    bit_index = 0

    init_bits = list(classical[bit_index: bit_index + 4])
    append_prepare_plus_state(
        circuit,
        qubits=list(surface),
        add_barriers=add_barriers,
        barrier_label="surface d3 |+> product state",
    )
    append_rotated_d3_z_stabilizer_measurements(
        circuit,
        surface_qubits=list(surface),
        z_ancillas=list(surface_z_ancillas),
        classical_bits=init_bits,
        round_label="init",
        prepare_ancillas=False,
        add_barriers=add_barriers,
    )
    bit_index += 4

    color_z_0145 = [color[i] for i in (0, 1, 4, 5)]
    color_z_0235 = [color[i] for i in (0, 2, 3, 5)]
    color_x_0235 = [color[i] for i in (0, 2, 3, 5)]
    color_z_0246 = [color[i] for i in (0, 2, 4, 6)]
    color_x_0246 = [color[i] for i in (0, 2, 4, 6)]

    for round_index in range(1, 4):
        measure_z_parity(
            circuit,
            ancilla=merge_ancilla[0],
            data_qubits=[color[1], label_to_surface_qubit["a"]],
            classical_bit=classical[bit_index],
        )
        bit_index += 1
        if add_barriers:
            circuit.barrier(label=f"r{round_index} merge Z_1a")

        measure_z_parity(
            circuit,
            ancilla=merge_ancilla[0],
            data_qubits=[color[3], color[5], label_to_surface_qubit["b"], label_to_surface_qubit["c"]],
            classical_bit=classical[bit_index],
        )
        bit_index += 1
        if add_barriers:
            circuit.barrier(label=f"r{round_index} merge Z_35bc")

        if include_surface_stabilizer_rounds:
            z_bits = list(classical[bit_index: bit_index + 4])
            x_bits = list(classical[bit_index + 4: bit_index + 8])
            append_surface_d3_stabilizer_round(
                circuit,
                surface_qubits=list(surface),
                z_ancillas=list(surface_z_ancillas),
                x_ancillas=list(surface_x_ancillas),
                z_bits=z_bits,
                x_bits=x_bits,
                round_label=f"r{round_index} surface",
                prepare_z_ancillas=True,
                prepare_x_ancillas=(round_index > 1),
                add_barriers=add_barriers,
            )
            bit_index += 8

        if round_index == 1:
            for qubits, measure_fn in (
                (color_z_0145, measure_z_parity),
                (color_z_0235, measure_z_parity),
                (color_x_0235, measure_x_parity),
                (color_z_0246, measure_z_parity),
                (color_x_0246, measure_x_parity),
            ):
                measure_fn(
                    circuit,
                    ancilla=merge_ancilla[0],
                    data_qubits=qubits,
                    classical_bit=classical[bit_index],
                )
                bit_index += 1
            if add_barriers:
                circuit.barrier(label="color round 1")

        if round_index == 2:
            for qubits in (color_z_0145, color_z_0235, color_z_0246):
                measure_z_parity(
                    circuit,
                    ancilla=merge_ancilla[0],
                    data_qubits=qubits,
                    classical_bit=classical[bit_index],
                )
                bit_index += 1
            if add_barriers:
                circuit.barrier(label="color round 2")

    measure_x_parity(
        circuit,
        ancilla=merge_ancilla[0],
        data_qubits=[label_to_surface_qubit["a"], label_to_surface_qubit["b"]],
        classical_bit=classical[bit_index],
    )
    bit_index += 1

    if add_barriers:
        circuit.barrier(label="recover X_ab")

    for paper_index in range(7):
        circuit.h(color[paper_index])
        circuit.measure(color[paper_index], classical[bit_index])
        bit_index += 1

    if add_barriers:
        circuit.barrier(label="destructive color X readout")

    return CircuitBuildResult(circuit=circuit, measurement_labels=labels)
