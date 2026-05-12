from research_qec_haoran.surface_code.decoding import StabilizerFrame
from research_qec_haoran.surface_code.rotated_d3_qiskit import (
    ROTATED_D3_INIT_Z_LABELS,
    build_rotated_d3_logical_plus_initialization_circuit,
    rotated_d3_logical_plus_expectations,
    rotated_d3_z_syndrome_bits_from_measurements,
)


class TestStabilizerFrame:
    def test_record_and_current(self) -> None:
        frame = StabilizerFrame()
        frame.record("Z_abde", 1)
        assert frame.current("Z_abde") == 1

    def test_compare_without_update(self) -> None:
        frame = StabilizerFrame()
        frame.record_many({"Z_abde": 1, "Z_cf": 0})
        assert frame.compare("Z_abde", 1, update=False) == 0
        assert frame.compare("Z_cf", 1, update=False) == 1
        assert frame.current("Z_cf") == 0

    def test_advance_updates_reference(self) -> None:
        frame = StabilizerFrame()
        frame.record_many({"Z_abde": 1, "Z_cf": 0})
        deltas = frame.advance({"Z_abde": 0, "Z_cf": 0})
        assert deltas == {"Z_abde": 1, "Z_cf": 0}
        assert frame.as_dict() == {"Z_abde": 0, "Z_cf": 0}


def test_rotated_d3_initialization_labels() -> None:
    result = build_rotated_d3_logical_plus_initialization_circuit()
    assert result.measurement_labels == ROTATED_D3_INIT_Z_LABELS


def test_measurement_labels_convert_to_z_syndrome() -> None:
    syndrome = rotated_d3_z_syndrome_bits_from_measurements(
        {
            "init_z_abde": 1,
            "init_z_cf": 0,
            "init_z_dg": 1,
            "init_z_efhi": 0,
        }
    )
    assert syndrome == {
        "Z_abde": 1,
        "Z_cf": 0,
        "Z_dg": 1,
        "Z_efhi": 0,
    }


def test_nontrivial_z_syndrome_branch_preserves_logical_plus() -> None:
    expectations = rotated_d3_logical_plus_expectations(
        {
            "Z_abde": 1,
            "Z_cf": 0,
            "Z_dg": 1,
            "Z_efhi": 0,
        }
    )

    assert expectations["Z_abde"] < -0.999999
    assert expectations["Z_cf"] > 0.999999
    assert expectations["Z_dg"] < -0.999999
    assert expectations["Z_efhi"] > 0.999999
    assert expectations["logical_X"] > 0.999999
    assert abs(expectations["logical_Z"]) < 1e-12
