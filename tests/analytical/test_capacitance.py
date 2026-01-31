"""Analytical validation tests for capacitance extraction.

These tests validate the field solver against known analytical solutions.
"""


from jax_extract.extraction.field_solver import (
    EPSILON_0,
    extract_capacitance_parallel_plate,
    extract_capacitance_wire,
)


class TestParallelPlateCapacitance:
    """Validate parallel plate capacitor formula."""

    def test_basic_parallel_plate(self):
        """Test basic parallel plate capacitor."""
        # 10um x 10um plate, 100nm dielectric, SiO2 (eps=3.9)
        area_um2 = 100.0  # 10um x 10um
        thickness_nm = 100.0
        permittivity = 3.9

        c = extract_capacitance_parallel_plate(area_um2, thickness_nm, permittivity)

        # Manual calculation:
        # C = eps_0 * eps_r * A / d
        # A = 100 um^2 = 100e-12 m^2
        # d = 100 nm = 100e-9 m
        # C = 8.854e-12 * 3.9 * 100e-12 / 100e-9
        # C = 3.45e-14 F = 34.5 fF
        expected = EPSILON_0 * permittivity * 100e-12 / 100e-9

        assert abs(c - expected) < 1e-20
        assert abs(c - 3.45e-14) < 0.1e-14  # Within 0.1fF

    def test_scaling_with_area(self):
        """Test capacitance scales linearly with area."""
        thickness_nm = 100.0

        c1 = extract_capacitance_parallel_plate(100.0, thickness_nm)
        c2 = extract_capacitance_parallel_plate(200.0, thickness_nm)

        assert abs(c2 / c1 - 2.0) < 1e-10

    def test_scaling_with_thickness(self):
        """Test capacitance scales inversely with thickness."""
        area_um2 = 100.0

        c1 = extract_capacitance_parallel_plate(area_um2, 100.0)
        c2 = extract_capacitance_parallel_plate(area_um2, 200.0)

        assert abs(c1 / c2 - 2.0) < 1e-10

    def test_scaling_with_permittivity(self):
        """Test capacitance scales linearly with permittivity."""
        area_um2 = 100.0
        thickness_nm = 100.0

        c1 = extract_capacitance_parallel_plate(area_um2, thickness_nm, 3.9)
        c2 = extract_capacitance_parallel_plate(area_um2, thickness_nm, 7.8)

        assert abs(c2 / c1 - 2.0) < 1e-10

    def test_typical_mim_cap(self):
        """Test typical MIM capacitor values."""
        # 50um x 50um MIM cap with 50nm dielectric (Si3N4, eps=7.5)
        area_um2 = 2500.0
        thickness_nm = 50.0
        permittivity = 7.5

        c = extract_capacitance_parallel_plate(area_um2, thickness_nm, permittivity)

        # Expected: ~3.3 pF
        assert c > 3e-12
        assert c < 4e-12


class TestWireCapacitance:
    """Validate wire capacitance estimation."""

    def test_wire_capacitance_area_component(self):
        """Test that wire capacitance includes area component."""
        # Long narrow wire should have measurable capacitance
        length_um = 100.0
        width_um = 0.5
        height_nm = 450.0
        thickness_nm = 360.0

        c = extract_capacitance_wire(length_um, width_um, height_nm, thickness_nm)

        # Capacitance should be positive
        assert c > 0

        # Should be in reasonable range for this geometry (few fF)
        assert c > 0.1e-15
        assert c < 100e-15

    def test_wire_capacitance_scales_with_length(self):
        """Test capacitance scales with wire length."""
        width_um = 0.5
        height_nm = 450.0
        thickness_nm = 360.0

        c1 = extract_capacitance_wire(100.0, width_um, height_nm, thickness_nm)
        c2 = extract_capacitance_wire(200.0, width_um, height_nm, thickness_nm)

        # Should approximately double (not exact due to fringe effects)
        ratio = c2 / c1
        assert ratio > 1.8
        assert ratio < 2.2

    def test_wire_capacitance_scales_with_width(self):
        """Test capacitance scales with wire width."""
        length_um = 100.0
        height_nm = 450.0
        thickness_nm = 360.0

        c1 = extract_capacitance_wire(length_um, 0.5, height_nm, thickness_nm)
        c2 = extract_capacitance_wire(length_um, 1.0, height_nm, thickness_nm)

        # Wider wire has more area capacitance
        assert c2 > c1

    def test_fringe_capacitance_contribution(self):
        """Test fringe capacitance adds to area capacitance."""
        length_um = 100.0
        width_um = 0.5
        height_nm = 450.0
        thickness_nm = 360.0

        c_total = extract_capacitance_wire(length_um, width_um, height_nm, thickness_nm)

        # Calculate area-only capacitance
        area_m2 = length_um * width_um * 1e-12
        height_m = height_nm * 1e-9
        c_area = EPSILON_0 * 3.9 * area_m2 / height_m

        # Total should be greater than area-only
        assert c_total > c_area


class TestPhysicalConstants:
    """Validate physical constants are correct."""

    def test_epsilon_0(self):
        """Test vacuum permittivity value."""
        # EPSILON_0 should be approximately 8.854e-12 F/m
        assert abs(EPSILON_0 - 8.854187817e-12) < 1e-20

    def test_sio2_permittivity(self):
        """Test SiO2 relative permittivity is reasonable."""
        # SiO2 has relative permittivity around 3.9
        # This is used as default in many functions
        c1 = extract_capacitance_parallel_plate(100.0, 100.0, 3.9)
        c2 = extract_capacitance_parallel_plate(100.0, 100.0, 1.0)

        # SiO2 should give 3.9x more capacitance than vacuum
        assert abs(c1 / c2 - 3.9) < 1e-10


class TestCapacitanceUnits:
    """Validate unit conversions in capacitance calculations."""

    def test_output_in_farads(self):
        """Test that output is in Farads."""
        c = extract_capacitance_parallel_plate(100.0, 100.0, 3.9)

        # Result should be on order of 1e-14 F for these dimensions
        assert c > 1e-16
        assert c < 1e-12

    def test_input_dimensions_correct(self):
        """Test input dimension handling (um^2, nm)."""
        # 1um x 1um = 1 um^2, 1um = 1000nm
        area_um2 = 1.0
        thickness_nm = 1000.0

        c = extract_capacitance_parallel_plate(area_um2, thickness_nm, 1.0)

        # Should equal eps_0 * 1um^2 / 1um = eps_0 * 1um
        # = 8.854e-12 * 1e-6 = 8.854e-18 F
        expected = EPSILON_0 * 1e-6
        assert abs(c - expected) < 1e-23
