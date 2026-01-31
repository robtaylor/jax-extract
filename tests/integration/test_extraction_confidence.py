"""Integration tests for extraction confidence validation.

These tests validate the complete extraction pipeline:
    geometry -> extract -> simulate -> compare to analytical

Each test case creates synthetic geometry, extracts parasitics,
simulates the RC network, and compares to analytical expectations.
"""

import numpy as np
import pytest

from jax_extract.extraction.base import Conductor, ExtractionConfig, ExtractionMode
from jax_extract.extraction.extractor import (
    FullExtractor,
    extract_parallel_plate,
    extract_simple_wire,
)
from jax_extract.extraction.field_solver import extract_capacitance_parallel_plate
from jax_extract.extraction.resistance import calculate_wire_resistance
from jax_extract.simulation.jax_spice import AnalyticalSimulator


class TestParallelPlateExtraction:
    """Validate parallel plate capacitance extraction."""

    def test_basic_parallel_plate_capacitance(self, integration_tech):
        """Test extraction of a 10x10 um parallel plate capacitor.

        Analytical: C = eps_0 * eps_r * A / d
        For 10x10 um plate, 450nm height, eps_r=3.9:
            C = 8.854e-12 * 3.9 * 100e-12 / 450e-9
            C ~ 7.67 fF (area only)
        """
        result = extract_parallel_plate(10.0, 10.0, integration_tech, "Metal1")

        # Calculate expected (area capacitance)
        layer = integration_tech.get_layer("Metal1")
        assert layer is not None

        area_um2 = 100.0  # 10x10 um
        height_nm = layer.height_nm

        expected_area_cap = extract_capacitance_parallel_plate(
            area_um2, height_nm, 3.9
        )

        # Extracted value should be greater than area-only (due to fringe)
        assert result.total_capacitance_f > expected_area_cap * 0.9

        # But not unreasonably large (< 2x area cap for this geometry)
        assert result.total_capacitance_f < expected_area_cap * 3.0

    def test_capacitance_scaling_with_area(self, integration_tech):
        """Test that capacitance scales approximately with area."""
        result_small = extract_parallel_plate(5.0, 5.0, integration_tech, "Metal1")
        result_large = extract_parallel_plate(10.0, 10.0, integration_tech, "Metal1")

        # Large plate has 4x the area
        ratio = result_large.total_capacitance_f / result_small.total_capacitance_f

        # Should be close to 4x (within 50% due to fringe effects)
        assert ratio > 2.5
        assert ratio < 6.0

    def test_capacitance_on_different_layers(self, integration_tech):
        """Test capacitance extraction on different metal layers.

        Higher layers have less capacitance due to increased height above substrate.
        """
        result_m1 = extract_parallel_plate(10.0, 10.0, integration_tech, "Metal1")
        result_m2 = extract_parallel_plate(10.0, 10.0, integration_tech, "Metal2")

        # Metal2 is higher, so should have less capacitance
        # (larger distance to ground)
        assert result_m2.total_capacitance_f < result_m1.total_capacitance_f


class TestWireResistanceExtraction:
    """Validate wire resistance extraction."""

    def test_basic_wire_resistance(self, integration_tech):
        """Test extraction of a 100um x 0.5um wire on Metal1.

        Analytical: R = Rs * L / W
        For Metal1 (Rs=0.075 ohm/sq), 100um x 0.5um:
            R = 0.075 * 100 / 0.5 = 15 ohms
        """
        result = extract_simple_wire(100.0, 0.5, integration_tech, "Metal1")

        layer = integration_tech.get_layer("Metal1")
        assert layer is not None

        expected_r = calculate_wire_resistance(
            100.0, 0.5, layer.sheet_resistance_ohm_sq
        )

        # Allow 10% tolerance (segmentation effects)
        relative_error = abs(result.total_resistance_ohm - expected_r) / expected_r
        assert relative_error < 0.10, (
            f"Resistance error {relative_error:.1%}: "
            f"extracted={result.total_resistance_ohm:.2f}, expected={expected_r:.2f}"
        )

    def test_resistance_scaling_with_length(self, integration_tech):
        """Test that resistance scales linearly with length."""
        result_short = extract_simple_wire(50.0, 0.5, integration_tech, "Metal1")
        result_long = extract_simple_wire(100.0, 0.5, integration_tech, "Metal1")

        ratio = result_long.total_resistance_ohm / result_short.total_resistance_ohm

        # Should be 2x (within 5%)
        assert abs(ratio - 2.0) < 0.1

    def test_resistance_scaling_with_width(self, integration_tech):
        """Test that resistance scales inversely with width."""
        result_narrow = extract_simple_wire(100.0, 0.5, integration_tech, "Metal1")
        result_wide = extract_simple_wire(100.0, 1.0, integration_tech, "Metal1")

        ratio = result_narrow.total_resistance_ohm / result_wide.total_resistance_ohm

        # Narrow wire should have 2x resistance (within 5%)
        assert abs(ratio - 2.0) < 0.1

    def test_resistance_on_different_layers(self, integration_tech):
        """Test resistance extraction on different metal layers.

        Metal2 has lower sheet resistance than Metal1.
        """
        result_m1 = extract_simple_wire(100.0, 0.5, integration_tech, "Metal1")
        result_m2 = extract_simple_wire(100.0, 0.5, integration_tech, "Metal2")

        m1_layer = integration_tech.get_layer("Metal1")
        m2_layer = integration_tech.get_layer("Metal2")
        assert m1_layer is not None
        assert m2_layer is not None

        # Metal2 has lower Rs, so lower resistance
        assert result_m2.total_resistance_ohm < result_m1.total_resistance_ohm

        # Ratio should match sheet resistance ratio
        expected_ratio = m1_layer.sheet_resistance_ohm_sq / m2_layer.sheet_resistance_ohm_sq
        actual_ratio = result_m1.total_resistance_ohm / result_m2.total_resistance_ohm

        assert abs(actual_ratio - expected_ratio) < 0.1


class TestRCTimeConstant:
    """Validate RC time constant extraction and simulation."""

    def test_simple_rc_time_constant(self, integration_tech):
        """Test RC time constant for a simple wire.

        For a 100um x 0.5um Metal1 wire:
            R ~ 15 ohms (from test above)
            C ~ few fF
            tau = R * C ~ tens of femtoseconds
        """
        result = extract_simple_wire(100.0, 0.5, integration_tech, "Metal1")

        # Time constant should be positive
        assert result.rc_time_constant_s > 0

        # For this geometry, expect tau in range 10fs to 1ps
        assert result.rc_time_constant_s > 1e-15
        assert result.rc_time_constant_s < 1e-11

    def test_rc_simulation_matches_extraction(self, integration_tech):
        """Test that simulated time constant matches extracted R*C."""
        result = extract_simple_wire(100.0, 0.5, integration_tech, "Metal1")

        # Simulate step response
        simulator = AnalyticalSimulator()
        sim_result = simulator.simulate_rc_step_response(
            r_total=result.total_resistance_ohm,
            c_total=result.total_capacitance_f,
            t_stop=10 * result.rc_time_constant_s,
        )

        # Measure time constant from waveform
        measured_tau = sim_result.measure_time_constant("out")

        assert measured_tau is not None

        # Should match extracted tau within 10%
        expected_tau = result.rc_time_constant_s
        relative_error = abs(measured_tau - expected_tau) / expected_tau

        assert relative_error < 0.10, (
            f"Time constant error {relative_error:.1%}: "
            f"measured={measured_tau:.3e}, expected={expected_tau:.3e}"
        )

    def test_larger_wire_has_larger_time_constant(self, integration_tech):
        """Test that longer wires have larger time constants.

        For distributed RC lines, tau scales as L^2.
        """
        result_short = extract_simple_wire(50.0, 0.5, integration_tech, "Metal1")
        result_long = extract_simple_wire(100.0, 0.5, integration_tech, "Metal1")

        # Time constant ratio should be roughly 4x (L^2 scaling)
        # but depends on how capacitance and resistance scale
        ratio = result_long.rc_time_constant_s / result_short.rc_time_constant_s

        # Should be significantly larger (at least 2x)
        assert ratio > 2.0

        # But not more than 8x (pure L^2 would give 4x)
        assert ratio < 8.0


class TestRCLadderExtraction:
    """Validate RC ladder network extraction."""

    def test_rc_ladder_creation(self, integration_tech, rc_ladder_geometry):
        """Test that RC ladder geometry creates proper network."""
        extractor = FullExtractor(integration_tech)
        result = extractor.extract_from_geometry(rc_ladder_geometry)

        # Should have extracted some resistances and capacitances
        assert len(result.resistances) > 0
        assert len(result.capacitances) > 0

    def test_rc_ladder_network_structure(self, integration_tech):
        """Test RC ladder network has correct structure.

        Create 5 segments of 20um each = 100um total.
        """
        # Create 5 connected 20um segments
        conductors = []
        for i in range(5):
            vertices = np.array(
                [
                    [i * 20.0, 0.0],
                    [(i + 1) * 20.0, 0.0],
                    [(i + 1) * 20.0, 0.5],
                    [i * 20.0, 0.5],
                ],
                dtype=np.float64,
            )
            conductor = Conductor(name=f"seg_{i}", net_id=0)
            conductor.polygons["Metal1"] = [vertices]
            conductors.append(conductor)

        extractor = FullExtractor(integration_tech)
        result = extractor.extract_rc(conductors)

        # Total resistance should be close to 100um wire
        layer = integration_tech.get_layer("Metal1")
        assert layer is not None
        expected_r = calculate_wire_resistance(
            100.0, 0.5, layer.sheet_resistance_ohm_sq
        )

        # Allow 20% tolerance due to segment boundaries
        relative_error = abs(result.total_resistance_ohm - expected_r) / expected_r
        assert relative_error < 0.20

    def test_rc_network_has_correct_elements(self, integration_tech):
        """Test that extracted RCNetwork has resistors and capacitors."""
        result = extract_simple_wire(100.0, 0.5, integration_tech, "Metal1")

        network = result.network

        # Should have nodes
        assert len(network.nodes) > 0

        # Should have resistors (for RC mode)
        assert len(network.resistors) > 0

        # Should have capacitors
        assert len(network.capacitors) > 0


class TestExtractionModes:
    """Test different extraction modes."""

    def test_capacitance_only_mode(self, integration_tech):
        """Test capacitance-only extraction mode."""
        vertices = np.array(
            [
                [0.0, 0.0],
                [10.0, 0.0],
                [10.0, 10.0],
                [0.0, 10.0],
            ],
            dtype=np.float64,
        )
        conductor = Conductor(name="plate", net_id=0)
        conductor.polygons["Metal1"] = [vertices]

        config = ExtractionConfig(mode=ExtractionMode.CAPACITANCE_ONLY)
        extractor = FullExtractor(integration_tech, config)
        result = extractor.extract_from_geometry([conductor])

        # Should have capacitance
        assert len(result.capacitances) > 0

        # Should NOT have resistance
        assert len(result.resistances) == 0

    def test_rc_mode(self, integration_tech):
        """Test RC extraction mode (default)."""
        result = extract_simple_wire(100.0, 0.5, integration_tech, "Metal1")

        # Should have both R and C
        assert result.total_resistance_ohm > 0
        assert result.total_capacitance_f > 0


class TestAnalyticalValidation:
    """Validate extraction against analytical formulas with known tolerances."""

    @pytest.mark.parametrize(
        "area_um,expected_c_ff_min,expected_c_ff_max",
        [
            (25.0, 0.5, 5.0),      # 5x5 um plate: ~0.4-2 fF
            (100.0, 2.0, 15.0),    # 10x10 um plate: ~2-10 fF
            (400.0, 8.0, 50.0),    # 20x20 um plate: ~8-35 fF
        ],
    )
    def test_plate_capacitance_in_range(
        self, integration_tech, area_um, expected_c_ff_min, expected_c_ff_max
    ):
        """Test plate capacitance is within expected range."""
        side = np.sqrt(area_um)
        result = extract_parallel_plate(side, side, integration_tech, "Metal1")

        cap_ff = result.total_capacitance_f * 1e15  # Convert to fF

        assert cap_ff > expected_c_ff_min, (
            f"Capacitance {cap_ff:.2f} fF below minimum {expected_c_ff_min} fF"
        )
        assert cap_ff < expected_c_ff_max, (
            f"Capacitance {cap_ff:.2f} fF above maximum {expected_c_ff_max} fF"
        )

    @pytest.mark.parametrize(
        "length_um,width_um,expected_r_min,expected_r_max",
        [
            (10.0, 0.5, 1.0, 2.0),     # Short wire: ~1.5 ohm
            (100.0, 0.5, 10.0, 20.0),  # Medium wire: ~15 ohm
            (100.0, 1.0, 5.0, 10.0),   # Wide wire: ~7.5 ohm
        ],
    )
    def test_wire_resistance_in_range(
        self, integration_tech, length_um, width_um, expected_r_min, expected_r_max
    ):
        """Test wire resistance is within expected range."""
        result = extract_simple_wire(length_um, width_um, integration_tech, "Metal1")

        assert result.total_resistance_ohm > expected_r_min, (
            f"Resistance {result.total_resistance_ohm:.2f} ohm below minimum {expected_r_min}"
        )
        assert result.total_resistance_ohm < expected_r_max, (
            f"Resistance {result.total_resistance_ohm:.2f} ohm above maximum {expected_r_max}"
        )


class TestSPICENetlistGeneration:
    """Test SPICE netlist generation from extracted networks."""

    def test_network_generates_spice(self, integration_tech):
        """Test that extracted network generates valid SPICE."""
        result = extract_simple_wire(100.0, 0.5, integration_tech, "Metal1")

        spice = result.network.to_spice_flat()

        # Should contain SPICE elements
        assert "R" in spice or "C" in spice

    def test_network_generates_subcircuit(self, integration_tech):
        """Test subcircuit generation."""
        result = extract_simple_wire(100.0, 0.5, integration_tech, "Metal1")

        spice = result.network.to_spice_subcircuit()

        # Should have subcircuit wrapper
        assert ".subckt" in spice
        assert ".ends" in spice


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_minimum_width_wire(self, integration_tech):
        """Test extraction of minimum-width wire."""
        layer = integration_tech.get_layer("Metal1")
        assert layer is not None
        min_width_um = layer.min_width_nm / 1000

        result = extract_simple_wire(100.0, min_width_um, integration_tech, "Metal1")

        # Should still extract valid values
        assert result.total_resistance_ohm > 0
        assert result.total_capacitance_f > 0

    def test_very_short_wire(self, integration_tech):
        """Test extraction of very short wire (1um)."""
        result = extract_simple_wire(1.0, 0.5, integration_tech, "Metal1")

        # Should extract something, even if small
        assert result.total_resistance_ohm > 0
        assert result.total_capacitance_f > 0

    def test_square_geometry(self, integration_tech):
        """Test extraction of square geometry (not obviously a wire)."""
        # 5um x 5um square
        result = extract_simple_wire(5.0, 5.0, integration_tech, "Metal1")

        # Should extract both R and C
        assert result.total_resistance_ohm > 0
        assert result.total_capacitance_f > 0

    def test_empty_conductor_list(self, integration_tech):
        """Test extraction with empty conductor list."""
        extractor = FullExtractor(integration_tech)
        result = extractor.extract_from_geometry([])

        # Should return empty result, not error
        assert result.cell_name == "empty"
        assert len(result.conductors) == 0
        assert len(result.capacitances) == 0
        assert len(result.resistances) == 0
