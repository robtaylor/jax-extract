"""Analytical validation tests for resistance extraction.

These tests validate resistance calculations against known analytical solutions.
"""


from jax_extract.extraction.resistance import (
    ResistanceNetwork,
    ViaInstance,
    WireSegment,
    calculate_serpentine_resistance,
    calculate_wire_resistance,
    elmore_delay,
)
from jax_extract.pdk.adapter import get_ihp_sg13g2_technology


class TestWireResistance:
    """Validate wire resistance calculation."""

    def test_basic_wire_resistance(self):
        """Test basic R = Rs * L / W formula."""
        # 10um long, 1um wide, 1 ohm/sq
        r = calculate_wire_resistance(10.0, 1.0, 1.0)
        assert abs(r - 10.0) < 1e-10

    def test_scaling_with_length(self):
        """Test resistance scales linearly with length."""
        r1 = calculate_wire_resistance(10.0, 1.0, 1.0)
        r2 = calculate_wire_resistance(20.0, 1.0, 1.0)
        assert abs(r2 / r1 - 2.0) < 1e-10

    def test_scaling_with_width(self):
        """Test resistance scales inversely with width."""
        r1 = calculate_wire_resistance(10.0, 1.0, 1.0)
        r2 = calculate_wire_resistance(10.0, 2.0, 1.0)
        assert abs(r1 / r2 - 2.0) < 1e-10

    def test_scaling_with_sheet_r(self):
        """Test resistance scales with sheet resistance."""
        r1 = calculate_wire_resistance(10.0, 1.0, 1.0)
        r2 = calculate_wire_resistance(10.0, 1.0, 2.0)
        assert abs(r2 / r1 - 2.0) < 1e-10

    def test_zero_width(self):
        """Test zero width returns infinity."""
        r = calculate_wire_resistance(10.0, 0.0, 1.0)
        assert r == float("inf")

    def test_typical_metal1_wire(self):
        """Test typical Metal1 wire on IHP SG13G2."""
        # Metal1: Rs = 0.075 ohm/sq
        # 100um long, 0.5um wide
        r = calculate_wire_resistance(100.0, 0.5, 0.075)
        # Expected: 0.075 * 100.0 / 0.5 = 15 ohms
        assert abs(r - 15.0) < 1e-10


class TestWireSegment:
    """Tests for WireSegment class."""

    def test_segment_length(self):
        """Test segment length calculation."""
        seg = WireSegment(
            layer="Metal1",
            start=(0.0, 0.0),
            end=(10.0, 0.0),
            width=1.0,
        )
        assert abs(seg.length() - 10.0) < 1e-10

    def test_segment_diagonal_length(self):
        """Test diagonal segment length."""
        seg = WireSegment(
            layer="Metal1",
            start=(0.0, 0.0),
            end=(3.0, 4.0),
            width=1.0,
        )
        assert abs(seg.length() - 5.0) < 1e-10

    def test_segment_resistance(self):
        """Test segment resistance calculation."""
        seg = WireSegment(
            layer="Metal1",
            start=(0.0, 0.0),
            end=(10.0, 0.0),
            width=0.5,
        )
        r = seg.resistance(0.075)
        expected = 0.075 * 10.0 / 0.5
        assert abs(r - expected) < 1e-10

    def test_segment_midpoint(self):
        """Test segment midpoint calculation."""
        seg = WireSegment(
            layer="Metal1",
            start=(0.0, 0.0),
            end=(10.0, 20.0),
            width=1.0,
        )
        mx, my = seg.midpoint()
        assert abs(mx - 5.0) < 1e-10
        assert abs(my - 10.0) < 1e-10


class TestViaInstance:
    """Tests for ViaInstance class."""

    def test_single_via_resistance(self):
        """Test single via resistance."""
        via = ViaInstance(
            via_type="Via1",
            x=10.0,
            y=20.0,
            bottom_layer="Metal1",
            top_layer="Metal2",
            count=1,
        )
        r = via.resistance(2.0)
        assert abs(r - 2.0) < 1e-10

    def test_via_array_resistance(self):
        """Test via array (parallel) resistance."""
        via = ViaInstance(
            via_type="Via1",
            x=10.0,
            y=20.0,
            bottom_layer="Metal1",
            top_layer="Metal2",
            count=4,
        )
        # 4 parallel vias: R = R_single / 4
        r = via.resistance(2.0)
        assert abs(r - 0.5) < 1e-10

    def test_zero_count(self):
        """Test zero via count returns infinity."""
        via = ViaInstance(
            via_type="Via1",
            x=0.0,
            y=0.0,
            bottom_layer="Metal1",
            top_layer="Metal2",
            count=0,
        )
        r = via.resistance(2.0)
        assert r == float("inf")


class TestSerpentineResistance:
    """Tests for serpentine test structure resistance."""

    def test_basic_serpentine(self):
        """Test basic serpentine resistance."""
        # 10 turns, each 100um long, 0.5um wide, 0.5um spacing
        # Rs = 0.075 ohm/sq
        r = calculate_serpentine_resistance(
            width_um=0.5,
            spacing_um=0.5,
            turns=10,
            turn_length_um=100.0,
            sheet_resistance=0.075,
        )

        # Total length = (10+1) * 100 + 10 * (0.5 + 0.5) = 1100 + 10 = 1110 um
        # R = 0.075 * 1110 / 0.5 = 166.5 ohms
        expected = 0.075 * 1110.0 / 0.5
        assert abs(r - expected) < 1e-6

    def test_serpentine_scales_with_turns(self):
        """Test resistance scales with number of turns."""
        r1 = calculate_serpentine_resistance(0.5, 0.5, 5, 100.0, 0.075)
        r2 = calculate_serpentine_resistance(0.5, 0.5, 10, 100.0, 0.075)

        # Approximately doubles (not exact due to connector segments)
        ratio = r2 / r1
        assert ratio > 1.8
        assert ratio < 2.2


class TestElmoreDelay:
    """Tests for Elmore delay calculation."""

    def test_simple_rc(self):
        """Test simple RC delay."""
        # Single R with downstream C
        delay = elmore_delay([(100.0, 1e-15)], [1e-15])
        expected = 100.0 * 1e-15  # = 100 fs = 0.1 ps
        assert abs(delay - expected) < 1e-20

    def test_rc_ladder(self):
        """Test RC ladder delay."""
        # Two-stage ladder: R1=100, C1=1fF, R2=100, C2=1fF
        # Elmore delay = R1*(C1+C2) + R2*C2 = 100*2fF + 100*1fF = 300 fs
        resistors = [
            (100.0, 2e-15),  # R1 drives C1+C2
            (100.0, 1e-15),  # R2 drives C2
        ]
        delay = elmore_delay(resistors, [])
        expected = 100.0 * 2e-15 + 100.0 * 1e-15
        assert abs(delay - expected) < 1e-20

    def test_distributed_rc_line(self):
        """Test distributed RC line (approximation)."""
        # N segments, each R/N with C/N at the end
        # Elmore delay for distributed RC ≈ 0.5 * R * C
        n = 100
        r_total = 100.0
        c_total = 10e-15
        r_seg = r_total / n
        c_seg = c_total / n

        # Build resistor list with downstream capacitance
        resistors = []
        for i in range(n):
            # Downstream capacitance = (n - i) * c_seg
            c_downstream = (n - i) * c_seg
            resistors.append((r_seg, c_downstream))

        delay = elmore_delay(resistors, [])

        # Theoretical: 0.5 * R * C for distributed line
        # But Elmore sum gives: sum(r_seg * (n-i) * c_seg) = r_seg * c_seg * sum(n-i)
        # = r_seg * c_seg * n*(n+1)/2 ≈ 0.5 * R * C for large n
        theoretical = 0.5 * r_total * c_total

        # Should be close for large n
        assert abs(delay - theoretical) / theoretical < 0.02  # Within 2%


class TestResistanceNetwork:
    """Tests for ResistanceNetwork class."""

    def test_total_wire_resistance(self):
        """Test total wire resistance calculation."""
        tech = get_ihp_sg13g2_technology()

        network = ResistanceNetwork(net_name="test")
        network.segments = [
            WireSegment("Metal1", (0, 0), (10, 0), 0.5),
            WireSegment("Metal1", (10, 0), (20, 0), 0.5),
        ]

        r = network.total_wire_resistance(tech)

        # Each segment: 0.075 * 10 / 0.5 = 1.5 ohms
        expected = 2 * 0.075 * 10.0 / 0.5
        assert abs(r - expected) < 1e-10

    def test_total_via_resistance(self):
        """Test total via resistance calculation."""
        tech = get_ihp_sg13g2_technology()

        network = ResistanceNetwork(net_name="test")
        network.vias = [
            ViaInstance("Via1", 10.0, 0.0, "Metal1", "Metal2", 2),
            ViaInstance("Via1", 20.0, 0.0, "Metal1", "Metal2", 1),
        ]

        r = network.total_via_resistance(tech)

        # Via1 resistance = 2.0 ohms per via
        # 2 parallel vias = 1.0 ohm, 1 via = 2.0 ohm
        expected = 1.0 + 2.0
        assert abs(r - expected) < 1e-10
