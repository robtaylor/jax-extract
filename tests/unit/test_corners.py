"""Unit tests for PVT corner handling."""


from jax_extract.pdk.adapter import get_ihp_sg13g2_technology
from jax_extract.pdk.corners import (
    CORNER_FACTORS,
    CornerConfig,
    CornerType,
    apply_corner_to_technology,
    get_temperature_resistance_factor,
)


class TestCornerType:
    """Tests for CornerType enum."""

    def test_corner_values(self):
        """Test corner type string values."""
        assert CornerType.TYPICAL.value == "typical"
        assert CornerType.CBEST.value == "cbest"
        assert CornerType.CWORST.value == "cworst"
        assert CornerType.RCBEST.value == "rcbest"
        assert CornerType.RCWORST.value == "rcworst"

    def test_corner_from_string(self):
        """Test creating corner type from string."""
        assert CornerType("typical") == CornerType.TYPICAL
        assert CornerType("rcworst") == CornerType.RCWORST


class TestCornerFactors:
    """Tests for CornerFactors."""

    def test_typical_factors(self):
        """Test typical corner has unity factors."""
        factors = CORNER_FACTORS[CornerType.TYPICAL]
        assert factors.capacitance == 1.0
        assert factors.resistance == 1.0

    def test_cbest_factors(self):
        """Test cbest has lower capacitance."""
        factors = CORNER_FACTORS[CornerType.CBEST]
        assert factors.capacitance < 1.0
        assert factors.resistance == 1.0

    def test_cworst_factors(self):
        """Test cworst has higher capacitance."""
        factors = CORNER_FACTORS[CornerType.CWORST]
        assert factors.capacitance > 1.0
        assert factors.resistance == 1.0

    def test_rcbest_factors(self):
        """Test rcbest has lower R and C."""
        factors = CORNER_FACTORS[CornerType.RCBEST]
        assert factors.capacitance < 1.0
        assert factors.resistance < 1.0

    def test_rcworst_factors(self):
        """Test rcworst has higher R and C."""
        factors = CORNER_FACTORS[CornerType.RCWORST]
        assert factors.capacitance > 1.0
        assert factors.resistance > 1.0


class TestCornerConfig:
    """Tests for CornerConfig."""

    def test_create_from_type(self):
        """Test creating config from corner type."""
        config = CornerConfig.from_corner_type(CornerType.RCWORST)

        assert config.corner_type == CornerType.RCWORST
        assert config.capacitance_factor > 1.0
        assert config.resistance_factor > 1.0

    def test_create_from_string(self):
        """Test creating config from string."""
        config = CornerConfig.from_corner_type("cbest")

        assert config.corner_type == CornerType.CBEST
        assert config.capacitance_factor < 1.0

    def test_temperature_default(self):
        """Test default temperature."""
        config = CornerConfig.from_corner_type(CornerType.TYPICAL)
        assert config.temperature_c == 25.0

    def test_temperature_custom(self):
        """Test custom temperature."""
        config = CornerConfig.from_corner_type(
            CornerType.TYPICAL,
            temperature_c=85.0,
        )
        assert config.temperature_c == 85.0


class TestApplyCorner:
    """Tests for applying corner to technology."""

    def test_apply_typical(self):
        """Test applying typical corner (no change)."""
        tech = get_ihp_sg13g2_technology()
        corner = CornerConfig.from_corner_type(CornerType.TYPICAL)

        scaled = apply_corner_to_technology(tech, corner)

        # Typical should not change values
        m1_orig = tech.get_layer("Metal1")
        m1_scaled = scaled.get_layer("Metal1")

        assert m1_orig is not None
        assert m1_scaled is not None
        assert m1_scaled.sheet_resistance_ohm_sq == m1_orig.sheet_resistance_ohm_sq

    def test_apply_rcworst(self):
        """Test applying rcworst corner."""
        tech = get_ihp_sg13g2_technology()
        corner = CornerConfig.from_corner_type(CornerType.RCWORST)

        scaled = apply_corner_to_technology(tech, corner)

        m1_orig = tech.get_layer("Metal1")
        m1_scaled = scaled.get_layer("Metal1")

        assert m1_orig is not None
        assert m1_scaled is not None

        # Resistance should be higher
        assert m1_scaled.sheet_resistance_ohm_sq > m1_orig.sheet_resistance_ohm_sq

        # Check via resistance too
        via1_orig = tech.get_via("Via1")
        via1_scaled = scaled.get_via("Via1")

        assert via1_orig is not None
        assert via1_scaled is not None
        assert via1_scaled.resistance_ohm > via1_orig.resistance_ohm

    def test_apply_cbest(self):
        """Test applying cbest corner."""
        tech = get_ihp_sg13g2_technology()
        corner = CornerConfig.from_corner_type(CornerType.CBEST)

        scaled = apply_corner_to_technology(tech, corner)

        # Capacitance factor affects permittivity
        ild_orig = tech.dielectrics[0]
        ild_scaled = scaled.dielectrics[0]

        assert ild_scaled.permittivity < ild_orig.permittivity

    def test_original_unchanged(self):
        """Test that original technology is not modified."""
        tech = get_ihp_sg13g2_technology()
        m1_orig = tech.get_layer("Metal1")
        orig_rs = m1_orig.sheet_resistance_ohm_sq if m1_orig else 0

        corner = CornerConfig.from_corner_type(CornerType.RCWORST)
        _ = apply_corner_to_technology(tech, corner)

        # Original should be unchanged
        m1_check = tech.get_layer("Metal1")
        assert m1_check is not None
        assert m1_check.sheet_resistance_ohm_sq == orig_rs

    def test_name_includes_corner(self):
        """Test that scaled tech name includes corner."""
        tech = get_ihp_sg13g2_technology()
        corner = CornerConfig.from_corner_type(CornerType.RCBEST)

        scaled = apply_corner_to_technology(tech, corner)

        assert "rcbest" in scaled.name.lower()


class TestTemperatureScaling:
    """Tests for temperature-dependent resistance."""

    def test_room_temperature(self):
        """Test factor at room temperature is 1.0."""
        factor = get_temperature_resistance_factor(25.0)
        assert abs(factor - 1.0) < 1e-10

    def test_higher_temperature(self):
        """Test higher temperature increases resistance."""
        factor = get_temperature_resistance_factor(85.0)
        assert factor > 1.0

        # 60C increase with TCO=0.003: factor = 1 + 0.003*60 = 1.18
        expected = 1.0 + 0.003 * 60
        assert abs(factor - expected) < 1e-10

    def test_lower_temperature(self):
        """Test lower temperature decreases resistance."""
        factor = get_temperature_resistance_factor(-40.0)
        assert factor < 1.0

    def test_custom_tco(self):
        """Test custom temperature coefficient."""
        # Copper has higher TCO (~0.004)
        factor = get_temperature_resistance_factor(85.0, tco=0.004)

        expected = 1.0 + 0.004 * 60
        assert abs(factor - expected) < 1e-10
