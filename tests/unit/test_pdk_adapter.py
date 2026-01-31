"""Unit tests for PDK adapter module."""


from jax_extract.pdk.adapter import (
    LayerStackEntry,
    TechnologyData,
    ViaDefinition,
    get_ihp_sg13g2_technology,
)


class TestLayerStackEntry:
    """Tests for LayerStackEntry dataclass."""

    def test_create_layer(self):
        """Test creating a layer entry."""
        layer = LayerStackEntry(
            name="Metal1",
            gds_layer=8,
            gds_datatype=0,
            thickness_nm=360,
            height_nm=450,
            sheet_resistance_ohm_sq=0.075,
        )

        assert layer.name == "Metal1"
        assert layer.gds_layer == 8
        assert layer.thickness_nm == 360
        assert layer.sheet_resistance_ohm_sq == 0.075

    def test_default_values(self):
        """Test default values are applied."""
        layer = LayerStackEntry(name="Test", gds_layer=1)

        assert layer.gds_datatype == 0
        assert layer.thickness_nm == 0.0
        assert layer.height_nm == 0.0
        assert layer.is_routing is True


class TestViaDefinition:
    """Tests for ViaDefinition dataclass."""

    def test_create_via(self):
        """Test creating a via definition."""
        via = ViaDefinition(
            name="Via1",
            gds_layer=19,
            bottom_layer="Metal1",
            top_layer="Metal2",
            resistance_ohm=2.0,
        )

        assert via.name == "Via1"
        assert via.bottom_layer == "Metal1"
        assert via.top_layer == "Metal2"
        assert via.resistance_ohm == 2.0


class TestTechnologyData:
    """Tests for TechnologyData dataclass."""

    def test_get_layer(self):
        """Test getting layer by name."""
        tech = TechnologyData(name="test")
        tech.layers = [
            LayerStackEntry(name="Metal1", gds_layer=8),
            LayerStackEntry(name="Metal2", gds_layer=10),
        ]

        layer = tech.get_layer("Metal1")
        assert layer is not None
        assert layer.name == "Metal1"

        assert tech.get_layer("NonExistent") is None

    def test_get_via(self):
        """Test getting via by name."""
        tech = TechnologyData(name="test")
        tech.vias = [
            ViaDefinition(name="Via1", gds_layer=19),
            ViaDefinition(name="Via2", gds_layer=29),
        ]

        via = tech.get_via("Via1")
        assert via is not None
        assert via.name == "Via1"

        assert tech.get_via("NonExistent") is None

    def test_get_layer_by_gds(self):
        """Test getting layer by GDS layer number."""
        tech = TechnologyData(name="test")
        tech.layers = [
            LayerStackEntry(name="Metal1", gds_layer=8, gds_datatype=0),
            LayerStackEntry(name="Metal1_drawing", gds_layer=8, gds_datatype=1),
        ]

        layer = tech.get_layer_by_gds(8, 0)
        assert layer is not None
        assert layer.name == "Metal1"

        layer = tech.get_layer_by_gds(8, 1)
        assert layer is not None
        assert layer.name == "Metal1_drawing"

        assert tech.get_layer_by_gds(999) is None


class TestIHPSG13G2Technology:
    """Tests for IHP SG13G2 technology data."""

    def test_get_technology(self):
        """Test getting IHP SG13G2 technology."""
        tech = get_ihp_sg13g2_technology()

        assert tech.name == "IHP-SG13G2"
        assert len(tech.layers) > 0
        assert len(tech.vias) > 0

    def test_metal_layers(self):
        """Test metal layer definitions."""
        tech = get_ihp_sg13g2_technology()

        # Check Metal1
        m1 = tech.get_layer("Metal1")
        assert m1 is not None
        assert m1.thickness_nm == 360
        assert m1.height_nm == 450
        assert m1.sheet_resistance_ohm_sq == 0.075

        # Check TopMetal2
        tm2 = tech.get_layer("TopMetal2")
        assert tm2 is not None
        assert tm2.thickness_nm == 2800
        assert tm2.sheet_resistance_ohm_sq == 0.010

    def test_via_resistance(self):
        """Test via resistance values."""
        tech = get_ihp_sg13g2_technology()

        via1 = tech.get_via("Via1")
        assert via1 is not None
        assert via1.resistance_ohm == 2.0

        topvia2 = tech.get_via("TopVia2")
        assert topvia2 is not None
        assert topvia2.resistance_ohm == 0.5

    def test_dielectric_layers(self):
        """Test dielectric layer definitions."""
        tech = get_ihp_sg13g2_technology()

        assert len(tech.dielectrics) > 0
        # First dielectric should be between substrate and Metal1
        assert tech.dielectrics[0].permittivity == 3.9  # SiO2
