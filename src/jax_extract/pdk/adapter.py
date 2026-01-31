"""PDKMaster adapter for extraction-specific data.

This module provides an adapter layer to extract parasitic-relevant data
from PDKMaster technology definitions, particularly for the IHP SG13G2 PDK.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class LayerStackEntry:
    """Single layer in the interconnect stack."""

    name: str
    """Layer name (e.g., 'Metal1', 'TopMetal1')."""

    gds_layer: int
    """GDS layer number."""

    gds_datatype: int = 0
    """GDS datatype."""

    thickness_nm: float = 0.0
    """Metal thickness in nanometers."""

    height_nm: float = 0.0
    """Height from substrate in nanometers."""

    sheet_resistance_ohm_sq: float = 0.0
    """Sheet resistance in ohms per square."""

    min_width_nm: float = 0.0
    """Minimum width in nanometers."""

    min_spacing_nm: float = 0.0
    """Minimum spacing in nanometers."""

    is_routing: bool = True
    """Whether this layer is used for routing."""


@dataclass
class ViaDefinition:
    """Via connecting two metal layers."""

    name: str
    """Via name (e.g., 'Via1', 'TopVia1')."""

    gds_layer: int
    """GDS layer number."""

    gds_datatype: int = 0
    """GDS datatype."""

    bottom_layer: str = ""
    """Bottom metal layer name."""

    top_layer: str = ""
    """Top metal layer name."""

    resistance_ohm: float = 0.0
    """Single via resistance in ohms."""

    min_size_nm: float = 0.0
    """Minimum via size in nanometers."""


@dataclass
class DielectricLayer:
    """Interlayer dielectric properties."""

    name: str
    """Dielectric layer name."""

    permittivity: float
    """Relative permittivity (epsilon_r)."""

    thickness_nm: float
    """Thickness in nanometers."""

    height_nm: float
    """Height from substrate in nanometers."""


@dataclass
class TechnologyData:
    """Complete technology data for parasitic extraction."""

    name: str
    """Technology name."""

    layers: list[LayerStackEntry] = field(default_factory=list)
    """Metal layer stack from bottom to top."""

    vias: list[ViaDefinition] = field(default_factory=list)
    """Via definitions."""

    dielectrics: list[DielectricLayer] = field(default_factory=list)
    """Dielectric layer stack."""

    substrate_permittivity: float = 11.7
    """Silicon substrate relative permittivity."""

    def get_layer(self, name: str) -> LayerStackEntry | None:
        """Get layer by name."""
        for layer in self.layers:
            if layer.name == name:
                return layer
        return None

    def get_via(self, name: str) -> ViaDefinition | None:
        """Get via by name."""
        for via in self.vias:
            if via.name == name:
                return via
        return None

    def get_layer_by_gds(self, gds_layer: int, gds_datatype: int = 0) -> LayerStackEntry | None:
        """Get layer by GDS layer/datatype."""
        for layer in self.layers:
            if layer.gds_layer == gds_layer and layer.gds_datatype == gds_datatype:
                return layer
        return None


class ExtractionAdapter:
    """Adapter to extract parasitic-relevant data from PDKMaster technology.

    This class bridges PDKMaster's technology abstraction with the specific
    data requirements for parasitic extraction.

    Example:
        >>> from c4m_pdk_ihpsg13g2 import tech as ihp_tech
        >>> adapter = ExtractionAdapter(ihp_tech)
        >>> tech_data = adapter.get_technology_data()
        >>> for layer in tech_data.layers:
        ...     print(f"{layer.name}: Rs={layer.sheet_resistance_ohm_sq}")
    """

    def __init__(self, tech: Any):
        """Initialize adapter with PDKMaster technology.

        Args:
            tech: PDKMaster Technology object.
        """
        self.tech = tech
        self._tech_data: TechnologyData | None = None

    def get_technology_data(self) -> TechnologyData:
        """Extract complete technology data for parasitic extraction.

        Returns:
            TechnologyData with layers, vias, and dielectric info.
        """
        if self._tech_data is not None:
            return self._tech_data

        self._tech_data = TechnologyData(name=self.tech.name)
        self._extract_layers()
        self._extract_vias()
        self._extract_dielectrics()

        return self._tech_data

    def _extract_layers(self) -> None:
        """Extract metal layer information from PDKMaster primitives."""
        assert self._tech_data is not None

        # Import here to avoid import errors if PDKMaster not installed
        from pdkmaster.technology.primitive import MetalWire

        for prim in self.tech.primitives:
            if isinstance(prim, MetalWire):
                # Extract GDS layer info from the wire's mask
                gds_layer = 0
                gds_datatype = 0
                if hasattr(prim, "mask") and hasattr(prim.mask, "gds_layer"):
                    gds_info = prim.mask.gds_layer
                    if isinstance(gds_info, tuple):
                        gds_layer, gds_datatype = gds_info
                    else:
                        gds_layer = gds_info

                # Extract sheet resistance if available
                sheet_r = 0.0
                if hasattr(prim, "sheet_resistance"):
                    sheet_r = float(prim.sheet_resistance)

                # Extract min width from the wire definition
                min_width = 0.0
                if hasattr(prim, "min_width"):
                    # PDKMaster uses micrometers, convert to nm
                    min_width = float(prim.min_width) * 1000

                entry = LayerStackEntry(
                    name=prim.name,
                    gds_layer=gds_layer,
                    gds_datatype=gds_datatype,
                    sheet_resistance_ohm_sq=sheet_r,
                    min_width_nm=min_width,
                )
                self._tech_data.layers.append(entry)

    def _extract_vias(self) -> None:
        """Extract via information from PDKMaster primitives."""
        assert self._tech_data is not None

        from pdkmaster.technology.primitive import Via

        for prim in self.tech.primitives:
            if isinstance(prim, Via):
                gds_layer = 0
                gds_datatype = 0
                if hasattr(prim, "mask") and hasattr(prim.mask, "gds_layer"):
                    gds_info = prim.mask.gds_layer
                    if isinstance(gds_info, tuple):
                        gds_layer, gds_datatype = gds_info
                    else:
                        gds_layer = gds_info

                # Get connected layer names
                bottom = ""
                top = ""
                if hasattr(prim, "bottom") and hasattr(prim.bottom, "name"):
                    bottom = prim.bottom.name
                if hasattr(prim, "top") and hasattr(prim.top, "name"):
                    top = prim.top.name

                via_def = ViaDefinition(
                    name=prim.name,
                    gds_layer=gds_layer,
                    gds_datatype=gds_datatype,
                    bottom_layer=bottom,
                    top_layer=top,
                )
                self._tech_data.vias.append(via_def)

    def _extract_dielectrics(self) -> None:
        """Extract dielectric information.

        Note: PDKMaster may not have explicit dielectric definitions.
        This provides defaults for SiO2-based ILD.
        """
        assert self._tech_data is not None

        # Default SiO2 dielectric constant
        default_permittivity = 3.9

        # Create dielectric layers between metal layers
        # This is a placeholder - actual values should come from process spec
        for layer in self._tech_data.layers:
            dielectric = DielectricLayer(
                name=f"ILD_{layer.name}",
                permittivity=default_permittivity,
                thickness_nm=0.0,  # To be filled from process spec
                height_nm=layer.height_nm,
            )
            self._tech_data.dielectrics.append(dielectric)

    def get_layer_stack(self) -> list[LayerStackEntry]:
        """Get the metal layer stack.

        Returns:
            List of LayerStackEntry from bottom to top.
        """
        return self.get_technology_data().layers

    def get_connectivity(self) -> dict[str, tuple[str, str]]:
        """Extract via connectivity from PDKMaster Via primitives.

        Returns:
            Dictionary mapping via name to (bottom_layer, top_layer) tuple.
        """
        tech_data = self.get_technology_data()
        connectivity = {}
        for via in tech_data.vias:
            connectivity[via.name] = (via.bottom_layer, via.top_layer)
        return connectivity


# IHP SG13G2 layer stack data from process specification
# Heights and thicknesses in nanometers
# Reference: SG13G2_os_process_spec.pdf

IHP_SG13G2_LAYER_DATA = {
    # Layer name: (thickness_nm, height_nm, sheet_resistance_ohm_sq)
    "Metal1": (360, 450, 0.075),
    "Metal2": (480, 1330, 0.055),
    "Metal3": (480, 2330, 0.055),
    "Metal4": (480, 3330, 0.055),
    "Metal5": (480, 4330, 0.055),
    "TopMetal1": (2000, 6530, 0.015),
    "TopMetal2": (2800, 10130, 0.010),
}

IHP_SG13G2_VIA_DATA = {
    # Via name: resistance_ohm per via
    "Via1": 2.0,
    "Via2": 2.0,
    "Via3": 2.0,
    "Via4": 2.0,
    "TopVia1": 0.8,
    "TopVia2": 0.5,
}

IHP_SG13G2_DIELECTRIC = {
    # Dielectric between layers - relative permittivity
    "SiO2": 3.9,
    "Si3N4": 7.5,
    # Low-k dielectrics if used
    "SiCOH": 3.0,
}


def get_ihp_sg13g2_technology() -> TechnologyData:
    """Get IHP SG13G2 technology data with process specification values.

    This function creates a TechnologyData object with layer thicknesses,
    sheet resistances, and dielectric properties from the IHP SG13G2
    process specification.

    Returns:
        TechnologyData for IHP SG13G2 process.
    """
    tech_data = TechnologyData(name="IHP-SG13G2")

    # Add metal layers
    for layer_name, (thickness, height, sheet_r) in IHP_SG13G2_LAYER_DATA.items():
        entry = LayerStackEntry(
            name=layer_name,
            gds_layer=0,  # Will be filled from PDKMaster
            thickness_nm=thickness,
            height_nm=height,
            sheet_resistance_ohm_sq=sheet_r,
        )
        tech_data.layers.append(entry)

    # Add vias
    for via_name, resistance in IHP_SG13G2_VIA_DATA.items():
        via = ViaDefinition(
            name=via_name,
            gds_layer=0,  # Will be filled from PDKMaster
            resistance_ohm=resistance,
        )
        tech_data.vias.append(via)

    # Add dielectric layers (simplified - one ILD between each metal)
    current_height = 0.0
    permittivity = IHP_SG13G2_DIELECTRIC["SiO2"]

    for i, layer in enumerate(tech_data.layers):
        if i == 0:
            # First dielectric from substrate to Metal1
            thickness = layer.height_nm
        else:
            # ILD between metal layers
            prev_layer = tech_data.layers[i - 1]
            prev_top = prev_layer.height_nm + prev_layer.thickness_nm
            thickness = layer.height_nm - prev_top

        dielectric = DielectricLayer(
            name=f"ILD{i}",
            permittivity=permittivity,
            thickness_nm=thickness,
            height_nm=current_height,
        )
        tech_data.dielectrics.append(dielectric)
        current_height = layer.height_nm + layer.thickness_nm

    return tech_data


def merge_pdk_with_process_spec(adapter: ExtractionAdapter) -> TechnologyData:
    """Merge PDKMaster layer info with process specification data.

    PDKMaster provides GDS layer mappings and design rules, while the
    process specification provides physical parameters like thickness
    and sheet resistance.

    Args:
        adapter: ExtractionAdapter with PDKMaster technology.

    Returns:
        TechnologyData with complete information.
    """
    # Get base data from PDKMaster
    tech_data = adapter.get_technology_data()

    # Merge with process spec data
    for layer in tech_data.layers:
        if layer.name in IHP_SG13G2_LAYER_DATA:
            thickness, height, sheet_r = IHP_SG13G2_LAYER_DATA[layer.name]
            layer.thickness_nm = thickness
            layer.height_nm = height
            # Only override sheet_r if not set by PDKMaster
            if layer.sheet_resistance_ohm_sq == 0.0:
                layer.sheet_resistance_ohm_sq = sheet_r

    for via in tech_data.vias:
        if via.name in IHP_SG13G2_VIA_DATA:
            via.resistance_ohm = IHP_SG13G2_VIA_DATA[via.name]

    return tech_data
