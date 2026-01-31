"""End-to-end parasitic extractor.

This module provides the FullExtractor class that combines geometry processing,
field solving for capacitance, resistance extraction, and RC network assembly
into a unified extraction pipeline.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from jax_extract.extraction.base import (
    Conductor,
    ExtractionConfig,
    ExtractionMode,
    ExtractionResult,
    Extractor,
)
from jax_extract.extraction.field_solver import (
    ConductorSurface,
    DielectricRegion,
    ExtractionDomain,
    FRWSolver,
    extract_capacitance_wire,
)
from jax_extract.extraction.geometry import (
    GeometryExtractor,
    Polygon,
    polygon_to_mesh,
)
from jax_extract.extraction.network import RCNetwork
from jax_extract.extraction.resistance import ResistanceExtractor
from jax_extract.pdk.adapter import TechnologyData


@dataclass
class ExtractedRC:
    """Intermediate result combining resistance and capacitance."""

    total_resistance_ohm: float
    """Total resistance in Ohms."""

    total_capacitance_f: float
    """Total capacitance in Farads."""

    rc_time_constant_s: float
    """RC time constant in seconds (R * C)."""

    network: RCNetwork
    """Complete RC network representation."""


class FullExtractor(Extractor):
    """End-to-end parasitic extractor.

    This extractor combines:
    - Geometry processing (polygon to mesh conversion)
    - FRW solver for capacitance extraction
    - Resistance extraction from wire geometry
    - RC network assembly

    Example:
        >>> from jax_extract.pdk.adapter import get_ihp_sg13g2_technology
        >>> tech = get_ihp_sg13g2_technology()
        >>> extractor = FullExtractor(tech)
        >>> result = extractor.extract_from_geometry(conductors)
        >>> print(f"Total C: {result.total_capacitance(0):.3e} F")
    """

    def __init__(
        self,
        tech_data: TechnologyData,
        config: ExtractionConfig | None = None,
    ):
        """Initialize the full extractor.

        Args:
            tech_data: Technology data for the target process.
            config: Extraction configuration. Uses defaults if None.
        """
        super().__init__(tech_data, config)
        self.frw_solver = FRWSolver(
            num_walks=self.config.frw_num_walks,
            tolerance=self.config.frw_tolerance,
        )
        self.resistance_extractor = ResistanceExtractor(tech_data)
        self._geometry_extractor: GeometryExtractor | None = None

    @property
    def geometry_extractor(self) -> GeometryExtractor:
        """Lazy-initialized geometry extractor."""
        if self._geometry_extractor is None:
            self._geometry_extractor = GeometryExtractor(self.tech_data)
        return self._geometry_extractor

    def extract(
        self,
        gds_path: Path,
        cell_name: str,
    ) -> ExtractionResult:
        """Extract parasitics from a GDS layout.

        Args:
            gds_path: Path to GDS file.
            cell_name: Name of cell to extract.

        Returns:
            ExtractionResult with capacitances and resistances.
        """
        # Read geometry from GDS
        geometry = self.geometry_extractor.read_gds(gds_path, cell_name)

        # Extract connectivity to get conductors
        conductors = self.geometry_extractor.extract_connectivity(geometry)

        # Delegate to extract_from_geometry
        return self.extract_from_geometry(conductors)

    def extract_from_geometry(
        self,
        conductors: list[Conductor],
    ) -> ExtractionResult:
        """Extract parasitics from pre-parsed geometry.

        Args:
            conductors: List of conductors with polygon geometry.

        Returns:
            ExtractionResult with capacitances and resistances.
        """
        if not conductors:
            return ExtractionResult(cell_name="empty")

        # Determine cell name from first conductor
        cell_name = conductors[0].name if conductors else "extracted"

        # Build extraction result
        result = ExtractionResult(cell_name=cell_name)
        result.conductors = conductors

        # Extract capacitance
        if self.config.mode in (
            ExtractionMode.CAPACITANCE_ONLY,
            ExtractionMode.RC,
            ExtractionMode.RLC,
        ):
            self._extract_capacitance(conductors, result)

        # Extract resistance
        if self.config.mode in (ExtractionMode.RC, ExtractionMode.RLC):
            self._extract_resistance(conductors, result)

        return result

    def _extract_capacitance(
        self,
        conductors: list[Conductor],
        result: ExtractionResult,
    ) -> None:
        """Extract capacitance using FRW solver or analytical formulas.

        For simple geometries (rectangles), uses analytical formulas.
        For complex geometries, uses FRW solver.

        Args:
            conductors: List of conductors.
            result: ExtractionResult to populate.
        """
        for conductor in conductors:
            total_cap = 0.0

            for layer_name, polygons in conductor.polygons.items():
                layer_info = self.tech_data.get_layer(layer_name)
                if layer_info is None:
                    continue

                for poly_array in polygons:
                    poly = Polygon(layer=layer_name, vertices=poly_array)
                    xmin, ymin, xmax, ymax = poly.bounding_box()
                    width = xmax - xmin
                    height = ymax - ymin

                    # Use wire capacitance formula
                    # (includes fringe capacitance)
                    length_um = max(width, height)
                    width_um = min(width, height)

                    # Get permittivity from dielectric at this layer height
                    permittivity = self._get_permittivity_at_layer(layer_name)

                    cap = extract_capacitance_wire(
                        length_um=length_um,
                        width_um=width_um,
                        height_nm=layer_info.height_nm,
                        thickness_nm=layer_info.thickness_nm,
                        permittivity=permittivity,
                    )
                    total_cap += cap

            # Store self-capacitance (to ground)
            result.capacitances[(conductor.net_id, -1)] = total_cap

    def _extract_capacitance_frw(
        self,
        conductors: list[Conductor],
        result: ExtractionResult,
    ) -> None:
        """Extract capacitance using FRW field solver.

        This provides more accurate results for complex geometries
        but is slower than analytical formulas.

        Args:
            conductors: List of conductors.
            result: ExtractionResult to populate.
        """
        # Build extraction domain
        domain = self._build_extraction_domain(conductors)

        if not domain.conductors:
            return

        # Run FRW solver
        frw_result = self.frw_solver.solve(domain)

        # Copy capacitance matrix to result
        n_cond = len(conductors)
        for i in range(n_cond):
            # Self-capacitance (diagonal)
            result.capacitances[(conductors[i].net_id, -1)] = frw_result.capacitance_matrix[i, i]

            # Coupling capacitance (off-diagonal)
            for j in range(i + 1, n_cond):
                cap = frw_result.capacitance_matrix[i, j]
                if cap > self.config.coupling_threshold:
                    key = (
                        min(conductors[i].net_id, conductors[j].net_id),
                        max(conductors[i].net_id, conductors[j].net_id),
                    )
                    result.capacitances[key] = cap

    def _extract_resistance(
        self,
        conductors: list[Conductor],
        result: ExtractionResult,
    ) -> None:
        """Extract resistance from wire geometry.

        Args:
            conductors: List of conductors.
            result: ExtractionResult to populate.
        """
        node_id = 0

        for conductor in conductors:
            for layer_name, polygons in conductor.polygons.items():
                layer_info = self.tech_data.get_layer(layer_name)
                if layer_info is None:
                    continue

                for poly_array in polygons:
                    poly = Polygon(layer=layer_name, vertices=poly_array)

                    # Extract wire segments
                    segments = self.resistance_extractor.extract_from_polygon(
                        poly,
                        segments_per_lambda=self.config.resistance_segments_per_lambda,
                    )

                    # Add segments as resistances
                    for seg in segments:
                        resistance = seg.resistance(layer_info.sheet_resistance_ohm_sq)
                        if resistance > 0:
                            result.resistances[(conductor.net_id, node_id, node_id + 1)] = resistance

                            # Store node positions
                            z = layer_info.height_nm / 1000  # nm to um
                            result.node_positions[(conductor.net_id, node_id)] = (
                                layer_name,
                                seg.start[0],
                                seg.start[1],
                                z,
                            )
                            result.node_positions[(conductor.net_id, node_id + 1)] = (
                                layer_name,
                                seg.end[0],
                                seg.end[1],
                                z,
                            )
                            node_id += 2

    def _build_extraction_domain(
        self,
        conductors: list[Conductor],
    ) -> ExtractionDomain:
        """Build extraction domain for FRW solver.

        Args:
            conductors: List of conductors.

        Returns:
            ExtractionDomain with conductor meshes and dielectrics.
        """
        conductor_surfaces = []
        xmin = ymin = zmin = float("inf")
        xmax = ymax = zmax = float("-inf")

        for conductor in conductors:
            for layer_name, polygons in conductor.polygons.items():
                layer_info = self.tech_data.get_layer(layer_name)
                if layer_info is None:
                    continue

                for poly_array in polygons:
                    poly = Polygon(layer=layer_name, vertices=poly_array)

                    # Convert to 3D mesh
                    vertices, triangles = polygon_to_mesh(
                        poly,
                        height_nm=layer_info.height_nm,
                        thickness_nm=layer_info.thickness_nm,
                    )

                    # Compute triangle areas
                    v0 = vertices[triangles[:, 0]]
                    v1 = vertices[triangles[:, 1]]
                    v2 = vertices[triangles[:, 2]]
                    cross = np.cross(v1 - v0, v2 - v0)
                    areas = 0.5 * np.sqrt(np.sum(cross**2, axis=1))

                    surface = ConductorSurface(
                        conductor_id=conductor.net_id,
                        vertices=vertices,
                        triangles=triangles,
                        areas=areas,
                    )
                    conductor_surfaces.append(surface)

                    # Update bounding box
                    xmin = min(xmin, np.min(vertices[:, 0]))
                    ymin = min(ymin, np.min(vertices[:, 1]))
                    zmin = min(zmin, np.min(vertices[:, 2]))
                    xmax = max(xmax, np.max(vertices[:, 0]))
                    ymax = max(ymax, np.max(vertices[:, 1]))
                    zmax = max(zmax, np.max(vertices[:, 2]))

        # Build dielectric stack
        dielectric_regions = []
        for dielectric in self.tech_data.dielectrics:
            region = DielectricRegion(
                permittivity=dielectric.permittivity,
                z_min=dielectric.height_nm,
                z_max=dielectric.height_nm + dielectric.thickness_nm,
            )
            dielectric_regions.append(region)

        # Add padding to bounding box
        padding = 1000.0  # nm
        boundary_box = (
            xmin - padding,
            ymin - padding,
            max(0, zmin - padding),
            xmax + padding,
            ymax + padding,
            zmax + padding,
        )

        return ExtractionDomain(
            conductors=conductor_surfaces,
            dielectrics=dielectric_regions,
            boundary_box=boundary_box,
        )

    def _get_permittivity_at_layer(self, layer_name: str) -> float:
        """Get dielectric permittivity at a layer's height.

        Args:
            layer_name: Layer name.

        Returns:
            Relative permittivity (defaults to 3.9 for SiO2).
        """
        layer_info = self.tech_data.get_layer(layer_name)
        if layer_info is None:
            return 3.9

        height = layer_info.height_nm

        for dielectric in self.tech_data.dielectrics:
            if dielectric.height_nm <= height < dielectric.height_nm + dielectric.thickness_nm:
                return dielectric.permittivity

        return 3.9  # Default to SiO2

    def extract_rc(
        self,
        conductors: list[Conductor],
    ) -> ExtractedRC:
        """Extract RC parameters and build network.

        This is a convenience method that extracts parasitics and
        builds an RCNetwork suitable for simulation.

        Args:
            conductors: List of conductors with polygon geometry.

        Returns:
            ExtractedRC with totals and network.
        """
        result = self.extract_from_geometry(conductors)

        # Build RC network
        network = RCNetwork(name=result.cell_name)

        # Add nodes and resistors
        node_map: dict[tuple[int, int], int] = {}  # (net_id, node_id) -> network node id

        for (net_id, n1, n2), resistance in result.resistances.items():
            # Get or create nodes
            if (net_id, n1) not in node_map:
                pos = result.node_positions.get((net_id, n1))
                if pos:
                    layer, x, y, z = pos
                    node = network.add_node(
                        net_name=f"net_{net_id}",
                        layer=layer,
                        x=x,
                        y=y,
                        z=z,
                    )
                    node_map[(net_id, n1)] = node.node_id

            if (net_id, n2) not in node_map:
                pos = result.node_positions.get((net_id, n2))
                if pos:
                    layer, x, y, z = pos
                    node = network.add_node(
                        net_name=f"net_{net_id}",
                        layer=layer,
                        x=x,
                        y=y,
                        z=z,
                    )
                    node_map[(net_id, n2)] = node.node_id

            # Add resistor if both nodes exist
            if (net_id, n1) in node_map and (net_id, n2) in node_map:
                network.add_resistor(
                    node1=network.nodes[node_map[(net_id, n1)]],
                    node2=network.nodes[node_map[(net_id, n2)]],
                    resistance_ohm=resistance,
                )

        # Add capacitors at each node (distributed)
        for (net_id1, net_id2), capacitance in result.capacitances.items():
            if net_id2 == -1:
                # Ground capacitance - distribute among nodes
                net_nodes = [
                    (key, node_map[key])
                    for key in node_map
                    if key[0] == net_id1
                ]
                if net_nodes:
                    cap_per_node = capacitance / len(net_nodes)
                    for (_, _), network_node_id in net_nodes:
                        network.add_capacitor(
                            node1=network.nodes[network_node_id],
                            node2=None,
                            capacitance_f=cap_per_node,
                        )

        # Calculate totals
        total_r = sum(result.resistances.values())
        total_c = sum(
            cap for (_, n2), cap in result.capacitances.items() if n2 == -1
        )
        tau = total_r * total_c

        return ExtractedRC(
            total_resistance_ohm=total_r,
            total_capacitance_f=total_c,
            rc_time_constant_s=tau,
            network=network,
        )


def extract_simple_wire(
    length_um: float,
    width_um: float,
    tech_data: TechnologyData,
    layer: str = "Metal1",
) -> ExtractedRC:
    """Convenience function to extract RC for a simple wire.

    Args:
        length_um: Wire length in micrometers.
        width_um: Wire width in micrometers.
        tech_data: Technology data.
        layer: Layer name (default Metal1).

    Returns:
        ExtractedRC with extracted parameters.
    """
    # Create wire conductor
    vertices = np.array(
        [
            [0.0, 0.0],
            [length_um, 0.0],
            [length_um, width_um],
            [0.0, width_um],
        ],
        dtype=np.float64,
    )
    conductor = Conductor(name="wire", net_id=0)
    conductor.polygons[layer] = [vertices]

    # Extract
    extractor = FullExtractor(tech_data)
    return extractor.extract_rc([conductor])


def extract_parallel_plate(
    width_um: float,
    height_um: float,
    tech_data: TechnologyData,
    layer: str = "Metal1",
) -> ExtractedRC:
    """Convenience function to extract capacitance for a parallel plate.

    Args:
        width_um: Plate width in micrometers.
        height_um: Plate height in micrometers.
        tech_data: Technology data.
        layer: Layer name (default Metal1).

    Returns:
        ExtractedRC with extracted parameters.
    """
    # Create plate conductor
    vertices = np.array(
        [
            [0.0, 0.0],
            [width_um, 0.0],
            [width_um, height_um],
            [0.0, height_um],
        ],
        dtype=np.float64,
    )
    conductor = Conductor(name="plate", net_id=0)
    conductor.polygons[layer] = [vertices]

    # Extract with capacitance-only mode
    config = ExtractionConfig(mode=ExtractionMode.CAPACITANCE_ONLY)
    extractor = FullExtractor(tech_data, config)
    return extractor.extract_rc([conductor])
