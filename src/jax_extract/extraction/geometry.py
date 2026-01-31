"""GDS geometry processing using KLayout.

This module handles reading GDS files and extracting geometry for
parasitic extraction.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from jax_extract.extraction.base import Conductor
from jax_extract.pdk.adapter import TechnologyData


@dataclass
class Polygon:
    """A polygon on a specific layer."""

    layer: str
    """Layer name."""

    vertices: NDArray[np.float64]
    """Polygon vertices as Nx2 array in micrometers."""

    def area(self) -> float:
        """Calculate polygon area using shoelace formula.

        Returns:
            Area in square micrometers.
        """
        x = self.vertices[:, 0]
        y = self.vertices[:, 1]
        # Shoelace formula
        return 0.5 * abs(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]) + x[-1] * y[0] - x[0] * y[-1])

    def perimeter(self) -> float:
        """Calculate polygon perimeter.

        Returns:
            Perimeter in micrometers.
        """
        diff = np.diff(self.vertices, axis=0, append=[self.vertices[0]])
        return float(np.sum(np.sqrt(np.sum(diff**2, axis=1))))

    def bounding_box(self) -> tuple[float, float, float, float]:
        """Get axis-aligned bounding box.

        Returns:
            Tuple of (xmin, ymin, xmax, ymax) in micrometers.
        """
        xmin = float(np.min(self.vertices[:, 0]))
        ymin = float(np.min(self.vertices[:, 1]))
        xmax = float(np.max(self.vertices[:, 0]))
        ymax = float(np.max(self.vertices[:, 1]))
        return xmin, ymin, xmax, ymax

    def centroid(self) -> tuple[float, float]:
        """Calculate polygon centroid.

        Returns:
            Tuple of (x, y) centroid coordinates in micrometers.
        """
        return float(np.mean(self.vertices[:, 0])), float(np.mean(self.vertices[:, 1]))


@dataclass
class Wire:
    """A wire segment for resistance extraction."""

    layer: str
    """Layer name."""

    start: tuple[float, float]
    """Start point (x, y) in micrometers."""

    end: tuple[float, float]
    """End point (x, y) in micrometers."""

    width: float
    """Wire width in micrometers."""

    def length(self) -> float:
        """Calculate wire length.

        Returns:
            Length in micrometers.
        """
        dx = self.end[0] - self.start[0]
        dy = self.end[1] - self.start[1]
        return float(np.sqrt(dx * dx + dy * dy))

    def resistance(self, sheet_resistance: float) -> float:
        """Calculate wire resistance.

        Args:
            sheet_resistance: Sheet resistance in Ohms/square.

        Returns:
            Resistance in Ohms.
        """
        return sheet_resistance * self.length() / self.width


@dataclass
class LayoutGeometry:
    """Complete geometry for a layout cell."""

    cell_name: str
    """Name of the cell."""

    polygons: dict[str, list[Polygon]] = field(default_factory=dict)
    """Polygons grouped by layer name."""

    wires: dict[str, list[Wire]] = field(default_factory=dict)
    """Extracted wires grouped by layer name."""

    labels: dict[str, list[tuple[str, float, float]]] = field(default_factory=dict)
    """Text labels as (text, x, y) grouped by layer."""

    def all_layers(self) -> set[str]:
        """Get all layers present in the geometry.

        Returns:
            Set of layer names.
        """
        return set(self.polygons.keys()) | set(self.wires.keys())

    def bounding_box(self) -> tuple[float, float, float, float]:
        """Get overall bounding box.

        Returns:
            Tuple of (xmin, ymin, xmax, ymax) in micrometers.
        """
        xmin = float("inf")
        ymin = float("inf")
        xmax = float("-inf")
        ymax = float("-inf")

        for layer_polys in self.polygons.values():
            for poly in layer_polys:
                bbox = poly.bounding_box()
                xmin = min(xmin, bbox[0])
                ymin = min(ymin, bbox[1])
                xmax = max(xmax, bbox[2])
                ymax = max(ymax, bbox[3])

        return xmin, ymin, xmax, ymax


class GeometryExtractor:
    """Extract geometry from GDS files using KLayout."""

    def __init__(self, tech_data: TechnologyData):
        """Initialize geometry extractor.

        Args:
            tech_data: Technology data for layer mapping.
        """
        self.tech_data = tech_data
        self._layer_map: dict[tuple[int, int], str] = {}
        self._build_layer_map()

    def _build_layer_map(self) -> None:
        """Build GDS layer number to layer name mapping."""
        for layer in self.tech_data.layers:
            key = (layer.gds_layer, layer.gds_datatype)
            self._layer_map[key] = layer.name

    def read_gds(self, gds_path: Path, cell_name: str | None = None) -> LayoutGeometry:
        """Read geometry from a GDS file.

        Args:
            gds_path: Path to GDS file.
            cell_name: Name of cell to read. If None, reads top cell.

        Returns:
            LayoutGeometry with polygons and labels.

        Raises:
            ImportError: If KLayout is not installed.
            FileNotFoundError: If GDS file doesn't exist.
            ValueError: If cell is not found.
        """
        try:
            import klayout.db as kdb
        except ImportError as e:
            raise ImportError(
                "KLayout Python module required. Install with: pip install klayout"
            ) from e

        if not gds_path.exists():
            raise FileNotFoundError(f"GDS file not found: {gds_path}")

        # Load layout
        layout = kdb.Layout()
        layout.read(str(gds_path))

        # Get cell
        if cell_name is None:
            cell = layout.top_cell()
            if cell is None:
                raise ValueError("No cells found in GDS file")
            cell_name = cell.name
        else:
            cell_idx = layout.cell_by_name(cell_name)
            if cell_idx is None:
                raise ValueError(f"Cell '{cell_name}' not found in GDS file")
            cell = layout.cell(cell_idx)

        geometry = LayoutGeometry(cell_name=cell_name)

        # Database unit to micrometers conversion
        dbu = layout.dbu

        # Extract polygons from each layer
        for layer_idx in layout.layer_indexes():
            info = layout.get_info(layer_idx)
            gds_key = (info.layer, info.datatype)

            # Map to layer name
            layer_name = self._layer_map.get(gds_key, f"L{info.layer}_{info.datatype}")

            if layer_name not in geometry.polygons:
                geometry.polygons[layer_name] = []

            # Get all shapes on this layer
            region = kdb.Region(cell.begin_shapes_rec(layer_idx))
            region.merge()  # Merge overlapping polygons

            for poly in region.each():
                # Convert to simple polygon (no holes for now)
                simple = poly.to_simple_polygon()
                vertices = []
                for pt in simple.each_point():
                    vertices.append([pt.x * dbu, pt.y * dbu])

                if vertices:
                    np_vertices = np.array(vertices, dtype=np.float64)
                    geometry.polygons[layer_name].append(
                        Polygon(layer=layer_name, vertices=np_vertices)
                    )

        return geometry

    def extract_connectivity(
        self,
        geometry: LayoutGeometry,
    ) -> list[Conductor]:
        """Extract connected conductors from geometry.

        Uses layer connectivity from technology to identify nets.

        Args:
            geometry: Layout geometry with polygons.

        Returns:
            List of Conductors representing connected nets.

        Note:
            This is a simplified implementation. A full implementation
            would use KLayout's LayoutToNetlist for proper connectivity.
        """
        import importlib.util

        if importlib.util.find_spec("klayout") is None:
            raise ImportError(
                "KLayout Python module required. Install with: pip install klayout"
            )

        # For now, return each layer's merged polygons as separate conductors
        # TODO: Use LayoutToNetlist for proper connectivity extraction
        conductors = []

        for net_id, (layer_name, polys) in enumerate(geometry.polygons.items()):
            conductor = Conductor(
                name=f"net_{layer_name}_{net_id}",
                net_id=net_id,
            )
            conductor.polygons[layer_name] = [p.vertices for p in polys]
            conductors.append(conductor)

        return conductors


def polygon_to_mesh(
    polygon: Polygon,
    height_nm: float,
    thickness_nm: float,
    mesh_size_nm: float = 100.0,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Convert polygon to 3D surface mesh for field solver.

    Creates a mesh of the conductor surface (top, bottom, and sides).

    Args:
        polygon: 2D polygon to extrude.
        height_nm: Base height in nanometers.
        thickness_nm: Thickness (extrusion height) in nanometers.
        mesh_size_nm: Target mesh element size in nanometers.

    Returns:
        Tuple of (vertices, triangles) where vertices is Nx3 and
        triangles is Mx3 array of vertex indices.
    """
    # Convert polygon vertices to nanometers
    vertices_2d = polygon.vertices * 1000  # um to nm

    n_pts = len(vertices_2d)
    z_bot = height_nm
    z_top = height_nm + thickness_nm

    # Create 3D vertices: bottom face, then top face
    vertices_3d = []

    # Bottom face vertices
    for x, y in vertices_2d:
        vertices_3d.append([x, y, z_bot])

    # Top face vertices
    for x, y in vertices_2d:
        vertices_3d.append([x, y, z_top])

    vertices = np.array(vertices_3d, dtype=np.float64)

    # Create triangles
    triangles = []

    # Bottom face (fan triangulation from first vertex)
    for i in range(1, n_pts - 1):
        triangles.append([0, i + 1, i])  # Note: reverse winding for outward normal

    # Top face
    for i in range(1, n_pts - 1):
        triangles.append([n_pts, n_pts + i, n_pts + i + 1])

    # Side faces (quads split into triangles)
    for i in range(n_pts):
        i_next = (i + 1) % n_pts
        # Bottom-left, bottom-right, top-right, top-left
        bl = i
        br = i_next
        tr = n_pts + i_next
        tl = n_pts + i

        triangles.append([bl, br, tr])
        triangles.append([bl, tr, tl])

    return vertices, np.array(triangles, dtype=np.int64)


def sample_surface_points(
    vertices: NDArray[np.float64],
    triangles: NDArray[np.int64],
    n_samples: int,
    rng: np.random.Generator | None = None,
) -> NDArray[np.float64]:
    """Sample random points on a triangulated surface.

    Args:
        vertices: Nx3 array of vertex positions.
        triangles: Mx3 array of triangle vertex indices.
        n_samples: Number of points to sample.
        rng: Random number generator (uses default if None).

    Returns:
        n_samples x 3 array of sample point positions.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Calculate triangle areas for weighted sampling
    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]

    # Cross product for area
    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.sqrt(np.sum(cross**2, axis=1))

    # Normalize to probabilities
    probs = areas / np.sum(areas)

    # Sample triangles
    tri_indices = rng.choice(len(triangles), size=n_samples, p=probs)

    # Sample random points within each triangle using barycentric coords
    r1 = rng.random(n_samples)
    r2 = rng.random(n_samples)

    # Ensure point is in triangle (not parallelogram)
    sqrt_r1 = np.sqrt(r1)
    u = 1 - sqrt_r1
    v = sqrt_r1 * (1 - r2)
    w = sqrt_r1 * r2

    # Compute points
    points = (
        u[:, np.newaxis] * vertices[triangles[tri_indices, 0]]
        + v[:, np.newaxis] * vertices[triangles[tri_indices, 1]]
        + w[:, np.newaxis] * vertices[triangles[tri_indices, 2]]
    )

    return points
