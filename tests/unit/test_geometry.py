"""Unit tests for geometry processing module."""

import numpy as np

from jax_extract.extraction.geometry import (
    LayoutGeometry,
    Polygon,
    Wire,
    polygon_to_mesh,
    sample_surface_points,
)


class TestPolygon:
    """Tests for Polygon class."""

    def test_create_polygon(self):
        """Test creating a polygon."""
        vertices = np.array([
            [0.0, 0.0],
            [10.0, 0.0],
            [10.0, 5.0],
            [0.0, 5.0],
        ])
        poly = Polygon(layer="Metal1", vertices=vertices)

        assert poly.layer == "Metal1"
        assert len(poly.vertices) == 4

    def test_area_rectangle(self):
        """Test area calculation for rectangle."""
        vertices = np.array([
            [0.0, 0.0],
            [10.0, 0.0],
            [10.0, 5.0],
            [0.0, 5.0],
        ])
        poly = Polygon(layer="Metal1", vertices=vertices)

        area = poly.area()
        assert abs(area - 50.0) < 1e-10

    def test_area_triangle(self):
        """Test area calculation for triangle."""
        vertices = np.array([
            [0.0, 0.0],
            [10.0, 0.0],
            [5.0, 10.0],
        ])
        poly = Polygon(layer="Metal1", vertices=vertices)

        area = poly.area()
        # Triangle: 0.5 * base * height = 0.5 * 10 * 10 = 50
        assert abs(area - 50.0) < 1e-10

    def test_perimeter(self):
        """Test perimeter calculation."""
        vertices = np.array([
            [0.0, 0.0],
            [10.0, 0.0],
            [10.0, 5.0],
            [0.0, 5.0],
        ])
        poly = Polygon(layer="Metal1", vertices=vertices)

        perimeter = poly.perimeter()
        expected = 2 * (10.0 + 5.0)
        assert abs(perimeter - expected) < 1e-10

    def test_bounding_box(self):
        """Test bounding box calculation."""
        vertices = np.array([
            [1.0, 2.0],
            [11.0, 2.0],
            [11.0, 7.0],
            [1.0, 7.0],
        ])
        poly = Polygon(layer="Metal1", vertices=vertices)

        xmin, ymin, xmax, ymax = poly.bounding_box()
        assert xmin == 1.0
        assert ymin == 2.0
        assert xmax == 11.0
        assert ymax == 7.0

    def test_centroid(self):
        """Test centroid calculation."""
        vertices = np.array([
            [0.0, 0.0],
            [10.0, 0.0],
            [10.0, 10.0],
            [0.0, 10.0],
        ])
        poly = Polygon(layer="Metal1", vertices=vertices)

        cx, cy = poly.centroid()
        assert abs(cx - 5.0) < 1e-10
        assert abs(cy - 5.0) < 1e-10


class TestWire:
    """Tests for Wire class."""

    def test_create_wire(self):
        """Test creating a wire."""
        wire = Wire(
            layer="Metal1",
            start=(0.0, 0.0),
            end=(10.0, 0.0),
            width=0.5,
        )

        assert wire.layer == "Metal1"
        assert wire.width == 0.5

    def test_wire_length(self):
        """Test wire length calculation."""
        wire = Wire(layer="Metal1", start=(0.0, 0.0), end=(10.0, 0.0), width=0.5)
        assert abs(wire.length() - 10.0) < 1e-10

        # Diagonal wire
        wire2 = Wire(layer="Metal1", start=(0.0, 0.0), end=(3.0, 4.0), width=0.5)
        assert abs(wire2.length() - 5.0) < 1e-10

    def test_wire_resistance(self):
        """Test wire resistance calculation."""
        wire = Wire(layer="Metal1", start=(0.0, 0.0), end=(10.0, 0.0), width=0.5)

        r = wire.resistance(0.075)  # Sheet resistance
        expected = 0.075 * 10.0 / 0.5
        assert abs(r - expected) < 1e-10


class TestLayoutGeometry:
    """Tests for LayoutGeometry class."""

    def test_create_geometry(self):
        """Test creating layout geometry."""
        geom = LayoutGeometry(cell_name="test_cell")

        assert geom.cell_name == "test_cell"
        assert len(geom.polygons) == 0

    def test_all_layers(self):
        """Test getting all layers."""
        geom = LayoutGeometry(cell_name="test")

        vertices = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float64)
        geom.polygons["Metal1"] = [Polygon("Metal1", vertices)]
        geom.polygons["Metal2"] = [Polygon("Metal2", vertices)]
        geom.wires["Metal3"] = [Wire("Metal3", (0, 0), (10, 0), 0.5)]

        layers = geom.all_layers()
        assert "Metal1" in layers
        assert "Metal2" in layers
        assert "Metal3" in layers
        assert len(layers) == 3

    def test_bounding_box(self):
        """Test overall bounding box."""
        geom = LayoutGeometry(cell_name="test")

        v1 = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float64)
        v2 = np.array([[20, 20], [30, 20], [30, 30], [20, 30]], dtype=np.float64)

        geom.polygons["Metal1"] = [Polygon("Metal1", v1)]
        geom.polygons["Metal2"] = [Polygon("Metal2", v2)]

        xmin, ymin, xmax, ymax = geom.bounding_box()
        assert xmin == 0.0
        assert ymin == 0.0
        assert xmax == 30.0
        assert ymax == 30.0


class TestPolygonToMesh:
    """Tests for 3D mesh generation."""

    def test_basic_mesh(self):
        """Test basic mesh generation."""
        vertices_2d = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ])
        poly = Polygon(layer="Metal1", vertices=vertices_2d)

        vertices, triangles = polygon_to_mesh(
            poly,
            height_nm=450.0,
            thickness_nm=360.0,
        )

        # Should have 8 vertices (4 bottom + 4 top)
        assert vertices.shape == (8, 3)

        # Check z coordinates
        assert np.all(vertices[:4, 2] == 450.0)  # Bottom
        assert np.all(vertices[4:, 2] == 810.0)  # Top (450 + 360)

        # Should have triangles for top, bottom, and 4 sides
        assert triangles.shape[0] > 0

    def test_mesh_x_y_correct(self):
        """Test that x,y coordinates are correct in nm."""
        vertices_2d = np.array([
            [0.0, 0.0],
            [10.0, 0.0],
            [10.0, 5.0],
            [0.0, 5.0],
        ])
        poly = Polygon(layer="Metal1", vertices=vertices_2d)

        vertices, _ = polygon_to_mesh(poly, height_nm=0.0, thickness_nm=100.0)

        # Vertices should be in nanometers (um * 1000)
        assert vertices[0, 0] == 0.0
        assert vertices[1, 0] == 10000.0  # 10um = 10000nm
        assert vertices[2, 1] == 5000.0   # 5um = 5000nm


class TestSampleSurfacePoints:
    """Tests for surface point sampling."""

    def test_sample_points_count(self):
        """Test that correct number of points are sampled."""
        # Simple triangle mesh
        vertices = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
        ], dtype=np.float64)
        triangles = np.array([[0, 1, 2]], dtype=np.int64)

        points = sample_surface_points(vertices, triangles, n_samples=100)
        assert points.shape == (100, 3)

    def test_sample_points_on_surface(self):
        """Test that points are on the triangle surface."""
        # Triangle in z=0 plane
        vertices = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0.5, 1, 0],
        ], dtype=np.float64)
        triangles = np.array([[0, 1, 2]], dtype=np.int64)

        points = sample_surface_points(vertices, triangles, n_samples=100)

        # All points should have z=0
        assert np.allclose(points[:, 2], 0.0)

    def test_sample_points_reproducible(self):
        """Test that same seed gives same results."""
        vertices = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
        ], dtype=np.float64)
        triangles = np.array([[0, 1, 2]], dtype=np.int64)

        rng1 = np.random.default_rng(42)
        points1 = sample_surface_points(vertices, triangles, 50, rng1)

        rng2 = np.random.default_rng(42)
        points2 = sample_surface_points(vertices, triangles, 50, rng2)

        assert np.allclose(points1, points2)
