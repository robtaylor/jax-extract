"""Shared pytest fixtures for jax-extract tests."""

import numpy as np
import pytest

from jax_extract.pdk.adapter import (
    LayerStackEntry,
    TechnologyData,
    ViaDefinition,
    get_ihp_sg13g2_technology,
)


@pytest.fixture
def ihp_tech() -> TechnologyData:
    """Get IHP SG13G2 technology data."""
    return get_ihp_sg13g2_technology()


@pytest.fixture
def simple_tech() -> TechnologyData:
    """Get simplified technology for testing."""
    tech = TechnologyData(name="test")
    tech.layers = [
        LayerStackEntry(
            name="Metal1",
            gds_layer=8,
            thickness_nm=360,
            height_nm=450,
            sheet_resistance_ohm_sq=0.075,
            min_width_nm=500,
        ),
        LayerStackEntry(
            name="Metal2",
            gds_layer=10,
            thickness_nm=480,
            height_nm=1330,
            sheet_resistance_ohm_sq=0.055,
            min_width_nm=600,
        ),
    ]
    tech.vias = [
        ViaDefinition(
            name="Via1",
            gds_layer=19,
            bottom_layer="Metal1",
            top_layer="Metal2",
            resistance_ohm=2.0,
        ),
    ]
    return tech


@pytest.fixture
def square_polygon():
    """Create a 10um x 10um square polygon."""
    from jax_extract.extraction.geometry import Polygon

    vertices = np.array([
        [0.0, 0.0],
        [10.0, 0.0],
        [10.0, 10.0],
        [0.0, 10.0],
    ], dtype=np.float64)
    return Polygon(layer="Metal1", vertices=vertices)


@pytest.fixture
def wire_polygon():
    """Create a long narrow wire polygon (100um x 0.5um)."""
    from jax_extract.extraction.geometry import Polygon

    vertices = np.array([
        [0.0, 0.0],
        [100.0, 0.0],
        [100.0, 0.5],
        [0.0, 0.5],
    ], dtype=np.float64)
    return Polygon(layer="Metal1", vertices=vertices)
