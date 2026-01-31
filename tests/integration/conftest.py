"""Shared fixtures for integration tests."""

import numpy as np
import pytest

from jax_extract.extraction.base import Conductor
from jax_extract.pdk.adapter import (
    DielectricLayer,
    LayerStackEntry,
    TechnologyData,
    ViaDefinition,
)


@pytest.fixture
def integration_tech() -> TechnologyData:
    """Technology data suitable for integration tests.

    This is a simplified technology with well-defined parameters
    for easier validation against analytical formulas.
    """
    tech = TechnologyData(name="integration_test")
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
        LayerStackEntry(
            name="Metal3",
            gds_layer=30,
            thickness_nm=480,
            height_nm=2290,
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
        ViaDefinition(
            name="Via2",
            gds_layer=29,
            bottom_layer="Metal2",
            top_layer="Metal3",
            resistance_ohm=2.0,
        ),
    ]
    tech.dielectrics = [
        DielectricLayer(
            name="ILD1",
            permittivity=3.9,
            thickness_nm=450,
            height_nm=0,
        ),
        DielectricLayer(
            name="ILD2",
            permittivity=3.9,
            thickness_nm=520,
            height_nm=810,
        ),
    ]
    return tech


def make_rectangular_conductor(
    name: str,
    net_id: int,
    x_min: float,
    y_min: float,
    width: float,
    height: float,
    layer: str,
) -> Conductor:
    """Create a conductor with a single rectangular polygon.

    Args:
        name: Conductor name.
        net_id: Unique numeric identifier.
        x_min: Left edge X coordinate (um).
        y_min: Bottom edge Y coordinate (um).
        width: Width in X direction (um).
        height: Height in Y direction (um).
        layer: Layer name.

    Returns:
        Conductor with rectangular polygon.
    """
    vertices = np.array(
        [
            [x_min, y_min],
            [x_min + width, y_min],
            [x_min + width, y_min + height],
            [x_min, y_min + height],
        ],
        dtype=np.float64,
    )
    conductor = Conductor(name=name, net_id=net_id)
    conductor.polygons[layer] = [vertices]
    return conductor


@pytest.fixture
def parallel_plate_geometry(integration_tech):
    """Create parallel plate geometry for capacitance validation.

    Returns a 10um x 10um plate on Metal1.
    """
    conductor = make_rectangular_conductor(
        name="plate",
        net_id=0,
        x_min=0.0,
        y_min=0.0,
        width=10.0,
        height=10.0,
        layer="Metal1",
    )
    return [conductor]


@pytest.fixture
def wire_geometry(integration_tech):
    """Create wire geometry for resistance validation.

    Returns a 100um x 0.5um wire on Metal1.
    """
    conductor = make_rectangular_conductor(
        name="wire",
        net_id=0,
        x_min=0.0,
        y_min=0.0,
        width=100.0,
        height=0.5,
        layer="Metal1",
    )
    return [conductor]


@pytest.fixture
def rc_ladder_geometry(integration_tech):
    """Create geometry for RC ladder validation.

    Returns a series of connected wire segments.
    """
    # Create 5 connected segments forming an RC ladder
    conductors = []
    for i in range(5):
        conductor = make_rectangular_conductor(
            name=f"seg_{i}",
            net_id=0,  # All part of same net
            x_min=i * 20.0,
            y_min=0.0,
            width=20.0,
            height=0.5,
            layer="Metal1",
        )
        conductors.append(conductor)
    return conductors
