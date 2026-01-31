"""PVT corner handling for parasitic extraction.

This module extends PDKMaster's technology data with process corner
variations for capacitance and resistance scaling.
"""

from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple

from jax_extract.pdk.adapter import TechnologyData


class CornerType(str, Enum):
    """Standard PVT corners for parasitic extraction."""

    TYPICAL = "typical"
    """Nominal process conditions."""

    CBEST = "cbest"
    """Best-case capacitance (low C, fast timing)."""

    CWORST = "cworst"
    """Worst-case capacitance (high C, slow timing)."""

    RCBEST = "rcbest"
    """Best-case RC (low R, low C, fastest)."""

    RCWORST = "rcworst"
    """Worst-case RC (high R, high C, slowest)."""


class CornerFactors(NamedTuple):
    """Scaling factors for a process corner."""

    capacitance: float
    """Capacitance scaling factor (1.0 = nominal)."""

    resistance: float
    """Resistance scaling factor (1.0 = nominal)."""


# Standard corner scaling factors
# These are typical values; actual values should come from foundry data
CORNER_FACTORS: dict[CornerType, CornerFactors] = {
    CornerType.TYPICAL: CornerFactors(1.0, 1.0),
    CornerType.CBEST: CornerFactors(0.85, 1.0),
    CornerType.CWORST: CornerFactors(1.15, 1.0),
    CornerType.RCBEST: CornerFactors(0.9, 0.85),
    CornerType.RCWORST: CornerFactors(1.1, 1.15),
}


@dataclass
class CornerConfig:
    """Configuration for a specific process corner."""

    corner_type: CornerType
    """The corner being configured."""

    temperature_c: float = 25.0
    """Temperature in Celsius."""

    voltage_v: float = 1.2
    """Supply voltage (for voltage-dependent effects)."""

    capacitance_factor: float = 1.0
    """Capacitance scaling factor."""

    resistance_factor: float = 1.0
    """Resistance scaling factor."""

    @classmethod
    def from_corner_type(
        cls,
        corner_type: CornerType | str,
        temperature_c: float = 25.0,
        voltage_v: float = 1.2,
    ) -> "CornerConfig":
        """Create corner config from corner type.

        Args:
            corner_type: Corner type or string name.
            temperature_c: Temperature in Celsius.
            voltage_v: Supply voltage in Volts.

        Returns:
            CornerConfig with appropriate scaling factors.
        """
        if isinstance(corner_type, str):
            corner_type = CornerType(corner_type.lower())

        factors = CORNER_FACTORS[corner_type]
        return cls(
            corner_type=corner_type,
            temperature_c=temperature_c,
            voltage_v=voltage_v,
            capacitance_factor=factors.capacitance,
            resistance_factor=factors.resistance,
        )


def apply_corner_to_technology(
    tech_data: TechnologyData,
    corner: CornerConfig,
) -> TechnologyData:
    """Apply corner scaling to technology data.

    Creates a new TechnologyData with scaled values. The original
    is not modified.

    Args:
        tech_data: Base technology data.
        corner: Corner configuration with scaling factors.

    Returns:
        New TechnologyData with scaled parameters.
    """
    # Create a copy with scaled values
    from copy import deepcopy

    scaled = deepcopy(tech_data)
    scaled.name = f"{tech_data.name}_{corner.corner_type.value}"

    # Scale sheet resistance
    for layer in scaled.layers:
        layer.sheet_resistance_ohm_sq *= corner.resistance_factor

    # Scale via resistance
    for via in scaled.vias:
        via.resistance_ohm *= corner.resistance_factor

    # Scale dielectric permittivity (affects capacitance)
    # Higher permittivity = higher capacitance
    for dielectric in scaled.dielectrics:
        dielectric.permittivity *= corner.capacitance_factor

    return scaled


def get_temperature_resistance_factor(
    temperature_c: float,
    reference_temp_c: float = 25.0,
    tco: float = 0.003,
) -> float:
    """Calculate temperature-dependent resistance scaling.

    Metal resistance increases with temperature due to phonon scattering.

    Args:
        temperature_c: Operating temperature in Celsius.
        reference_temp_c: Reference temperature (typically 25C).
        tco: Temperature coefficient of resistance (1/K).
              Typical value for aluminum: 0.003-0.004
              Typical value for copper: 0.004

    Returns:
        Resistance scaling factor relative to reference temperature.
    """
    delta_t = temperature_c - reference_temp_c
    return 1.0 + tco * delta_t


# IHP SG13G2 specific corner data (if available from PDK)
# These would be calibrated from foundry test chip data
IHP_SG13G2_CORNERS = {
    CornerType.TYPICAL: CornerFactors(1.0, 1.0),
    CornerType.CBEST: CornerFactors(0.85, 1.0),
    CornerType.CWORST: CornerFactors(1.15, 1.0),
    CornerType.RCBEST: CornerFactors(0.88, 0.82),
    CornerType.RCWORST: CornerFactors(1.12, 1.18),
}
