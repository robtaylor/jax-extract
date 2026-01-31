"""Simulation interface for jax-extract.

This module provides SPICE netlist generation and simulation
capabilities for validating extracted parasitic networks.
"""

from jax_extract.simulation.jax_spice import (
    AnalyticalSimulator,
    JaxSpiceSimulator,
    SimulationResult,
    validate_extraction,
)
from jax_extract.simulation.spice_netlist import (
    NetlistFormat,
    SimulationSetup,
    SimulationTestbench,
    SpiceNetlistWriter,
    generate_step_response_testbench,
    rc_network_to_spice,
)

__all__ = [
    # Netlist generation
    "SpiceNetlistWriter",
    "NetlistFormat",
    "SimulationSetup",
    "SimulationTestbench",
    "rc_network_to_spice",
    "generate_step_response_testbench",
    # Simulation
    "JaxSpiceSimulator",
    "AnalyticalSimulator",
    "SimulationResult",
    "validate_extraction",
]
