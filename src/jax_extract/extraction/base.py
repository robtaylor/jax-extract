"""Base classes and interfaces for parasitic extraction.

This module defines the core abstractions for the extraction engine.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from jax_extract.pdk.adapter import TechnologyData


class ExtractionMode(str, Enum):
    """Extraction fidelity modes."""

    CAPACITANCE_ONLY = "c"
    """Extract only capacitances."""

    RC = "rc"
    """Extract resistances and capacitances."""

    RLC = "rlc"
    """Extract resistances, inductances, and capacitances."""


@dataclass
class ExtractionConfig:
    """Configuration for parasitic extraction."""

    mode: ExtractionMode = ExtractionMode.RC
    """Extraction mode (C, RC, or RLC)."""

    coupling_threshold: float = 1e-18
    """Minimum coupling capacitance to include (Farads)."""

    max_coupling_distance_um: float = 10.0
    """Maximum distance for coupling extraction (micrometers)."""

    frw_num_walks: int = 10000
    """Number of random walks for FRW capacitance solver."""

    frw_tolerance: float = 0.01
    """Target relative tolerance for FRW convergence."""

    resistance_segments_per_lambda: int = 10
    """Number of resistance segments per wavelength for RC accuracy."""

    reduce_network: bool = True
    """Whether to apply model order reduction."""

    reduction_order: int = 50
    """Target order for reduced model."""


@dataclass
class Conductor:
    """A conductor (net) in the layout."""

    name: str
    """Net name."""

    net_id: int
    """Unique numeric identifier."""

    polygons: dict[str, list[NDArray[np.float64]]] = field(default_factory=dict)
    """Polygons per layer. Key is layer name, value is list of Nx2 arrays."""

    pins: list[tuple[str, float, float]] = field(default_factory=list)
    """Pin locations as (layer, x, y) tuples."""


@dataclass
class ExtractionResult:
    """Results of parasitic extraction for a design."""

    cell_name: str
    """Name of the extracted cell."""

    conductors: list[Conductor] = field(default_factory=list)
    """List of conductors in the design."""

    capacitances: dict[tuple[int, int], float] = field(default_factory=dict)
    """Capacitance matrix. Key is (net_id1, net_id2), value is capacitance in Farads.
    Self-capacitance uses (net_id, net_id) or (net_id, -1) for ground."""

    resistances: dict[tuple[int, int, int], float] = field(default_factory=dict)
    """Resistance network. Key is (net_id, node1, node2), value is resistance in Ohms."""

    node_positions: dict[tuple[int, int], tuple[str, float, float, float]] = field(
        default_factory=dict
    )
    """Node positions. Key is (net_id, node_id), value is (layer, x, y, z)."""

    def total_capacitance(self, net_id: int) -> float:
        """Get total capacitance for a net (sum of self and coupling).

        Args:
            net_id: Net identifier.

        Returns:
            Total capacitance in Farads.
        """
        total = 0.0
        for (id1, id2), cap in self.capacitances.items():
            if id1 == net_id or id2 == net_id:
                total += cap
        return total

    def coupling_capacitance(self, net1: int, net2: int) -> float:
        """Get coupling capacitance between two nets.

        Args:
            net1: First net identifier.
            net2: Second net identifier.

        Returns:
            Coupling capacitance in Farads, or 0 if no coupling.
        """
        key = (min(net1, net2), max(net1, net2))
        return self.capacitances.get(key, 0.0)

    def total_resistance(self, net_id: int) -> float:
        """Get total resistance for a net (sum of all segments).

        Args:
            net_id: Net identifier.

        Returns:
            Total resistance in Ohms.
        """
        total = 0.0
        for (nid, _n1, _n2), res in self.resistances.items():
            if nid == net_id:
                total += res
        return total

    def rc_time_constant(self, net_id: int) -> float:
        """Estimate RC time constant for a net.

        Uses Elmore delay approximation: tau = sum(R * downstream_C)

        Args:
            net_id: Net identifier.

        Returns:
            Estimated time constant in seconds.
        """
        # Simplified: R_total * C_total
        # More accurate Elmore delay requires network topology
        return self.total_resistance(net_id) * self.total_capacitance(net_id)


class Extractor(ABC):
    """Abstract base class for parasitic extractors."""

    def __init__(
        self,
        tech_data: TechnologyData,
        config: ExtractionConfig | None = None,
    ):
        """Initialize extractor.

        Args:
            tech_data: Technology data for the target process.
            config: Extraction configuration. Uses defaults if None.
        """
        self.tech_data = tech_data
        self.config = config or ExtractionConfig()

    @abstractmethod
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
        ...

    @abstractmethod
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
        ...


@dataclass
class AnalyticalReference:
    """Reference values for validation."""

    description: str
    """Description of the test case."""

    expected_capacitance_f: float | None = None
    """Expected capacitance in Farads."""

    expected_resistance_ohm: float | None = None
    """Expected resistance in Ohms."""

    expected_rc_ns: float | None = None
    """Expected RC time constant in nanoseconds."""

    tolerance: float = 0.05
    """Acceptable relative error (5% default)."""

    def validate_capacitance(self, measured: float) -> tuple[bool, float]:
        """Validate measured capacitance against expected.

        Args:
            measured: Measured capacitance in Farads.

        Returns:
            Tuple of (passed, relative_error).
        """
        if self.expected_capacitance_f is None:
            return True, 0.0

        error = abs(measured - self.expected_capacitance_f) / self.expected_capacitance_f
        return error <= self.tolerance, error

    def validate_resistance(self, measured: float) -> tuple[bool, float]:
        """Validate measured resistance against expected.

        Args:
            measured: Measured resistance in Ohms.

        Returns:
            Tuple of (passed, relative_error).
        """
        if self.expected_resistance_ohm is None:
            return True, 0.0

        error = abs(measured - self.expected_resistance_ohm) / self.expected_resistance_ohm
        return error <= self.tolerance, error
