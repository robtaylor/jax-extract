"""Resistance extraction from wire geometry.

This module handles extraction of wire resistance and via resistance
from layout geometry.
"""

from dataclasses import dataclass, field

import numpy as np

from jax_extract.extraction.geometry import Polygon
from jax_extract.extraction.network import Node, RCNetwork
from jax_extract.pdk.adapter import TechnologyData


@dataclass
class WireSegment:
    """A segment of wire for resistance calculation."""

    layer: str
    """Layer name."""

    start: tuple[float, float]
    """Start point (x, y) in micrometers."""

    end: tuple[float, float]
    """End point (x, y) in micrometers."""

    width: float
    """Wire width in micrometers."""

    def length(self) -> float:
        """Calculate segment length in micrometers."""
        dx = self.end[0] - self.start[0]
        dy = self.end[1] - self.start[1]
        return float(np.sqrt(dx * dx + dy * dy))

    def resistance(self, sheet_r: float) -> float:
        """Calculate segment resistance.

        Args:
            sheet_r: Sheet resistance in Ohms/square.

        Returns:
            Resistance in Ohms.
        """
        length = self.length()
        if self.width <= 0 or length <= 0:
            return 0.0
        return sheet_r * length / self.width

    def midpoint(self) -> tuple[float, float]:
        """Get segment midpoint."""
        return (
            (self.start[0] + self.end[0]) / 2,
            (self.start[1] + self.end[1]) / 2,
        )


@dataclass
class ViaInstance:
    """An instance of a via connecting layers."""

    via_type: str
    """Via type name."""

    x: float
    """X position in micrometers."""

    y: float
    """Y position in micrometers."""

    bottom_layer: str
    """Bottom metal layer."""

    top_layer: str
    """Top metal layer."""

    count: int = 1
    """Number of via cuts (for via arrays)."""

    def resistance(self, single_via_r: float) -> float:
        """Calculate via resistance.

        Args:
            single_via_r: Single via resistance in Ohms.

        Returns:
            Effective resistance for via array (parallel).
        """
        if self.count <= 0:
            return float("inf")
        return single_via_r / self.count


@dataclass
class ResistanceNetwork:
    """Resistance network for a single net."""

    net_name: str
    """Name of the net."""

    segments: list[WireSegment] = field(default_factory=list)
    """Wire segments."""

    vias: list[ViaInstance] = field(default_factory=list)
    """Via instances."""

    def total_wire_resistance(self, tech_data: TechnologyData) -> float:
        """Calculate total wire resistance.

        Args:
            tech_data: Technology data for sheet resistance.

        Returns:
            Total wire resistance in Ohms.
        """
        total = 0.0
        for seg in self.segments:
            layer = tech_data.get_layer(seg.layer)
            if layer:
                total += seg.resistance(layer.sheet_resistance_ohm_sq)
        return total

    def total_via_resistance(self, tech_data: TechnologyData) -> float:
        """Calculate total via resistance.

        Args:
            tech_data: Technology data for via resistance.

        Returns:
            Total via resistance in Ohms.
        """
        total = 0.0
        for via in self.vias:
            via_def = tech_data.get_via(via.via_type)
            if via_def:
                total += via.resistance(via_def.resistance_ohm)
        return total


class ResistanceExtractor:
    """Extract resistance from layout geometry."""

    def __init__(self, tech_data: TechnologyData):
        """Initialize resistance extractor.

        Args:
            tech_data: Technology data for layer properties.
        """
        self.tech_data = tech_data

    def extract_from_polygon(
        self,
        polygon: Polygon,
        segments_per_lambda: int = 10,
    ) -> list[WireSegment]:
        """Extract wire segments from a polygon.

        Approximates the polygon as a series of rectangular segments.

        Args:
            polygon: Input polygon.
            segments_per_lambda: Segments per minimum width (for accuracy).

        Returns:
            List of WireSegment objects.
        """
        # Get layer properties
        layer_info = self.tech_data.get_layer(polygon.layer)
        if not layer_info:
            return []

        min_width = layer_info.min_width_nm / 1000  # Convert to um

        # Compute bounding box
        xmin, ymin, xmax, ymax = polygon.bounding_box()
        width = xmax - xmin
        height = ymax - ymin

        # Determine wire orientation (longer dimension is length)
        if width >= height:
            # Horizontal wire
            wire_width = height
            wire_length = width

            # Number of segments
            if min_width > 0:
                n_segments = max(1, int(wire_length / min_width * segments_per_lambda / 10))
            else:
                n_segments = max(1, int(wire_length / wire_width))

            segment_length = wire_length / n_segments
            y_center = (ymin + ymax) / 2

            segments = []
            for i in range(n_segments):
                x_start = xmin + i * segment_length
                x_end = x_start + segment_length
                segments.append(
                    WireSegment(
                        layer=polygon.layer,
                        start=(x_start, y_center),
                        end=(x_end, y_center),
                        width=wire_width,
                    )
                )
            return segments
        else:
            # Vertical wire
            wire_width = width
            wire_length = height

            if min_width > 0:
                n_segments = max(1, int(wire_length / min_width * segments_per_lambda / 10))
            else:
                n_segments = max(1, int(wire_length / wire_width))

            segment_length = wire_length / n_segments
            x_center = (xmin + xmax) / 2

            segments = []
            for i in range(n_segments):
                y_start = ymin + i * segment_length
                y_end = y_start + segment_length
                segments.append(
                    WireSegment(
                        layer=polygon.layer,
                        start=(x_center, y_start),
                        end=(x_center, y_end),
                        width=wire_width,
                    )
                )
            return segments

    def extract_from_centerline(
        self,
        centerline: list[tuple[float, float]],
        width: float,
        layer: str,
    ) -> list[WireSegment]:
        """Extract wire segments from a centerline path.

        Args:
            centerline: List of (x, y) points defining the path.
            width: Wire width in micrometers.
            layer: Layer name.

        Returns:
            List of WireSegment objects.
        """
        segments = []

        for i in range(len(centerline) - 1):
            segments.append(
                WireSegment(
                    layer=layer,
                    start=centerline[i],
                    end=centerline[i + 1],
                    width=width,
                )
            )

        return segments

    def build_rc_network(
        self,
        net_name: str,
        segments: list[WireSegment],
        vias: list[ViaInstance],
    ) -> RCNetwork:
        """Build RC network from wire segments and vias.

        Args:
            net_name: Name of the net.
            segments: Wire segments.
            vias: Via instances.

        Returns:
            RCNetwork with resistance values.
        """
        network = RCNetwork(name=net_name)

        # Create nodes at segment endpoints
        node_map: dict[tuple[str, float, float], Node] = {}

        def get_or_create_node(layer: str, x: float, y: float) -> Node:
            """Get existing node or create new one at position."""
            # Round to avoid floating point comparison issues
            key = (layer, round(x, 3), round(y, 3))
            if key not in node_map:
                layer_info = self.tech_data.get_layer(layer)
                z = layer_info.height_nm / 1000 if layer_info else 0.0
                node = network.add_node(
                    net_name=net_name,
                    layer=layer,
                    x=x,
                    y=y,
                    z=z,
                )
                node_map[key] = node
            return node_map[key]

        # Add wire segments as resistors
        for seg in segments:
            layer_info = self.tech_data.get_layer(seg.layer)
            if not layer_info:
                continue

            node1 = get_or_create_node(seg.layer, seg.start[0], seg.start[1])
            node2 = get_or_create_node(seg.layer, seg.end[0], seg.end[1])

            resistance = seg.resistance(layer_info.sheet_resistance_ohm_sq)
            if resistance > 0:
                network.add_resistor(
                    node1=node1,
                    node2=node2,
                    resistance_ohm=resistance,
                    layer=seg.layer,
                )

        # Add vias as resistors
        for via in vias:
            via_def = self.tech_data.get_via(via.via_type)
            if not via_def:
                continue

            # Find or create nodes on both layers at via position
            node_bot = get_or_create_node(via.bottom_layer, via.x, via.y)
            node_top = get_or_create_node(via.top_layer, via.x, via.y)

            resistance = via.resistance(via_def.resistance_ohm)
            if resistance > 0 and resistance != float("inf"):
                network.add_resistor(
                    node1=node_bot,
                    node2=node_top,
                    resistance_ohm=resistance,
                    is_via=True,
                )

        return network


def calculate_wire_resistance(
    length_um: float,
    width_um: float,
    sheet_resistance: float,
) -> float:
    """Calculate wire resistance from dimensions.

    Args:
        length_um: Wire length in micrometers.
        width_um: Wire width in micrometers.
        sheet_resistance: Sheet resistance in Ohms/square.

    Returns:
        Resistance in Ohms.
    """
    if width_um <= 0:
        return float("inf")
    return sheet_resistance * length_um / width_um


def calculate_serpentine_resistance(
    width_um: float,
    spacing_um: float,
    turns: int,
    turn_length_um: float,
    sheet_resistance: float,
) -> float:
    """Calculate resistance of a serpentine test structure.

    Args:
        width_um: Wire width in micrometers.
        spacing_um: Spacing between runs in micrometers.
        turns: Number of turns.
        turn_length_um: Length of each horizontal run in micrometers.
        sheet_resistance: Sheet resistance in Ohms/square.

    Returns:
        Total resistance in Ohms.
    """
    # Total length = horizontal runs + vertical connectors
    n_runs = turns + 1
    horizontal_length = n_runs * turn_length_um
    vertical_length = turns * (width_um + spacing_um)
    total_length = horizontal_length + vertical_length

    return calculate_wire_resistance(total_length, width_um, sheet_resistance)


def elmore_delay(
    resistors: list[tuple[float, float]],
    capacitances: list[float],
) -> float:
    """Calculate Elmore delay for an RC ladder.

    Args:
        resistors: List of (resistance, downstream_capacitance) pairs.
        capacitances: List of node capacitances (not used in simple form).

    Returns:
        Elmore delay in seconds.
    """
    delay = 0.0
    for r, c_downstream in resistors:
        delay += r * c_downstream
    return delay
