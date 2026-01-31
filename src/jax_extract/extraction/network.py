"""RC/RLC network representation and manipulation.

This module provides data structures for representing extracted parasitic
networks and utilities for network manipulation and SPICE output.
"""

from dataclasses import dataclass, field
from io import StringIO

import numpy as np
from numpy.typing import NDArray


@dataclass
class Node:
    """A node in the RC network."""

    node_id: int
    """Unique node identifier within the net."""

    net_name: str
    """Name of the parent net."""

    layer: str = ""
    """Layer where this node is located."""

    x: float = 0.0
    """X coordinate in micrometers."""

    y: float = 0.0
    """Y coordinate in micrometers."""

    z: float = 0.0
    """Z coordinate (height) in micrometers."""

    is_pin: bool = False
    """Whether this node is a pin (external connection point)."""

    pin_name: str = ""
    """Pin name if this is a pin node."""

    @property
    def spice_name(self) -> str:
        """Generate SPICE node name."""
        if self.is_pin and self.pin_name:
            return self.pin_name
        return f"{self.net_name}_n{self.node_id}"


@dataclass
class Resistor:
    """A resistor element in the network."""

    element_id: int
    """Unique element identifier."""

    node1: Node
    """First terminal node."""

    node2: Node
    """Second terminal node."""

    resistance_ohm: float
    """Resistance value in Ohms."""

    layer: str = ""
    """Layer where this resistor is located (for wire resistors)."""

    is_via: bool = False
    """Whether this is a via resistance."""

    def to_spice(self) -> str:
        """Generate SPICE resistor statement."""
        return f"R{self.element_id} {self.node1.spice_name} {self.node2.spice_name} {self.resistance_ohm:.6g}"


@dataclass
class Capacitor:
    """A capacitor element in the network."""

    element_id: int
    """Unique element identifier."""

    node1: Node
    """First terminal node (or None for ground capacitance)."""

    node2: Node | None
    """Second terminal node (None for ground capacitance)."""

    capacitance_f: float
    """Capacitance value in Farads."""

    is_coupling: bool = False
    """Whether this is a coupling capacitance between nets."""

    def to_spice(self) -> str:
        """Generate SPICE capacitor statement."""
        node2_name = self.node2.spice_name if self.node2 else "0"
        # Express capacitance in femtofarads for readability
        cap_ff = self.capacitance_f * 1e15
        return f"C{self.element_id} {self.node1.spice_name} {node2_name} {cap_ff:.6g}f"


@dataclass
class Inductor:
    """An inductor element in the network (for RLC extraction)."""

    element_id: int
    """Unique element identifier."""

    node1: Node
    """First terminal node."""

    node2: Node
    """Second terminal node."""

    inductance_h: float
    """Inductance value in Henries."""

    def to_spice(self) -> str:
        """Generate SPICE inductor statement."""
        # Express inductance in picohenries for readability
        ind_ph = self.inductance_h * 1e12
        return f"L{self.element_id} {self.node1.spice_name} {self.node2.spice_name} {ind_ph:.6g}p"


@dataclass
class RCNetwork:
    """Complete RC/RLC network for a design or net."""

    name: str
    """Network name (typically cell or net name)."""

    nodes: list[Node] = field(default_factory=list)
    """All nodes in the network."""

    resistors: list[Resistor] = field(default_factory=list)
    """All resistors in the network."""

    capacitors: list[Capacitor] = field(default_factory=list)
    """All capacitors in the network."""

    inductors: list[Inductor] = field(default_factory=list)
    """All inductors in the network (empty for RC mode)."""

    _node_counter: int = field(default=0, repr=False)
    _element_counter: int = field(default=0, repr=False)

    def add_node(
        self,
        net_name: str,
        layer: str = "",
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
        is_pin: bool = False,
        pin_name: str = "",
    ) -> Node:
        """Create and add a new node to the network.

        Args:
            net_name: Name of the parent net.
            layer: Layer name.
            x: X coordinate in micrometers.
            y: Y coordinate in micrometers.
            z: Z coordinate in micrometers.
            is_pin: Whether this is a pin node.
            pin_name: Pin name if applicable.

        Returns:
            The newly created Node.
        """
        node = Node(
            node_id=self._node_counter,
            net_name=net_name,
            layer=layer,
            x=x,
            y=y,
            z=z,
            is_pin=is_pin,
            pin_name=pin_name,
        )
        self._node_counter += 1
        self.nodes.append(node)
        return node

    def add_resistor(
        self,
        node1: Node,
        node2: Node,
        resistance_ohm: float,
        layer: str = "",
        is_via: bool = False,
    ) -> Resistor:
        """Create and add a resistor to the network.

        Args:
            node1: First terminal node.
            node2: Second terminal node.
            resistance_ohm: Resistance value in Ohms.
            layer: Layer name.
            is_via: Whether this is a via resistance.

        Returns:
            The newly created Resistor.
        """
        resistor = Resistor(
            element_id=self._element_counter,
            node1=node1,
            node2=node2,
            resistance_ohm=resistance_ohm,
            layer=layer,
            is_via=is_via,
        )
        self._element_counter += 1
        self.resistors.append(resistor)
        return resistor

    def add_capacitor(
        self,
        node1: Node,
        node2: Node | None,
        capacitance_f: float,
        is_coupling: bool = False,
    ) -> Capacitor:
        """Create and add a capacitor to the network.

        Args:
            node1: First terminal node.
            node2: Second terminal node (None for ground cap).
            capacitance_f: Capacitance value in Farads.
            is_coupling: Whether this is inter-net coupling.

        Returns:
            The newly created Capacitor.
        """
        capacitor = Capacitor(
            element_id=self._element_counter,
            node1=node1,
            node2=node2,
            capacitance_f=capacitance_f,
            is_coupling=is_coupling,
        )
        self._element_counter += 1
        self.capacitors.append(capacitor)
        return capacitor

    def add_inductor(
        self,
        node1: Node,
        node2: Node,
        inductance_h: float,
    ) -> Inductor:
        """Create and add an inductor to the network.

        Args:
            node1: First terminal node.
            node2: Second terminal node.
            inductance_h: Inductance value in Henries.

        Returns:
            The newly created Inductor.
        """
        inductor = Inductor(
            element_id=self._element_counter,
            node1=node1,
            node2=node2,
            inductance_h=inductance_h,
        )
        self._element_counter += 1
        self.inductors.append(inductor)
        return inductor

    def get_pins(self) -> list[Node]:
        """Get all pin nodes.

        Returns:
            List of nodes marked as pins.
        """
        return [n for n in self.nodes if n.is_pin]

    def total_capacitance(self) -> float:
        """Calculate total capacitance in the network.

        Returns:
            Sum of all capacitances in Farads.
        """
        return sum(c.capacitance_f for c in self.capacitors)

    def total_resistance(self) -> float:
        """Calculate total resistance in the network.

        Returns:
            Sum of all resistances in Ohms.
        """
        return sum(r.resistance_ohm for r in self.resistors)

    def to_spice_subcircuit(self) -> str:
        """Generate SPICE subcircuit for this network.

        Returns:
            SPICE subcircuit definition as string.
        """
        buf = StringIO()

        # Get pin names for subcircuit ports
        pins = self.get_pins()
        pin_names = " ".join(p.spice_name for p in pins) if pins else "in out"

        buf.write(f"* Extracted parasitics for {self.name}\n")
        buf.write(f".subckt {self.name}_parasitic {pin_names}\n\n")

        # Resistors
        if self.resistors:
            buf.write("* Resistors\n")
            for r in self.resistors:
                buf.write(f"{r.to_spice()}\n")
            buf.write("\n")

        # Capacitors
        if self.capacitors:
            buf.write("* Capacitors\n")
            for c in self.capacitors:
                buf.write(f"{c.to_spice()}\n")
            buf.write("\n")

        # Inductors
        if self.inductors:
            buf.write("* Inductors\n")
            for ind in self.inductors:
                buf.write(f"{ind.to_spice()}\n")
            buf.write("\n")

        buf.write(f".ends {self.name}_parasitic\n")

        return buf.getvalue()

    def to_spice_flat(self) -> str:
        """Generate flat SPICE netlist (no subcircuit wrapper).

        Returns:
            SPICE element statements as string.
        """
        buf = StringIO()

        buf.write(f"* Extracted parasitics for {self.name}\n\n")

        for r in self.resistors:
            buf.write(f"{r.to_spice()}\n")

        for c in self.capacitors:
            buf.write(f"{c.to_spice()}\n")

        for ind in self.inductors:
            buf.write(f"{ind.to_spice()}\n")

        return buf.getvalue()

    def get_admittance_matrices(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Build G and C matrices for the network.

        For the system: G*v + C*dv/dt = I
        (Modified nodal analysis form)

        Returns:
            Tuple of (G, C) matrices as numpy arrays.
        """
        n = len(self.nodes)
        G = np.zeros((n, n), dtype=np.float64)
        C = np.zeros((n, n), dtype=np.float64)

        # Build node index map
        node_idx = {node.node_id: i for i, node in enumerate(self.nodes)}

        # Add resistor stamps
        for r in self.resistors:
            i = node_idx[r.node1.node_id]
            j = node_idx[r.node2.node_id]
            g = 1.0 / r.resistance_ohm

            G[i, i] += g
            G[j, j] += g
            G[i, j] -= g
            G[j, i] -= g

        # Add capacitor stamps
        for cap in self.capacitors:
            i = node_idx[cap.node1.node_id]
            c = cap.capacitance_f

            if cap.node2 is not None:
                j = node_idx[cap.node2.node_id]
                C[i, i] += c
                C[j, j] += c
                C[i, j] -= c
                C[j, i] -= c
            else:
                # Ground capacitance
                C[i, i] += c

        return G, C


def merge_networks(networks: list[RCNetwork], name: str = "merged") -> RCNetwork:
    """Merge multiple RC networks into one.

    Args:
        networks: List of networks to merge.
        name: Name for the merged network.

    Returns:
        New RCNetwork containing all elements from input networks.
    """
    merged = RCNetwork(name=name)

    # Track node ID mapping to avoid collisions
    node_map: dict[int, Node] = {}

    for net in networks:
        # Remap and add nodes
        for old_node in net.nodes:
            new_node = merged.add_node(
                net_name=old_node.net_name,
                layer=old_node.layer,
                x=old_node.x,
                y=old_node.y,
                z=old_node.z,
                is_pin=old_node.is_pin,
                pin_name=old_node.pin_name,
            )
            node_map[old_node.node_id] = new_node

        # Remap and add resistors
        for r in net.resistors:
            merged.add_resistor(
                node1=node_map[r.node1.node_id],
                node2=node_map[r.node2.node_id],
                resistance_ohm=r.resistance_ohm,
                layer=r.layer,
                is_via=r.is_via,
            )

        # Remap and add capacitors
        for c in net.capacitors:
            node2 = node_map[c.node2.node_id] if c.node2 else None
            merged.add_capacitor(
                node1=node_map[c.node1.node_id],
                node2=node2,
                capacitance_f=c.capacitance_f,
                is_coupling=c.is_coupling,
            )

        # Remap and add inductors
        for ind in net.inductors:
            merged.add_inductor(
                node1=node_map[ind.node1.node_id],
                node2=node_map[ind.node2.node_id],
                inductance_h=ind.inductance_h,
            )

    return merged
