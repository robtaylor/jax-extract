"""Unit tests for RC network module."""


from jax_extract.extraction.network import (
    Capacitor,
    Node,
    RCNetwork,
    Resistor,
    merge_networks,
)


class TestNode:
    """Tests for Node class."""

    def test_create_node(self):
        """Test creating a node."""
        node = Node(
            node_id=0,
            net_name="VDD",
            layer="Metal1",
            x=10.0,
            y=20.0,
            z=0.45,
        )

        assert node.node_id == 0
        assert node.net_name == "VDD"
        assert node.x == 10.0
        assert node.y == 20.0

    def test_spice_name(self):
        """Test SPICE node name generation."""
        node = Node(node_id=5, net_name="net1")
        assert node.spice_name == "net1_n5"

        pin_node = Node(node_id=0, net_name="VDD", is_pin=True, pin_name="VDD")
        assert pin_node.spice_name == "VDD"


class TestResistor:
    """Tests for Resistor class."""

    def test_create_resistor(self):
        """Test creating a resistor."""
        n1 = Node(node_id=0, net_name="net")
        n2 = Node(node_id=1, net_name="net")

        r = Resistor(
            element_id=0,
            node1=n1,
            node2=n2,
            resistance_ohm=100.0,
            layer="Metal1",
        )

        assert r.resistance_ohm == 100.0
        assert r.layer == "Metal1"
        assert not r.is_via

    def test_to_spice(self):
        """Test SPICE output."""
        n1 = Node(node_id=0, net_name="net")
        n2 = Node(node_id=1, net_name="net")
        r = Resistor(element_id=0, node1=n1, node2=n2, resistance_ohm=150.5)

        spice = r.to_spice()
        assert spice == "R0 net_n0 net_n1 150.5"


class TestCapacitor:
    """Tests for Capacitor class."""

    def test_ground_capacitor(self):
        """Test ground capacitor (single-ended)."""
        n1 = Node(node_id=0, net_name="net")
        c = Capacitor(
            element_id=0,
            node1=n1,
            node2=None,
            capacitance_f=1e-15,
        )

        assert c.capacitance_f == 1e-15
        assert c.node2 is None
        assert not c.is_coupling

    def test_coupling_capacitor(self):
        """Test coupling capacitor (between nets)."""
        n1 = Node(node_id=0, net_name="net1")
        n2 = Node(node_id=0, net_name="net2")
        c = Capacitor(
            element_id=0,
            node1=n1,
            node2=n2,
            capacitance_f=0.5e-15,
            is_coupling=True,
        )

        assert c.is_coupling
        assert c.node2 is not None

    def test_to_spice(self):
        """Test SPICE output."""
        n1 = Node(node_id=0, net_name="net")

        # Ground cap
        c = Capacitor(element_id=0, node1=n1, node2=None, capacitance_f=1e-15)
        spice = c.to_spice()
        assert "C0 net_n0 0 1f" in spice

        # Coupling cap
        n2 = Node(node_id=1, net_name="net")
        c2 = Capacitor(element_id=1, node1=n1, node2=n2, capacitance_f=0.5e-15)
        spice2 = c2.to_spice()
        assert "C1 net_n0 net_n1 0.5f" in spice2


class TestRCNetwork:
    """Tests for RCNetwork class."""

    def test_create_network(self):
        """Test creating an empty network."""
        net = RCNetwork(name="test")
        assert net.name == "test"
        assert len(net.nodes) == 0
        assert len(net.resistors) == 0
        assert len(net.capacitors) == 0

    def test_add_node(self):
        """Test adding nodes."""
        net = RCNetwork(name="test")

        n1 = net.add_node(net_name="net1", layer="Metal1", x=0.0, y=0.0)
        n2 = net.add_node(net_name="net1", layer="Metal1", x=10.0, y=0.0)

        assert len(net.nodes) == 2
        assert n1.node_id == 0
        assert n2.node_id == 1

    def test_add_resistor(self):
        """Test adding resistors."""
        net = RCNetwork(name="test")
        n1 = net.add_node(net_name="net1")
        n2 = net.add_node(net_name="net1")

        r = net.add_resistor(n1, n2, 100.0, layer="Metal1")

        assert len(net.resistors) == 1
        assert r.resistance_ohm == 100.0

    def test_add_capacitor(self):
        """Test adding capacitors."""
        net = RCNetwork(name="test")
        n1 = net.add_node(net_name="net1")

        c = net.add_capacitor(n1, None, 1e-15)

        assert len(net.capacitors) == 1
        assert c.capacitance_f == 1e-15

    def test_get_pins(self):
        """Test getting pin nodes."""
        net = RCNetwork(name="test")
        net.add_node(net_name="net", is_pin=True, pin_name="A")
        net.add_node(net_name="net")
        net.add_node(net_name="net", is_pin=True, pin_name="B")

        pins = net.get_pins()
        assert len(pins) == 2
        assert pins[0].pin_name == "A"
        assert pins[1].pin_name == "B"

    def test_total_capacitance(self):
        """Test total capacitance calculation."""
        net = RCNetwork(name="test")
        n1 = net.add_node(net_name="net")
        n2 = net.add_node(net_name="net")

        net.add_capacitor(n1, None, 1e-15)
        net.add_capacitor(n2, None, 2e-15)
        net.add_capacitor(n1, n2, 0.5e-15)

        total = net.total_capacitance()
        assert abs(total - 3.5e-15) < 1e-20

    def test_total_resistance(self):
        """Test total resistance calculation."""
        net = RCNetwork(name="test")
        n1 = net.add_node(net_name="net")
        n2 = net.add_node(net_name="net")
        n3 = net.add_node(net_name="net")

        net.add_resistor(n1, n2, 100.0)
        net.add_resistor(n2, n3, 200.0)

        total = net.total_resistance()
        assert abs(total - 300.0) < 1e-10

    def test_to_spice_subcircuit(self):
        """Test SPICE subcircuit generation."""
        net = RCNetwork(name="test_net")
        n1 = net.add_node(net_name="net", is_pin=True, pin_name="in")
        n2 = net.add_node(net_name="net")
        n3 = net.add_node(net_name="net", is_pin=True, pin_name="out")

        net.add_resistor(n1, n2, 100.0)
        net.add_resistor(n2, n3, 100.0)
        net.add_capacitor(n2, None, 1e-15)

        spice = net.to_spice_subcircuit()

        assert ".subckt test_net_parasitic in out" in spice
        assert "R0 in net_n1 100" in spice
        assert "R1 net_n1 out 100" in spice
        assert "C2 net_n1 0 1f" in spice
        assert ".ends test_net_parasitic" in spice

    def test_admittance_matrices(self):
        """Test G and C matrix construction."""
        net = RCNetwork(name="test")
        n1 = net.add_node(net_name="net")
        n2 = net.add_node(net_name="net")

        net.add_resistor(n1, n2, 100.0)  # g = 0.01
        net.add_capacitor(n1, None, 1e-15)
        net.add_capacitor(n2, None, 2e-15)

        G, C = net.get_admittance_matrices()

        assert G.shape == (2, 2)
        assert C.shape == (2, 2)

        # Check G matrix (conductance)
        assert abs(G[0, 0] - 0.01) < 1e-10
        assert abs(G[1, 1] - 0.01) < 1e-10
        assert abs(G[0, 1] - (-0.01)) < 1e-10

        # Check C matrix
        assert abs(C[0, 0] - 1e-15) < 1e-20
        assert abs(C[1, 1] - 2e-15) < 1e-20


class TestMergeNetworks:
    """Tests for network merging."""

    def test_merge_two_networks(self):
        """Test merging two networks."""
        net1 = RCNetwork(name="net1")
        n1a = net1.add_node(net_name="net1")
        n1b = net1.add_node(net_name="net1")
        net1.add_resistor(n1a, n1b, 100.0)

        net2 = RCNetwork(name="net2")
        n2a = net2.add_node(net_name="net2")
        n2b = net2.add_node(net_name="net2")
        net2.add_resistor(n2a, n2b, 200.0)
        net2.add_capacitor(n2a, None, 1e-15)

        merged = merge_networks([net1, net2], name="merged")

        assert merged.name == "merged"
        assert len(merged.nodes) == 4
        assert len(merged.resistors) == 2
        assert len(merged.capacitors) == 1
