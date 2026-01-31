"""Unit tests for simulation module."""

import numpy as np

from jax_extract.extraction.network import RCNetwork
from jax_extract.simulation.jax_spice import (
    AnalyticalSimulator,
    JaxSpiceSimulator,
    SimulationResult,
)
from jax_extract.simulation.spice_netlist import (
    NetlistFormat,
    SimulationSetup,
    SimulationTestbench,
    SpiceNetlistWriter,
    generate_step_response_testbench,
    rc_network_to_spice,
)


class TestSimulationResult:
    """Tests for SimulationResult class."""

    def test_create_result(self):
        """Test creating a simulation result."""
        t = np.linspace(0, 1e-9, 100)
        v = 1.0 - np.exp(-t / 0.1e-9)

        result = SimulationResult(
            time=t,
            voltages={"out": v},
        )

        assert result.time is not None
        assert len(result.time) == 100
        assert "out" in result.voltages

    def test_get_voltage(self):
        """Test getting voltage waveform."""
        t = np.linspace(0, 1e-9, 100)
        v = np.ones(100)

        result = SimulationResult(time=t, voltages={"out": v})

        assert result.get_voltage("out") is not None
        assert result.get_voltage("nonexistent") is None

    def test_measure_time_constant(self):
        """Test measuring RC time constant."""
        tau = 1e-9  # 1ns
        t = np.linspace(0, 10 * tau, 1000)
        v = 1.0 * (1 - np.exp(-t / tau))

        result = SimulationResult(time=t, voltages={"out": v})

        measured_tau = result.measure_time_constant("out")
        assert measured_tau is not None
        # Should be within 5% of actual tau
        assert abs(measured_tau - tau) / tau < 0.05

    def test_measure_delay(self):
        """Test measuring propagation delay."""
        t = np.linspace(0, 10e-9, 1000)
        delay = 2e-9

        # Input step at t=0
        v_in = np.where(t > 0, 1.0, 0.0)

        # Output delayed step
        v_out = np.where(t > delay, 1.0, 0.0)

        result = SimulationResult(
            time=t,
            voltages={"in": v_in, "out": v_out},
        )

        measured_delay = result.measure_delay("in", "out")
        assert measured_delay is not None
        # Should be close to expected delay
        assert abs(measured_delay - delay) < 0.1e-9


class TestSpiceNetlistWriter:
    """Tests for SpiceNetlistWriter class."""

    def test_generate_subcircuit(self):
        """Test generating a subcircuit."""
        network = RCNetwork(name="test")
        n1 = network.add_node("net", is_pin=True, pin_name="in")
        n2 = network.add_node("net")
        n3 = network.add_node("net", is_pin=True, pin_name="out")

        network.add_resistor(n1, n2, 100.0)
        network.add_resistor(n2, n3, 100.0)
        network.add_capacitor(n2, None, 1e-15)

        writer = SpiceNetlistWriter()
        spice = writer.generate_subcircuit(network)

        assert ".subckt test_parasitic" in spice
        assert "R0" in spice
        assert "R1" in spice
        assert "C2" in spice
        assert ".ends" in spice

    def test_generate_testbench(self):
        """Test generating a complete testbench."""
        network = RCNetwork(name="dut")
        n1 = network.add_node("net", is_pin=True, pin_name="in")
        n2 = network.add_node("net", is_pin=True, pin_name="out")
        network.add_resistor(n1, n2, 100.0)
        network.add_capacitor(n2, None, 1e-15)

        config = SimulationTestbench(
            name="rc_test",
            dut_subckt="dut_parasitic",
            input_node="in",
            output_node="out",
        )

        setup = SimulationSetup(
            analysis_type="tran",
            t_stop=100e-9,
        )

        writer = SpiceNetlistWriter()
        spice = writer.generate_testbench(network, config, setup)

        assert "Testbench: rc_test" in spice
        assert ".TRAN" in spice
        assert "VIN" in spice
        assert ".END" in spice

    def test_different_formats(self):
        """Test different netlist formats."""
        network = RCNetwork(name="test")
        n1 = network.add_node("net", is_pin=True, pin_name="a")
        n2 = network.add_node("net", is_pin=True, pin_name="b")
        network.add_resistor(n1, n2, 100.0)

        for fmt in [NetlistFormat.SPICE, NetlistFormat.HSPICE]:
            writer = SpiceNetlistWriter(format=fmt)
            spice = writer.generate_subcircuit(network)
            assert ".subckt" in spice


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_rc_network_to_spice(self):
        """Test rc_network_to_spice function."""
        network = RCNetwork(name="simple")
        n1 = network.add_node("net", is_pin=True, pin_name="p1")
        n2 = network.add_node("net", is_pin=True, pin_name="p2")
        network.add_resistor(n1, n2, 50.0)

        spice = rc_network_to_spice(network)

        assert ".subckt simple_parasitic" in spice
        assert "R0 p1 p2 50" in spice

    def test_generate_step_response_testbench(self):
        """Test generate_step_response_testbench function."""
        network = RCNetwork(name="rc_ladder")
        n1 = network.add_node("net", is_pin=True, pin_name="in")
        n2 = network.add_node("net", is_pin=True, pin_name="out")
        network.add_resistor(n1, n2, 100.0)
        network.add_capacitor(n2, None, 10e-15)

        spice = generate_step_response_testbench(network, t_stop=50e-9)

        assert ".TRAN" in spice
        assert "50" in spice or "5e-08" in spice.lower()


class TestAnalyticalSimulator:
    """Tests for AnalyticalSimulator class."""

    def test_rc_step_response(self):
        """Test analytical RC step response."""
        sim = AnalyticalSimulator()

        tau = 1e-9  # 1ns
        r = 1000  # 1k ohm
        c = tau / r  # 1fF

        result = sim.simulate_rc_step_response(
            r_total=r,
            c_total=c,
            t_stop=10 * tau,
            v_step=1.0,
        )

        assert result.time is not None
        assert result.voltages is not None
        assert "out" in result.voltages

        # Check final value approaches v_step
        assert abs(result.voltages["out"][-1] - 1.0) < 0.01

        # Check time constant
        measured_tau = result.measure_time_constant("out")
        assert measured_tau is not None
        assert abs(measured_tau - tau) / tau < 0.05

    def test_elmore_delay_simulation(self):
        """Test Elmore delay-based simulation."""
        network = RCNetwork(name="ladder")
        n1 = network.add_node("net", is_pin=True, pin_name="in")
        n2 = network.add_node("net")
        n3 = network.add_node("net", is_pin=True, pin_name="out")

        # Use larger values for measurable time constant
        # R = 10k ohm, C = 10fF each -> tau = 10k * 20fF = 200ps
        network.add_resistor(n1, n2, 10000.0)
        network.add_resistor(n2, n3, 10000.0)
        network.add_capacitor(n2, None, 10e-15)
        network.add_capacitor(n3, None, 10e-15)

        sim = AnalyticalSimulator()
        # tau = R_total * C_total = 20k * 20fF = 400ps
        expected_tau = 20000.0 * 20e-15  # 400ps

        # Simulate for 10*tau with enough points
        result = sim.simulate_elmore_delay(
            network,
            t_stop=10 * expected_tau,
            n_points=10000,
        )

        assert result.time is not None
        assert result.voltages is not None

        measured_tau = result.measure_time_constant("out")

        assert measured_tau is not None
        assert abs(measured_tau - expected_tau) / expected_tau < 0.1


class TestJaxSpiceSimulator:
    """Tests for JaxSpiceSimulator class."""

    def test_availability_check(self):
        """Test jax-spice availability check."""
        sim = JaxSpiceSimulator()
        # Should return False since jax-spice is not installed
        # (or True if it happens to be installed)
        assert isinstance(sim.is_available(), bool)

    def test_simulate_without_jax_spice(self):
        """Test that simulation raises ImportError without jax-spice."""
        sim = JaxSpiceSimulator()

        if not sim.is_available():
            network = RCNetwork(name="test")

            try:
                sim.simulate_step_response(network)
                raise AssertionError("Should have raised ImportError")
            except ImportError as e:
                assert "jax-spice" in str(e)
