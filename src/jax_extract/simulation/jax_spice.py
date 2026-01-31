"""jax-spice integration for parasitic validation.

This module provides an interface to jax-spice for simulating
extracted parasitic networks and validating extraction accuracy.
"""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from jax_extract.extraction.network import RCNetwork


@runtime_checkable
class SpiceEngine(Protocol):
    """Protocol for SPICE simulation engines.

    This protocol defines the interface that any SPICE engine
    must implement to be used with jax-extract.
    """

    def tran(
        self,
        t_stop: float,
        t_step: float,
    ) -> "SimulationResult":
        """Run transient analysis.

        Args:
            t_stop: Stop time in seconds.
            t_step: Time step in seconds.

        Returns:
            SimulationResult with time and voltage waveforms.
        """
        ...

    def ac(
        self,
        f_start: float,
        f_stop: float,
        points_per_decade: int,
    ) -> "SimulationResult":
        """Run AC analysis.

        Args:
            f_start: Start frequency in Hz.
            f_stop: Stop frequency in Hz.
            points_per_decade: Number of frequency points per decade.

        Returns:
            SimulationResult with frequency response.
        """
        ...


@dataclass
class SimulationResult:
    """Results from a SPICE simulation."""

    time: NDArray[np.float64] | None = None
    """Time points for transient analysis (seconds)."""

    frequency: NDArray[np.float64] | None = None
    """Frequency points for AC analysis (Hz)."""

    voltages: dict[str, NDArray[np.float64]] | None = None
    """Node voltages keyed by node name."""

    currents: dict[str, NDArray[np.float64]] | None = None
    """Branch currents keyed by element name."""

    def get_voltage(self, node: str) -> NDArray[np.float64] | None:
        """Get voltage waveform for a node.

        Args:
            node: Node name.

        Returns:
            Voltage array or None if not found.
        """
        if self.voltages is None:
            return None
        return self.voltages.get(node)

    def measure_delay(
        self,
        input_node: str,
        output_node: str,
        threshold: float = 0.5,
    ) -> float | None:
        """Measure propagation delay between nodes.

        Args:
            input_node: Input node name.
            output_node: Output node name.
            threshold: Threshold as fraction of swing (0.5 = 50%).

        Returns:
            Delay in seconds, or None if measurement fails.
        """
        if self.time is None or self.voltages is None:
            return None

        v_in = self.voltages.get(input_node)
        v_out = self.voltages.get(output_node)

        if v_in is None or v_out is None:
            return None

        # Find threshold crossings
        v_in_max = np.max(v_in)
        v_in_min = np.min(v_in)
        v_in_th = v_in_min + threshold * (v_in_max - v_in_min)

        v_out_max = np.max(v_out)
        v_out_min = np.min(v_out)
        v_out_th = v_out_min + threshold * (v_out_max - v_out_min)

        # Find first rising edge crossing for input
        in_crossings = np.where(np.diff(np.sign(v_in - v_in_th)) > 0)[0]
        if len(in_crossings) == 0:
            return None
        t_in = self.time[in_crossings[0]]

        # Find first rising edge crossing for output
        out_crossings = np.where(np.diff(np.sign(v_out - v_out_th)) > 0)[0]
        if len(out_crossings) == 0:
            return None
        t_out = self.time[out_crossings[0]]

        return float(t_out - t_in)

    def measure_time_constant(
        self,
        node: str,
        target_fraction: float = 0.632,
    ) -> float | None:
        """Measure RC time constant from step response.

        For a first-order RC system, the time constant tau is the
        time to reach 63.2% (1 - 1/e) of the final value.

        Args:
            node: Node to measure.
            target_fraction: Fraction of final value (default 1-1/e).

        Returns:
            Time constant in seconds, or None if measurement fails.
        """
        if self.time is None or self.voltages is None:
            return None

        v = self.voltages.get(node)
        if v is None:
            return None

        v_initial = v[0]
        v_final = v[-1]

        if abs(v_final - v_initial) < 1e-12:
            return None

        v_target = v_initial + target_fraction * (v_final - v_initial)

        # Find crossing
        if v_final > v_initial:
            crossings = np.where(v >= v_target)[0]
        else:
            crossings = np.where(v <= v_target)[0]

        if len(crossings) == 0:
            return None

        return float(self.time[crossings[0]])


class JaxSpiceSimulator:
    """Simulator using jax-spice for RC network validation.

    This class wraps jax-spice to simulate extracted parasitic
    networks and measure their electrical characteristics.

    Example:
        >>> sim = JaxSpiceSimulator()
        >>> if sim.is_available():
        ...     result = sim.simulate_step_response(network)
        ...     tau = result.measure_time_constant("out")
    """

    def __init__(self):
        """Initialize jax-spice simulator."""
        self._engine = None
        self._available = self._check_availability()

    def _check_availability(self) -> bool:
        """Check if jax-spice is available."""
        try:
            import jax_spice  # noqa: F401

            return True
        except ImportError:
            return False

    def is_available(self) -> bool:
        """Check if jax-spice is available for simulation.

        Returns:
            True if jax-spice can be used.
        """
        return self._available

    def simulate_step_response(
        self,
        network: RCNetwork,
        t_stop: float = 100e-9,
        t_step: float = 0.1e-9,
        v_step: float = 1.0,
    ) -> SimulationResult:
        """Simulate step response of an RC network.

        Args:
            network: Extracted RC network to simulate.
            t_stop: Simulation stop time in seconds.
            t_step: Time step in seconds.
            v_step: Step voltage amplitude.

        Returns:
            SimulationResult with transient waveforms.

        Raises:
            ImportError: If jax-spice is not available.
        """
        if not self._available:
            raise ImportError(
                "jax-spice is not available. Install with: pip install jax-spice"
            )

        # Import jax-spice
        from jax_spice import CircuitEngine  # type: ignore

        # Build circuit from network
        # TODO: Implement proper jax-spice MIR conversion
        # For now, use the admittance matrices directly
        G, C = network.get_admittance_matrices()

        # Create engine and simulate
        # This is a placeholder - actual implementation depends on jax-spice API
        engine = CircuitEngine.from_matrices(G, C)

        result = engine.tran(t_stop=t_stop, tstep=t_step)

        return SimulationResult(
            time=np.array(result.time),
            voltages={n.spice_name: np.array(result.v[i]) for i, n in enumerate(network.nodes)},
        )

    def simulate_ac(
        self,
        network: RCNetwork,
        f_start: float = 1e3,
        f_stop: float = 10e9,
        points_per_decade: int = 10,
    ) -> SimulationResult:
        """Simulate AC frequency response of an RC network.

        Args:
            network: Extracted RC network to simulate.
            f_start: Start frequency in Hz.
            f_stop: Stop frequency in Hz.
            points_per_decade: Frequency points per decade.

        Returns:
            SimulationResult with frequency response.

        Raises:
            ImportError: If jax-spice is not available.
        """
        if not self._available:
            raise ImportError(
                "jax-spice is not available. Install with: pip install jax-spice"
            )

        # Placeholder for AC simulation
        raise NotImplementedError("AC simulation not yet implemented")


class AnalyticalSimulator:
    """Analytical simulation for simple RC networks.

    This provides a fallback when jax-spice is not available,
    using analytical solutions for simple RC ladder networks.
    """

    def simulate_rc_step_response(
        self,
        r_total: float,
        c_total: float,
        t_stop: float = 100e-9,
        n_points: int = 1000,
        v_step: float = 1.0,
    ) -> SimulationResult:
        """Simulate step response of a simple RC network.

        Uses the analytical solution: v(t) = V_step * (1 - exp(-t/tau))
        where tau = R * C.

        Args:
            r_total: Total resistance in Ohms.
            c_total: Total capacitance in Farads.
            t_stop: Simulation stop time in seconds.
            n_points: Number of time points.
            v_step: Step voltage amplitude.

        Returns:
            SimulationResult with analytical waveforms.
        """
        tau = r_total * c_total
        t = np.linspace(0, t_stop, n_points)

        # Input step
        v_in = np.where(t > 0, v_step, 0.0)

        # Output response
        v_out = v_step * (1 - np.exp(-t / tau))

        return SimulationResult(
            time=t,
            voltages={
                "in": v_in,
                "out": v_out,
            },
        )

    def simulate_elmore_delay(
        self,
        network: RCNetwork,
        t_stop: float = 100e-9,
        n_points: int = 1000,
        v_step: float = 1.0,
    ) -> SimulationResult:
        """Simulate using Elmore delay approximation.

        Args:
            network: RC network.
            t_stop: Simulation stop time.
            n_points: Number of time points.
            v_step: Step voltage.

        Returns:
            SimulationResult with approximate waveforms.
        """
        r_total = network.total_resistance()
        c_total = network.total_capacitance()

        return self.simulate_rc_step_response(
            r_total=r_total,
            c_total=c_total,
            t_stop=t_stop,
            n_points=n_points,
            v_step=v_step,
        )


def validate_extraction(
    network: RCNetwork,
    expected_tau: float,
    tolerance: float = 0.1,
) -> tuple[bool, float, float]:
    """Validate extracted network against expected time constant.

    Uses either jax-spice (if available) or analytical simulation
    to measure the actual time constant and compare.

    Args:
        network: Extracted RC network.
        expected_tau: Expected time constant in seconds.
        tolerance: Acceptable relative error (0.1 = 10%).

    Returns:
        Tuple of (passed, measured_tau, relative_error).
    """
    # Try jax-spice first, fall back to analytical
    jax_sim = JaxSpiceSimulator()

    if jax_sim.is_available():
        result = jax_sim.simulate_step_response(
            network,
            t_stop=10 * expected_tau,
        )
        out_node = network.get_pins()[-1].spice_name if network.get_pins() else "out"
        measured_tau = result.measure_time_constant(out_node)
    else:
        # Use analytical simulation
        ana_sim = AnalyticalSimulator()
        result = ana_sim.simulate_elmore_delay(
            network,
            t_stop=10 * expected_tau,
        )
        measured_tau = result.measure_time_constant("out")

    if measured_tau is None:
        return False, 0.0, float("inf")

    relative_error = abs(measured_tau - expected_tau) / expected_tau
    passed = relative_error <= tolerance

    return passed, measured_tau, relative_error
