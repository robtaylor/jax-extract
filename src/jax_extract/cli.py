"""Command-line interface for jax-extract.

Provides commands for parasitic extraction, test structure generation,
validation, and calibration.
"""

import logging
import sys
from pathlib import Path

import click

from jax_extract import __version__

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("jax-extract")


@click.group()
@click.version_option(version=__version__, prog_name="jax-extract")
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose output")
@click.pass_context
def main(ctx: click.Context, verbose: bool) -> None:
    """jax-extract: JAX-accelerated post-layout parasitic extraction.

    A tool for extracting RC parasitics from GDS layouts using a
    field solver approach with iterative validation.
    """
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose

    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)


@main.command()
@click.option(
    "--gds",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Input GDS file",
)
@click.option(
    "--cell",
    type=str,
    required=True,
    help="Cell name to extract",
)
@click.option(
    "--pdk",
    type=click.Choice(["ihp-sg13g2"]),
    default="ihp-sg13g2",
    help="Target PDK",
)
@click.option(
    "--corner",
    type=click.Choice(["typical", "cbest", "cworst", "rcbest", "rcworst"]),
    default="typical",
    help="Process corner",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(path_type=Path),
    help="Output file (SPICE netlist)",
)
@click.option(
    "--mode",
    type=click.Choice(["c", "rc", "rlc"]),
    default="rc",
    help="Extraction mode: c=capacitance only, rc=RC, rlc=RLC",
)
@click.option(
    "--num-walks",
    type=int,
    default=10000,
    help="Number of random walks for FRW solver",
)
@click.option(
    "--coupling-threshold",
    type=float,
    default=1e-18,
    help="Minimum coupling capacitance to include (Farads)",
)
@click.pass_context
def extract(
    ctx: click.Context,
    gds: Path,
    cell: str,
    pdk: str,
    corner: str,
    output: Path | None,
    mode: str,
    num_walks: int,
    coupling_threshold: float,
) -> None:
    """Extract parasitics from a GDS layout.

    Reads the specified GDS file, extracts RC parasitics for the given cell,
    and outputs a SPICE netlist with the extracted values.

    Example:

        jax-extract extract --gds design.gds --cell top --pdk ihp-sg13g2 -o parasitics.spice
    """
    from jax_extract.extraction.base import ExtractionConfig, ExtractionMode
    from jax_extract.pdk.adapter import ExtractionAdapter, get_ihp_sg13g2_technology
    from jax_extract.pdk.corners import CornerConfig, CornerType, apply_corner_to_technology

    click.echo(f"Extracting parasitics from {gds}")
    click.echo(f"  Cell: {cell}")
    click.echo(f"  PDK: {pdk}")
    click.echo(f"  Corner: {corner}")
    click.echo(f"  Mode: {mode}")

    # Get technology data
    if pdk == "ihp-sg13g2":
        try:
            from c4m_pdk_ihpsg13g2 import tech as ihp_tech

            adapter = ExtractionAdapter(ihp_tech)
            tech_data = adapter.get_technology_data()
            click.echo("  Using PDKMaster technology definition")
        except ImportError:
            click.echo("  PDKMaster not available, using built-in layer data")
            tech_data = get_ihp_sg13g2_technology()

    # Apply corner
    corner_config = CornerConfig.from_corner_type(CornerType(corner))
    tech_data = apply_corner_to_technology(tech_data, corner_config)

    # Configure extraction
    mode_map = {
        "c": ExtractionMode.CAPACITANCE_ONLY,
        "rc": ExtractionMode.RC,
        "rlc": ExtractionMode.RLC,
    }
    _config = ExtractionConfig(
        mode=mode_map[mode],
        frw_num_walks=num_walks,
        coupling_threshold=coupling_threshold,
    )

    click.echo(f"  FRW walks: {num_walks}")
    click.echo(f"  Coupling threshold: {coupling_threshold:.2e} F")

    # TODO: Implement full extraction pipeline
    # For now, just validate the setup
    click.echo("\nExtraction pipeline not yet fully implemented.")
    click.echo("Use 'jax-extract testgen' to generate test structures for validation.")

    if output:
        click.echo(f"\nOutput would be written to: {output}")


@main.command()
@click.option(
    "--type",
    "structure_type",
    type=click.Choice(["serpentine", "comb", "via-chain", "area-cap", "all"]),
    required=True,
    help="Type of test structure to generate",
)
@click.option(
    "--width",
    type=float,
    default=0.5,
    help="Wire width in micrometers",
)
@click.option(
    "--spacing",
    type=float,
    default=0.5,
    help="Wire spacing in micrometers",
)
@click.option(
    "--length",
    type=float,
    default=100.0,
    help="Structure length in micrometers",
)
@click.option(
    "--layer",
    type=str,
    default="Metal1",
    help="Metal layer name",
)
@click.option(
    "--turns",
    type=int,
    default=10,
    help="Number of turns (for serpentine)",
)
@click.option(
    "--fingers",
    type=int,
    default=20,
    help="Number of fingers (for comb)",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(path_type=Path),
    required=True,
    help="Output GDS file",
)
@click.pass_context
def testgen(
    ctx: click.Context,
    structure_type: str,
    width: float,
    spacing: float,
    length: float,
    layer: str,
    turns: int,
    fingers: int,
    output: Path,
) -> None:
    """Generate test structures for calibration.

    Creates GDS files containing test structures for validating and
    calibrating the extraction tool.

    Example:

        jax-extract testgen --type serpentine --width 0.5 --layer Metal1 -o serpentine.gds
    """
    click.echo(f"Generating {structure_type} test structure")
    click.echo(f"  Width: {width} um")
    click.echo(f"  Spacing: {spacing} um")
    click.echo(f"  Length: {length} um")
    click.echo(f"  Layer: {layer}")

    if structure_type == "serpentine":
        click.echo(f"  Turns: {turns}")
        _generate_serpentine(output, width, spacing, length, turns, layer)
    elif structure_type == "comb":
        click.echo(f"  Fingers: {fingers}")
        _generate_comb(output, width, spacing, length, fingers, layer)
    elif structure_type == "via-chain":
        _generate_via_chain(output, width, layer)
    elif structure_type == "area-cap":
        _generate_area_cap(output, length, layer)
    elif structure_type == "all":
        # Generate all test structures
        base = output.stem
        parent = output.parent

        _generate_serpentine(
            parent / f"{base}_serpentine.gds", width, spacing, length, turns, layer
        )
        _generate_comb(
            parent / f"{base}_comb.gds", width, spacing, length, fingers, layer
        )
        _generate_via_chain(parent / f"{base}_via_chain.gds", width, layer)
        _generate_area_cap(parent / f"{base}_area_cap.gds", length, layer)


def _generate_serpentine(
    output: Path,
    width: float,
    spacing: float,
    length: float,
    turns: int,
    layer: str,
) -> None:
    """Generate serpentine test structure."""
    try:
        import klayout.db as kdb
    except ImportError:
        click.echo("Error: KLayout Python module required. Install with: pip install klayout")
        sys.exit(1)

    layout = kdb.Layout()
    layout.dbu = 0.001  # 1nm database unit

    # Create cell
    cell = layout.create_cell("serpentine")

    # Get layer (assume Metal1 = layer 8)
    layer_map = {"Metal1": 8, "Metal2": 10, "Metal3": 30, "Metal4": 50, "Metal5": 67}
    gds_layer = layer_map.get(layer, 8)
    layer_idx = layout.layer(gds_layer, 0)

    # Convert to database units
    w = int(width / layout.dbu)
    s = int(spacing / layout.dbu)
    len_dbu = int(length / layout.dbu)

    # Generate serpentine pattern
    y = 0
    for i in range(turns + 1):
        # Horizontal segment
        if i % 2 == 0:
            x_start = 0
            x_end = len_dbu
        else:
            x_start = len_dbu
            x_end = 0

        box = kdb.Box(
            min(x_start, x_end),
            y,
            max(x_start, x_end),
            y + w,
        )
        cell.shapes(layer_idx).insert(box)

        # Vertical connector (except after last horizontal)
        if i < turns:
            if i % 2 == 0:
                # Right side connector
                conn_box = kdb.Box(len_dbu - w, y + w, len_dbu, y + w + s + w)
            else:
                # Left side connector
                conn_box = kdb.Box(0, y + w, w, y + w + s + w)
            cell.shapes(layer_idx).insert(conn_box)

        y += w + s

    # Write GDS
    layout.write(str(output))
    click.echo(f"  Written to: {output}")

    # Calculate expected resistance
    from jax_extract.extraction.resistance import calculate_serpentine_resistance
    from jax_extract.pdk.adapter import get_ihp_sg13g2_technology

    tech = get_ihp_sg13g2_technology()
    layer_info = tech.get_layer(layer)
    if layer_info:
        expected_r = calculate_serpentine_resistance(
            width, spacing, turns, length, layer_info.sheet_resistance_ohm_sq
        )
        click.echo(f"  Expected resistance: {expected_r:.2f} Ohms")


def _generate_comb(
    output: Path,
    width: float,
    spacing: float,
    length: float,
    fingers: int,
    layer: str,
) -> None:
    """Generate comb/fork test structure."""
    try:
        import klayout.db as kdb
    except ImportError:
        click.echo("Error: KLayout Python module required.")
        sys.exit(1)

    layout = kdb.Layout()
    layout.dbu = 0.001

    cell = layout.create_cell("comb")

    layer_map = {"Metal1": 8, "Metal2": 10, "Metal3": 30, "Metal4": 50, "Metal5": 67}
    gds_layer = layer_map.get(layer, 8)
    layer_idx = layout.layer(gds_layer, 0)

    w = int(width / layout.dbu)
    s = int(spacing / layout.dbu)
    len_dbu = int(length / layout.dbu)

    # Bus bars at top and bottom
    total_width = fingers * (w + s) - s
    bus_height = w * 2

    # Bottom bus (signal A)
    cell.shapes(layer_idx).insert(kdb.Box(0, 0, total_width, bus_height))

    # Top bus (signal B)
    top_y = len_dbu + bus_height * 2
    cell.shapes(layer_idx).insert(kdb.Box(0, top_y, total_width, top_y + bus_height))

    # Interdigitated fingers
    for i in range(fingers):
        x = i * (w + s)
        if i % 2 == 0:
            # Connected to bottom bus
            cell.shapes(layer_idx).insert(
                kdb.Box(x, bus_height, x + w, bus_height + len_dbu)
            )
        else:
            # Connected to top bus
            cell.shapes(layer_idx).insert(
                kdb.Box(x, bus_height + bus_height, x + w, top_y)
            )

    layout.write(str(output))
    click.echo(f"  Written to: {output}")


def _generate_via_chain(output: Path, width: float, layer: str) -> None:
    """Generate via chain test structure."""
    click.echo("  Via chain generation not yet implemented")
    click.echo(f"  Would write to: {output}")


def _generate_area_cap(output: Path, size: float, layer: str) -> None:
    """Generate area capacitor test structure."""
    try:
        import klayout.db as kdb
    except ImportError:
        click.echo("Error: KLayout Python module required.")
        sys.exit(1)

    layout = kdb.Layout()
    layout.dbu = 0.001

    cell = layout.create_cell("area_cap")

    layer_map = {"Metal1": 8, "Metal2": 10, "Metal3": 30, "Metal4": 50, "Metal5": 67}
    gds_layer = layer_map.get(layer, 8)
    layer_idx = layout.layer(gds_layer, 0)

    s = int(size / layout.dbu)
    cell.shapes(layer_idx).insert(kdb.Box(0, 0, s, s))

    layout.write(str(output))
    click.echo(f"  Written to: {output}")

    # Calculate expected capacitance
    from jax_extract.extraction.field_solver import extract_capacitance_parallel_plate
    from jax_extract.pdk.adapter import get_ihp_sg13g2_technology

    tech = get_ihp_sg13g2_technology()
    layer_info = tech.get_layer(layer)
    if layer_info:
        expected_c = extract_capacitance_parallel_plate(
            size * size,  # area in um^2
            layer_info.height_nm,  # distance to substrate
            3.9,  # SiO2 permittivity
        )
        click.echo(f"  Expected capacitance: {expected_c * 1e15:.3f} fF")


@main.command()
@click.option(
    "--extracted",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Extracted SPICE netlist",
)
@click.option(
    "--reference",
    type=click.Choice(["analytical", "simulation", "measurement"]),
    default="analytical",
    help="Reference for comparison",
)
@click.option(
    "--tolerance",
    type=float,
    default=0.05,
    help="Acceptable relative error",
)
@click.pass_context
def validate(
    ctx: click.Context,
    extracted: Path,
    reference: str,
    tolerance: float,
) -> None:
    """Validate extracted parasitics against reference.

    Compares extracted values against analytical formulas, simulation
    results, or measurement data.

    Example:

        jax-extract validate --extracted parasitics.spice --reference analytical
    """
    click.echo(f"Validating {extracted} against {reference} reference")
    click.echo(f"  Tolerance: {tolerance * 100:.1f}%")

    # TODO: Implement validation
    click.echo("\nValidation not yet implemented.")


@main.command()
@click.option(
    "--pdk",
    type=click.Choice(["ihp-sg13g2"]),
    default="ihp-sg13g2",
    help="Target PDK",
)
@click.option(
    "--structures",
    type=click.Choice(["serpentine", "comb", "via-chain", "area-cap", "all"]),
    default="all",
    help="Test structures to use",
)
@click.option(
    "--iterations",
    type=int,
    default=10,
    help="Number of calibration iterations",
)
@click.option(
    "--target-error",
    type=float,
    default=0.05,
    help="Target relative error for convergence",
)
@click.pass_context
def calibrate(
    ctx: click.Context,
    pdk: str,
    structures: str,
    iterations: int,
    target_error: float,
) -> None:
    """Run iterative calibration loop.

    Generates test structures, extracts parasitics, simulates, and
    compares against analytical references. Adjusts parameters to
    minimize error.

    Example:

        jax-extract calibrate --pdk ihp-sg13g2 --structures all --iterations 10
    """
    click.echo(f"Running calibration for {pdk}")
    click.echo(f"  Structures: {structures}")
    click.echo(f"  Iterations: {iterations}")
    click.echo(f"  Target error: {target_error * 100:.1f}%")

    # TODO: Implement calibration loop
    click.echo("\nCalibration loop not yet implemented.")


if __name__ == "__main__":
    main()
