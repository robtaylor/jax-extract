# jax-extract

JAX-accelerated post-layout parasitic extraction tool targeting the IHP SG13G2 PDK.

## Overview

jax-extract is a parasitic extraction tool that uses:

- **Floating Random Walk (FRW)** field solver for capacitance extraction
- **JAX** for GPU acceleration and parallel random walk execution
- **PDKMaster** for technology abstraction (via c4m-pdk-ihpsg13g2)
- **KLayout** for GDS geometry processing

## Installation

```bash
# Using uv
uv sync

# Or pip
pip install -e .
```

## Quick Start

```bash
# Extract parasitics from a GDS file
jax-extract extract --gds design.gds --cell top --pdk ihp-sg13g2 -o parasitics.spice

# Generate test structures for calibration
jax-extract testgen --type serpentine --width 0.5 --layer Metal1 -o test.gds

# Validate against analytical references
jax-extract validate --extracted parasitics.spice --reference analytical
```

## Features

- **Capacitance extraction** using Monte Carlo FRW field solver
- **Resistance extraction** from wire geometry and via stacks
- **Corner support** (typical, cbest, cworst, rcbest, rcworst)
- **Test structure generation** for validation and calibration
- **Multi-level validation** against analytical formulas

## Project Status

This is Phase 1 (MVP) implementation with:

- [x] PDK adapter for IHP SG13G2
- [x] FRW field solver (JAX-accelerated)
- [x] Resistance extraction
- [x] RC network representation
- [x] CLI interface
- [x] Analytical validation tests

## License

Apache-2.0
