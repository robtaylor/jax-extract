"""Floating Random Walk (FRW) field solver for capacitance extraction.

This module implements a JAX-accelerated FRW solver for extracting
capacitance from 3D conductor geometry. The algorithm uses Monte Carlo
random walks to solve Laplace's equation.

References:
    - RWCap: A Floating Random Walk Solver for 3-D Capacitance Extraction
      (IEEE Trans. CAD, 2013)
    - GPU-Friendly Floating Random Walk Algorithm for Capacitance Extraction
      (IEEE Trans. VLSI, 2013)
"""

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

# Physical constants
EPSILON_0 = 8.854187817e-12  # F/m, vacuum permittivity


@dataclass
class DielectricRegion:
    """A dielectric region in the domain."""

    permittivity: float
    """Relative permittivity (epsilon_r)."""

    z_min: float
    """Minimum z coordinate in nanometers."""

    z_max: float
    """Maximum z coordinate in nanometers."""


@dataclass
class ConductorSurface:
    """Conductor surface for capacitance extraction."""

    conductor_id: int
    """Unique conductor identifier."""

    vertices: NDArray[np.float64]
    """Surface mesh vertices, Nx3 in nanometers."""

    triangles: NDArray[np.int64]
    """Triangle connectivity, Mx3 vertex indices."""

    areas: NDArray[np.float64]
    """Pre-computed triangle areas for sampling."""


@dataclass
class ExtractionDomain:
    """Complete extraction domain with conductors and dielectrics."""

    conductors: list[ConductorSurface]
    """List of conductor surfaces."""

    dielectrics: list[DielectricRegion]
    """Dielectric layer stack."""

    boundary_box: tuple[float, float, float, float, float, float]
    """Domain bounding box (xmin, ymin, zmin, xmax, ymax, zmax) in nm."""

    def get_permittivity_at(self, z: float) -> float:
        """Get relative permittivity at height z.

        Args:
            z: Height in nanometers.

        Returns:
            Relative permittivity at that height.
        """
        for region in self.dielectrics:
            if region.z_min <= z < region.z_max:
                return region.permittivity
        # Default to SiO2 if outside defined regions
        return 3.9


# JAX-accelerated functions


@jax.jit
def compute_cube_greens_function(size: float, epsilon_r: float) -> float:
    """Compute Green's function for a transition cube.

    For a random walk starting at the cube center, this gives the
    contribution to capacitance when the walk exits through the surface.

    Args:
        size: Cube side length in nanometers.
        epsilon_r: Relative permittivity of the medium.

    Returns:
        Green's function weight.
    """
    # For a cube of side a, Green's function at center is approximately:
    # G = 1 / (4 * pi * epsilon * a * 0.6)
    # where 0.6 is a geometric factor for cube
    epsilon = EPSILON_0 * epsilon_r * 1e-9  # Convert to F/nm
    return 1.0 / (4.0 * jnp.pi * epsilon * size * 0.6)


@jax.jit
def sample_cube_exit_point(
    key: jax.Array,
    center: jax.Array,
    size: float,
) -> jax.Array:
    """Sample a random exit point on cube surface.

    Args:
        key: JAX random key.
        center: Cube center (x, y, z).
        size: Cube side length.

    Returns:
        Exit point coordinates (x, y, z).
    """
    half = size / 2.0

    # Sample which face (0-5)
    key, subkey = jax.random.split(key)
    face = jax.random.randint(subkey, (), 0, 6)

    # Sample position on face
    key, subkey1, subkey2 = jax.random.split(key, 3)
    u = jax.random.uniform(subkey1) * size - half
    v = jax.random.uniform(subkey2) * size - half

    # Build exit point based on face
    # Faces: 0=-x, 1=+x, 2=-y, 3=+y, 4=-z, 5=+z
    exit_point = jnp.where(
        face == 0,
        jnp.array([center[0] - half, center[1] + u, center[2] + v]),
        jnp.where(
            face == 1,
            jnp.array([center[0] + half, center[1] + u, center[2] + v]),
            jnp.where(
                face == 2,
                jnp.array([center[0] + u, center[1] - half, center[2] + v]),
                jnp.where(
                    face == 3,
                    jnp.array([center[0] + u, center[1] + half, center[2] + v]),
                    jnp.where(
                        face == 4,
                        jnp.array([center[0] + u, center[1] + v, center[2] - half]),
                        jnp.array([center[0] + u, center[1] + v, center[2] + half]),
                    ),
                ),
            ),
        ),
    )

    return exit_point


@jax.jit
def point_in_box(
    point: jax.Array,
    box_min: jax.Array,
    box_max: jax.Array,
) -> jax.Array:
    """Check if point is inside axis-aligned box.

    Args:
        point: Point coordinates (x, y, z).
        box_min: Box minimum (xmin, ymin, zmin).
        box_max: Box maximum (xmax, ymax, zmax).

    Returns:
        Boolean indicating if point is inside.
    """
    return jnp.all((point >= box_min) & (point <= box_max))


@jax.jit
def distance_to_boundary(
    point: jax.Array,
    box_min: jax.Array,
    box_max: jax.Array,
) -> jax.Array:
    """Compute distance to nearest domain boundary.

    Args:
        point: Point coordinates (x, y, z).
        box_min: Domain minimum (xmin, ymin, zmin).
        box_max: Domain maximum (xmax, ymax, zmax).

    Returns:
        Distance to nearest boundary.
    """
    dist_to_min = point - box_min
    dist_to_max = box_max - point
    return jnp.minimum(jnp.min(dist_to_min), jnp.min(dist_to_max))


@partial(jax.jit, static_argnums=(3,))
def point_to_triangle_distance_batch(
    point: jax.Array,
    vertices: jax.Array,
    triangles: jax.Array,
    n_triangles: int,
) -> jax.Array:
    """Compute distance from point to each triangle in mesh.

    Args:
        point: Query point (x, y, z).
        vertices: All vertices, Nx3.
        triangles: Triangle indices, Mx3.
        n_triangles: Number of triangles (static for JIT).

    Returns:
        Array of distances to each triangle.
    """

    def triangle_distance(tri_idx):
        v0 = vertices[triangles[tri_idx, 0]]
        v1 = vertices[triangles[tri_idx, 1]]
        v2 = vertices[triangles[tri_idx, 2]]

        # Project point onto triangle plane
        e1 = v1 - v0
        e2 = v2 - v0
        normal = jnp.cross(e1, e2)
        normal = normal / jnp.linalg.norm(normal)

        # Distance to plane (unused in simplified version but kept for reference)
        _d = jnp.dot(point - v0, normal)

        # Check if projection is inside triangle using barycentric coords
        # Simplified: just use distance to centroid as approximation
        centroid = (v0 + v1 + v2) / 3.0
        dist_to_centroid = jnp.linalg.norm(point - centroid)

        return dist_to_centroid

    distances = jax.vmap(triangle_distance)(jnp.arange(n_triangles))
    return distances


def find_max_transition_cube(
    position: NDArray[np.float64],
    domain: ExtractionDomain,
    conductors_jax: list[tuple[jax.Array, jax.Array]],
    min_size: float = 1.0,
    max_size: float = 1000.0,
) -> float:
    """Find maximum transition cube size that doesn't intersect conductors.

    Args:
        position: Current walker position (x, y, z) in nm.
        domain: Extraction domain.
        conductors_jax: Pre-converted conductor meshes as JAX arrays.
        min_size: Minimum cube size in nm.
        max_size: Maximum cube size in nm.

    Returns:
        Maximum safe cube side length in nm.
    """
    # Start with distance to domain boundary
    pos = jnp.array(position)
    box_min = jnp.array(domain.boundary_box[:3])
    box_max = jnp.array(domain.boundary_box[3:])

    max_cube = float(distance_to_boundary(pos, box_min, box_max))

    # Check distance to each conductor
    for vertices, triangles in conductors_jax:
        n_tri = triangles.shape[0]
        if n_tri > 0:
            distances = point_to_triangle_distance_batch(pos, vertices, triangles, n_tri)
            min_dist = float(jnp.min(distances))
            max_cube = min(max_cube, min_dist)

    # Clamp to valid range
    return max(min_size, min(max_cube * 0.9, max_size))  # 0.9 safety factor


@dataclass
class FRWResult:
    """Result of FRW capacitance extraction."""

    capacitance_matrix: NDArray[np.float64]
    """Capacitance matrix in Farads. C[i,j] is capacitance between
    conductor i and j. C[i,i] is self-capacitance to ground."""

    variance: NDArray[np.float64]
    """Estimated variance of capacitance values."""

    num_walks: int
    """Total number of random walks performed."""

    convergence_history: list[float]
    """History of capacitance estimates for convergence tracking."""


class FRWSolver:
    """Floating Random Walk solver for capacitance extraction.

    This solver uses Monte Carlo random walks to solve Laplace's equation
    and extract capacitance values. It is accelerated using JAX for
    parallel random walk execution.

    Example:
        >>> solver = FRWSolver(num_walks=10000)
        >>> domain = ExtractionDomain(conductors=[...], dielectrics=[...])
        >>> result = solver.solve(domain)
        >>> print(f"Capacitance: {result.capacitance_matrix[0,0]:.3e} F")
    """

    def __init__(
        self,
        num_walks: int = 10000,
        tolerance: float = 0.01,
        max_steps: int = 10000,
        seed: int = 42,
    ):
        """Initialize FRW solver.

        Args:
            num_walks: Number of random walks per conductor.
            tolerance: Target relative tolerance for convergence.
            max_steps: Maximum steps per random walk.
            seed: Random seed for reproducibility.
        """
        self.num_walks = num_walks
        self.tolerance = tolerance
        self.max_steps = max_steps
        self.seed = seed

    def solve(self, domain: ExtractionDomain) -> FRWResult:
        """Solve for capacitance using FRW algorithm.

        Args:
            domain: Extraction domain with conductors and dielectrics.

        Returns:
            FRWResult with capacitance matrix.
        """
        n_conductors = len(domain.conductors)
        if n_conductors == 0:
            return FRWResult(
                capacitance_matrix=np.array([]),
                variance=np.array([]),
                num_walks=0,
                convergence_history=[],
            )

        # Initialize capacitance accumulator
        cap_sum = np.zeros((n_conductors, n_conductors))
        cap_sum_sq = np.zeros((n_conductors, n_conductors))

        # Convert conductor meshes to JAX arrays
        conductors_jax = []
        for cond in domain.conductors:
            vertices = jnp.array(cond.vertices)
            triangles = jnp.array(cond.triangles)
            conductors_jax.append((vertices, triangles))

        # Pre-compute sampling probabilities (reserved for future use)
        _total_areas = [np.sum(cond.areas) for cond in domain.conductors]

        # Initialize random key
        key = jax.random.PRNGKey(self.seed)

        convergence_history = []

        # Run random walks from each conductor
        for source_id, source_cond in enumerate(domain.conductors):
            # Sample starting points on source conductor surface
            key, subkey = jax.random.split(key)
            start_points = self._sample_surface_points(
                source_cond, self.num_walks, subkey
            )

            # Run walks in batches
            batch_size = min(1000, self.num_walks)
            n_batches = (self.num_walks + batch_size - 1) // batch_size

            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, self.num_walks)
                batch_points = start_points[start_idx:end_idx]

                key, subkey = jax.random.split(key)
                weights, targets = self._run_walk_batch(
                    batch_points, domain, conductors_jax, subkey
                )

                # Accumulate results
                for weight, target_id in zip(weights, targets, strict=True):
                    if target_id >= 0:  # Hit a conductor
                        cap_sum[source_id, target_id] += weight
                        cap_sum_sq[source_id, target_id] += weight * weight
                    else:  # Hit boundary (ground)
                        cap_sum[source_id, source_id] += weight
                        cap_sum_sq[source_id, source_id] += weight * weight

            # Track convergence
            if self.num_walks > 0:
                avg_cap = cap_sum[source_id, source_id] / self.num_walks
                convergence_history.append(float(avg_cap))

        # Compute final capacitance (average of walks)
        total_walks = self.num_walks * n_conductors
        capacitance_matrix = cap_sum / self.num_walks

        # Compute variance
        variance = (cap_sum_sq / self.num_walks - (cap_sum / self.num_walks) ** 2) / self.num_walks

        return FRWResult(
            capacitance_matrix=capacitance_matrix,
            variance=variance,
            num_walks=total_walks,
            convergence_history=convergence_history,
        )

    def _sample_surface_points(
        self,
        conductor: ConductorSurface,
        n_samples: int,
        key: jax.Array,
    ) -> NDArray[np.float64]:
        """Sample random points on conductor surface.

        Args:
            conductor: Conductor surface mesh.
            n_samples: Number of points to sample.
            key: JAX random key.

        Returns:
            Array of sampled points, n_samples x 3.
        """
        # Sample triangles weighted by area
        probs = conductor.areas / np.sum(conductor.areas)
        key, subkey = jax.random.split(key)
        tri_indices = jax.random.choice(
            subkey, len(conductor.triangles), shape=(n_samples,), p=jnp.array(probs)
        )
        tri_indices = np.array(tri_indices)

        # Sample random barycentric coordinates
        key, subkey1, subkey2 = jax.random.split(key, 3)
        r1 = np.array(jax.random.uniform(subkey1, shape=(n_samples,)))
        r2 = np.array(jax.random.uniform(subkey2, shape=(n_samples,)))

        sqrt_r1 = np.sqrt(r1)
        u = 1 - sqrt_r1
        v = sqrt_r1 * (1 - r2)
        w = sqrt_r1 * r2

        # Compute sample points
        v0 = conductor.vertices[conductor.triangles[tri_indices, 0]]
        v1 = conductor.vertices[conductor.triangles[tri_indices, 1]]
        v2 = conductor.vertices[conductor.triangles[tri_indices, 2]]

        points = u[:, np.newaxis] * v0 + v[:, np.newaxis] * v1 + w[:, np.newaxis] * v2

        return points

    def _run_walk_batch(
        self,
        start_points: NDArray[np.float64],
        domain: ExtractionDomain,
        conductors_jax: list[tuple[jax.Array, jax.Array]],
        key: jax.Array,
    ) -> tuple[list[float], list[int]]:
        """Run a batch of random walks.

        Args:
            start_points: Starting positions, Nx3.
            domain: Extraction domain.
            conductors_jax: Pre-converted conductor meshes.
            key: JAX random key.

        Returns:
            Tuple of (weights, target_conductor_ids).
            target_id is -1 for ground (boundary hit).
        """
        weights = []
        targets = []

        box_min = np.array(domain.boundary_box[:3])
        box_max = np.array(domain.boundary_box[3:])

        for _i, start in enumerate(start_points):
            key, walk_key = jax.random.split(key)
            weight, target = self._single_walk(
                start, domain, conductors_jax, box_min, box_max, walk_key
            )
            weights.append(weight)
            targets.append(target)

        return weights, targets

    def _single_walk(
        self,
        start: NDArray[np.float64],
        domain: ExtractionDomain,
        conductors_jax: list[tuple[jax.Array, jax.Array]],
        box_min: NDArray[np.float64],
        box_max: NDArray[np.float64],
        key: jax.Array,
    ) -> tuple[float, int]:
        """Execute a single random walk.

        Args:
            start: Starting position.
            domain: Extraction domain.
            conductors_jax: Conductor meshes as JAX arrays.
            box_min: Domain minimum.
            box_max: Domain maximum.
            key: JAX random key.

        Returns:
            Tuple of (accumulated_weight, target_conductor_id).
        """
        position = start.copy()
        accumulated_weight = 0.0

        for _step in range(self.max_steps):
            # Check if we're still in domain
            if not np.all((position >= box_min) & (position <= box_max)):
                # Hit domain boundary (ground)
                return accumulated_weight, -1

            # Find maximum transition cube
            cube_size = find_max_transition_cube(
                position, domain, conductors_jax, min_size=1.0
            )

            # Check if we've hit a conductor (cube_size very small)
            if cube_size < 2.0:  # Within 2nm, consider it a hit
                # Find which conductor we hit
                pos_jax = jnp.array(position)
                for cond_id, (vertices, triangles) in enumerate(conductors_jax):
                    n_tri = triangles.shape[0]
                    if n_tri > 0:
                        distances = point_to_triangle_distance_batch(
                            pos_jax, vertices, triangles, n_tri
                        )
                        if float(jnp.min(distances)) < 10.0:  # Within 10nm
                            return accumulated_weight, cond_id
                # No conductor found, treat as ground
                return accumulated_weight, -1

            # Get permittivity at current height
            epsilon_r = domain.get_permittivity_at(position[2])

            # Compute Green's function contribution
            weight = float(compute_cube_greens_function(cube_size, epsilon_r))
            accumulated_weight += weight

            # Sample exit point
            key, subkey = jax.random.split(key)
            exit_point = sample_cube_exit_point(
                subkey, jnp.array(position), cube_size
            )
            position = np.array(exit_point)

        # Max steps reached, treat as ground
        return accumulated_weight, -1


def extract_capacitance_parallel_plate(
    area_um2: float,
    thickness_nm: float,
    permittivity: float = 3.9,
) -> float:
    """Analytical capacitance for parallel plate (for validation).

    Args:
        area_um2: Plate area in square micrometers.
        thickness_nm: Dielectric thickness in nanometers.
        permittivity: Relative permittivity.

    Returns:
        Capacitance in Farads.
    """
    area_m2 = area_um2 * 1e-12
    thickness_m = thickness_nm * 1e-9
    return EPSILON_0 * permittivity * area_m2 / thickness_m


def extract_capacitance_wire(
    length_um: float,
    width_um: float,
    height_nm: float,
    thickness_nm: float,
    permittivity: float = 3.9,
) -> float:
    """Estimate wire capacitance (area + fringe) for validation.

    Uses empirical formula: C = C_area + C_fringe
    where C_fringe ≈ 2 * ε * length * ln(1 + thickness/width)

    Args:
        length_um: Wire length in micrometers.
        width_um: Wire width in micrometers.
        height_nm: Height above substrate in nanometers.
        thickness_nm: Wire thickness in nanometers.
        permittivity: Relative permittivity.

    Returns:
        Approximate capacitance in Farads.
    """
    # Area capacitance (bottom of wire to substrate)
    area_m2 = length_um * width_um * 1e-12
    height_m = height_nm * 1e-9
    c_area = EPSILON_0 * permittivity * area_m2 / height_m

    # Fringe capacitance (empirical)
    length_m = length_um * 1e-6
    thickness_m = thickness_nm * 1e-9
    width_m = width_um * 1e-6
    c_fringe = 2 * EPSILON_0 * permittivity * length_m * np.log(1 + thickness_m / width_m)

    return c_area + c_fringe
