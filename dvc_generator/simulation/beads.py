"""
Bead/particle generation and rendering for synthetic confocal microscopy.
"""

import numpy as np
from scipy.spatial.distance import cdist


class BeadGenerator:
    """Generate random fluorescent beads/particles."""

    def __init__(self, volume_shape, config):
        """
        Args:
            volume_shape: Tuple (D, H, W)
            config: Configuration dict for particles
        """
        self.volume_shape = np.array(volume_shape)
        self.config = config

    def generate_beads(self):
        """
        Generate random beads with positions, radii, and intensities.
        If config contains particle_groups, each group supplies num_range,
        radius_range, and intensity_range.

        Returns:
            beads: dict with keys 'positions', 'radii', 'intensities'
        """
        particle_groups = self.config.get('particle_groups')
        if particle_groups:
            group_radii = []
            group_intensities = []
            group_ids = []

            for group_idx, group in enumerate(particle_groups):
                num_min, num_max = group['num_range']
                num_beads = np.random.randint(int(num_min), int(num_max) + 1)

                radii = self._sample_radii(num_beads, group)
                intensities = self._sample_intensities(num_beads, group)
                group_radii.append(radii)
                group_intensities.append(intensities)
                group_ids.append(np.full(num_beads, group_idx, dtype=np.int32))

            radii = np.concatenate(group_radii)
            intensities = np.concatenate(group_intensities)
            group_ids = np.concatenate(group_ids)
            num_beads = len(radii)
            positions = self._generate_positions(num_beads, radii=radii)
            placed_count = len(positions)

            beads = {
                'positions': positions,  # (N, 3) array
                'radii': radii[:placed_count],  # (N,) array
                'intensities': intensities[:placed_count],  # (N,) array
                'group_ids': group_ids[:placed_count],  # (N,) array
                'num_beads': placed_count
            }

            return beads

        # Sample number of beads
        if self.config['count_stratified']:
            # Stratified sampling for uniform density distribution
            num_beads = self._stratified_count_sampling()
        else:
            num_beads = np.random.randint(self.config['count_min'],
                                         self.config['count_max'] + 1)

        # Generate bead positions (with minimum distance constraint)
        positions = self._generate_positions(num_beads)

        # Sample radii
        radii = self._sample_radii(num_beads)

        # Sample intensities
        intensities = self._sample_intensities(num_beads)

        beads = {
            'positions': positions,  # (N, 3) array
            'radii': radii,          # (N,) array
            'intensities': intensities,  # (N,) array
            'num_beads': num_beads
        }

        return beads

    def _stratified_count_sampling(self):
        """Sample bead count with stratification to ensure density diversity."""
        # Divide range into 4 tiers
        count_min = self.config['count_min']
        count_max = self.config['count_max']
        tier_size = (count_max - count_min) / 4

        # Randomly choose a tier
        tier = np.random.randint(0, 4)

        # Sample within that tier
        tier_min = int(count_min + tier * tier_size)
        tier_max = int(count_min + (tier + 1) * tier_size)

        return np.random.randint(tier_min, tier_max + 1)

    def _generate_positions(self, num_beads, radii=None):
        """
        Generate bead positions with minimum distance constraint.

        Args:
            num_beads: Number of beads to generate
            radii: Optional sampled radii for each bead. When provided, the
                minimum distance constraint uses each bead's actual radius.

        Returns:
            positions: (N, 3) array of bead centers
        """
        if radii is not None:
            radii = np.asarray(radii)

        # Start with random positions
        positions = []
        placed_radii = []

        # Safety margin from boundaries (to avoid edge effects)
        # Can be configured in config, defaults to 5 voxels for backward compatibility
        margin = self.config.get('margin_voxels', 5)

        # Maximum attempts to place a bead
        max_attempts = num_beads * 20

        attempts = 0
        while len(positions) < num_beads and attempts < max_attempts:
            bead_idx = len(positions)
            radius = radii[bead_idx] if radii is not None else None

            # Random position
            pos = np.random.rand(3) * (self.volume_shape - 2 * margin) + margin

            # Check minimum distance to existing beads
            if len(positions) == 0:
                positions.append(pos)
                if radius is not None:
                    placed_radii.append(radius)
            else:
                positions_array = np.array(positions)
                distances = np.linalg.norm(positions_array - pos, axis=1)

                if radius is not None:
                    min_dist = self.config['min_distance_factor'] * \
                              (np.asarray(placed_radii) + radius)
                else:
                    # Minimum distance is a factor of typical radius
                    min_dist = self.config['min_distance_factor'] * \
                              (self.config['radius_min'] + self.config['radius_max']) / 2

                if np.all(distances > min_dist):
                    positions.append(pos)
                    if radius is not None:
                        placed_radii.append(radius)

            attempts += 1

        if len(positions) < num_beads:
            print(f"Warning: Only placed {len(positions)}/{num_beads} beads "
                  f"(min distance constraint)")

        return np.array(positions)

    def _sample_radii(self, num_beads, radius_range=None):
        """
        Sample bead radii.

        Args:
            num_beads: Number of bead radii to generate
            radius_range: Optional [min, max] range for a particle group's radius_range.
        """
        if isinstance(radius_range, dict):
            radius_range = radius_range.get('radius_range')

        if radius_range is not None:
            radius_min, radius_max = radius_range
            return np.random.uniform(radius_min, radius_max, num_beads)

        if self.config['radius_distribution'] == 'lognormal':
            mean = self.config['radius_lognormal_mean']
            std = self.config['radius_lognormal_std']
            radii = np.random.lognormal(mean, std, num_beads)

            # Clip to valid range
            radii = np.clip(radii, self.config['radius_min'],
                           self.config['radius_max'])
        else:  # uniform
            radii = np.random.uniform(self.config['radius_min'],
                                     self.config['radius_max'],
                                     num_beads)

        return radii

    def _sample_intensities(self, num_beads, intensity_range=None):
        """
        Sample bead intensities.

        Args:
            num_beads: Number of bead intensities to generate
            intensity_range: Optional [min, max] range for a particle group's intensity_range.
        """
        if isinstance(intensity_range, dict):
            intensity_range = intensity_range.get('intensity_range')

        if intensity_range is not None:
            intensity_min, intensity_max = intensity_range
            return np.random.uniform(intensity_min, intensity_max, num_beads)

        intensities = np.random.normal(self.config['intensity_mean'],
                                      self.config['intensity_std'],
                                      num_beads)

        # Clip to valid range
        intensities = np.clip(intensities,
                             self.config['intensity_min'],
                             self.config['intensity_max'])

        return intensities


class BeadRenderer:
    """Render beads as 3D Gaussians into a volume."""

    def __init__(self, volume_shape, config=None):
        """
        Args:
            volume_shape: Tuple (D, H, W)
            config: Optional configuration dict for rendering parameters
        """
        self.volume_shape = volume_shape

        # Get Gaussian sharpness parameter (default 2.0 for backward compatibility)
        if config is not None:
            self.gaussian_sharpness = config.get('gaussian_sharpness', 2.0)
            self.anisotropy = np.asarray(
                config.get('anisotropy', [1.0, 1.0, 1.0]),
                dtype=np.float32
            )
            depth_attenuation = config.get('depth_attenuation', {})
            self.depth_attenuation_enabled = depth_attenuation.get('enabled', False)
            self.depth_attenuation_mode = depth_attenuation.get('mode', 'exponential')
            self.depth_attenuation_decay_rate = float(
                depth_attenuation.get('decay_rate', 0.0)
            )
        else:
            self.gaussian_sharpness = 2.0
            self.anisotropy = np.ones(3, dtype=np.float32)
            self.depth_attenuation_enabled = False
            self.depth_attenuation_mode = 'exponential'
            self.depth_attenuation_decay_rate = 0.0

        if self.anisotropy.shape != (3,):
            raise ValueError("particles.anisotropy must be [sigma_z, sigma_y, sigma_x]")

        if np.any(self.anisotropy <= 0):
            raise ValueError("particles.anisotropy values must be positive")

        if self.depth_attenuation_mode not in ('linear', 'exponential'):
            raise ValueError(
                "particles.depth_attenuation.mode must be 'linear' or 'exponential'"
            )

        if self.depth_attenuation_decay_rate < 0:
            raise ValueError("particles.depth_attenuation.decay_rate must be non-negative")

    def render(self, beads):
        """
        Render beads into a volume.

        Args:
            beads: dict with 'positions', 'radii', 'intensities'

        Returns:
            volume: (D, H, W) numpy array
        """
        volume = np.zeros(self.volume_shape, dtype=np.float32)

        positions = beads['positions']
        radii = beads['radii']
        intensities = beads['intensities']

        for i in range(len(positions)):
            self._render_single_bead(
                volume,
                positions[i],
                radii[i],
                intensities[i]
            )

        return volume

    def _render_single_bead(self, volume, position, radius, intensity):
        """
        Render a single bead as a 3D Gaussian.

        Args:
            volume: (D, H, W) array to render into (modified in-place)
            position: (3,) array - bead center
            radius: float - base bead radius
            intensity: float - peak intensity
        """
        sigma = radius * self.anisotropy

        # Determine bounding box (3 sigma) for each axis
        bbox_radius = np.ceil(3 * sigma).astype(int)

        # Integer bounding box
        z0 = max(0, int(position[0]) - bbox_radius[0])
        z1 = min(self.volume_shape[0], int(position[0]) + bbox_radius[0] + 1)
        y0 = max(0, int(position[1]) - bbox_radius[1])
        y1 = min(self.volume_shape[1], int(position[1]) + bbox_radius[1] + 1)
        x0 = max(0, int(position[2]) - bbox_radius[2])
        x1 = min(self.volume_shape[2], int(position[2]) + bbox_radius[2] + 1)

        # Skip if bead is outside volume
        if z0 >= z1 or y0 >= y1 or x0 >= x1:
            return

        # Create local coordinate grids
        z_grid, y_grid, x_grid = np.meshgrid(
            np.arange(z0, z1),
            np.arange(y0, y1),
            np.arange(x0, x1),
            indexing='ij'
        )

        # Distance from bead center
        dz = z_grid - position[0]
        dy = y_grid - position[1]
        dx = x_grid - position[2]
        dist_sq = (
            dz**2 / sigma[0]**2 +
            dy**2 / sigma[1]**2 +
            dx**2 / sigma[2]**2
        )

        if self.depth_attenuation_enabled:
            z = position[0]
            if self.depth_attenuation_mode == 'linear':
                attenuation = max(0.0, 1.0 - self.depth_attenuation_decay_rate * z)
            else:
                attenuation = np.exp(-self.depth_attenuation_decay_rate * z)
            intensity *= attenuation

        # 3D anisotropic Gaussian with configurable sharpness
        gaussian = intensity * np.exp(-dist_sq / self.gaussian_sharpness)

        # Add to volume (accumulate if overlap)
        volume[z0:z1, y0:y1, x0:x1] += gaussian


def compute_bead_statistics(beads):
    """
    Compute statistics of generated beads.

    Args:
        beads: dict with bead properties

    Returns:
        stats: dict with statistics
    """
    positions = beads['positions']
    radii = beads['radii']
    intensities = beads['intensities']

    # Volume fraction (approximate)
    total_volume = np.prod([128, 128, 128])  # Assume 128^3
    bead_volume = np.sum((4/3) * np.pi * radii**3)
    volume_fraction = bead_volume / total_volume

    # Spatial distribution (check for clustering)
    if len(positions) > 1:
        # Average nearest neighbor distance
        from scipy.spatial import cKDTree
        tree = cKDTree(positions)
        distances, _ = tree.query(positions, k=2)  # k=2 to exclude self
        nn_dist = np.mean(distances[:, 1])
    else:
        nn_dist = 0

    stats = {
        'num_beads': len(positions),
        'radius_mean': float(np.mean(radii)),
        'radius_std': float(np.std(radii)),
        'intensity_mean': float(np.mean(intensities)),
        'intensity_std': float(np.std(intensities)),
        'volume_fraction': float(volume_fraction),
        'nearest_neighbor_dist': float(nn_dist)
    }

    if 'group_ids' in beads:
        group_ids = beads['group_ids']
        unique_group_ids, counts = np.unique(group_ids, return_counts=True)
        group_counts = {
            int(group_id): int(count)
            for group_id, count in zip(unique_group_ids, counts)
        }
        stats['group_counts'] = group_counts

    return stats
