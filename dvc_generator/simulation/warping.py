"""
Volume warping functions for DVC dataset generation.

Provides both bead-based forward warping (Pipeline A) and dense volume
warping (Pipeline B) with two strategies:
  - backward_swap: exact, via scipy.ndimage.map_coordinates
  - forward:       approximate, via trilinear splatting

No PyTorch dependency.
"""

import numpy as np
from scipy.ndimage import map_coordinates


# ---------------------------------------------------------------------------
# Dense volume warping (used by Pipeline B, also available to Pipeline A)
# ---------------------------------------------------------------------------

def backward_warp(volume: np.ndarray, backward_flow: np.ndarray) -> np.ndarray:
    """
    Backward warp a volume using a backward flow field.

    For each output position q, sample the input at position (q - flow(q)):
        output[z, y, x] = input[z - F[0,z,y,x],
                                y - F[1,z,y,x],
                                x - F[2,z,y,x]]

    This is mathematically exact because the flow is defined at output-grid
    positions, so we directly know *where* to look in the source volume.

    Uses scipy.ndimage.map_coordinates with order=1 (trilinear) and
    mode='nearest' (border clamping).

    Args:
        volume:        (D, H, W) source volume, float32
        backward_flow: (3, D, H, W) backward flow field [dz, dy, dx]

    Returns:
        warped: (D, H, W) warped volume, float32
    """
    D, H, W = volume.shape
    grid_z, grid_y, grid_x = np.meshgrid(
        np.arange(D, dtype=np.float32),
        np.arange(H, dtype=np.float32),
        np.arange(W, dtype=np.float32),
        indexing='ij'
    )

    sample_z = grid_z - backward_flow[0]
    sample_y = grid_y - backward_flow[1]
    sample_x = grid_x - backward_flow[2]

    coords = np.array([sample_z.ravel(), sample_y.ravel(), sample_x.ravel()])
    warped = map_coordinates(volume.astype(np.float64), coords,
                             order=1, mode='nearest')
    return warped.reshape(D, H, W).astype(np.float32)


def forward_warp(volume: np.ndarray, forward_flow: np.ndarray) -> np.ndarray:
    """
    Forward warp a volume using trilinear splatting.

    For each source voxel at position p, its intensity is distributed to
    the 8 neighboring target voxels at position (p + flow(p)).

    Forward warping can leave holes (unmapped output voxels) where no
    source voxel maps to. These holes are left as zero.

    Args:
        volume:       (D, H, W) source volume, float32
        forward_flow: (3, D, H, W) forward flow field [dz, dy, dx]

    Returns:
        warped: (D, H, W) warped volume, float32
    """
    D, H, W = volume.shape
    warped = np.zeros((D, H, W), dtype=np.float32)
    weight = np.zeros((D, H, W), dtype=np.float32)

    # Only splat non-background voxels (fast path)
    nz_z, nz_y, nz_x = np.where(volume > 0.01)

    for k in range(len(nz_z)):
        z, y, x = int(nz_z[k]), int(nz_y[k]), int(nz_x[k])
        intensity = volume[z, y, x]

        tz = z + forward_flow[0, z, y, x]
        ty = y + forward_flow[1, z, y, x]
        tx = x + forward_flow[2, z, y, x]

        z0, y0, x0 = int(np.floor(tz)), int(np.floor(ty)), int(np.floor(tx))
        z1, y1, x1 = z0 + 1, y0 + 1, x0 + 1

        if z0 < 0 or z1 >= D or y0 < 0 or y1 >= H or x0 < 0 or x1 >= W:
            continue

        dz = tz - z0
        dy = ty - y0
        dx = tx - x0

        warped[z0, y0, x0] += (1-dz)*(1-dy)*(1-dx) * intensity
        warped[z0, y0, x1] += (1-dz)*(1-dy)*dx     * intensity
        warped[z0, y1, x0] += (1-dz)*dy*(1-dx)     * intensity
        warped[z0, y1, x1] += (1-dz)*dy*dx          * intensity
        warped[z1, y0, x0] += dz*(1-dy)*(1-dx)      * intensity
        warped[z1, y0, x1] += dz*(1-dy)*dx           * intensity
        warped[z1, y1, x0] += dz*dy*(1-dx)           * intensity
        warped[z1, y1, x1] += dz*dy*dx                * intensity

        weight[z0, y0, x0] += (1-dz)*(1-dy)*(1-dx)
        weight[z0, y0, x1] += (1-dz)*(1-dy)*dx
        weight[z0, y1, x0] += (1-dz)*dy*(1-dx)
        weight[z0, y1, x1] += (1-dz)*dy*dx
        weight[z1, y0, x0] += dz*(1-dy)*(1-dx)
        weight[z1, y0, x1] += dz*(1-dy)*dx
        weight[z1, y1, x0] += dz*dy*(1-dx)
        weight[z1, y1, x1] += dz*dy*dx

    mask = weight > 0
    warped[mask] /= weight[mask]
    return warped


# ---------------------------------------------------------------------------
# Bead-based forward warping (used by Pipeline A synthetic generation)
# ---------------------------------------------------------------------------

class ForwardWarper:
    """Forward warp beads using flow field sampling + trilinear splatting."""

    def __init__(self, volume_shape, particles_config=None):
        """
        Args:
            volume_shape: Tuple (D, H, W)
            particles_config: Configuration dict for particles (optional)
        """
        self.volume_shape = np.array(volume_shape)
        if particles_config is not None:
            self.margin = particles_config.get('margin_voxels', 1)
        else:
            self.margin = 1

    def warp_beads(self, beads, flow):
        """
        Warp beads using forward warping with flow field sampling.

        Args:
            beads: dict with 'positions', 'radii', 'intensities'
            flow: (3, D, H, W) displacement field

        Returns:
            warped_beads: dict with warped bead positions
            num_visible: number of beads still in FOV after warping
        """
        positions = beads['positions']  # (N, 3)
        radii = beads['radii']
        intensities = beads['intensities']

        # Sample flow at bead positions
        displacements = self._sample_flow_at_positions(flow, positions)

        # Compute new positions
        new_positions = positions + displacements

        # Check which beads are still visible
        in_bounds = np.all(
            (new_positions >= self.margin) &
            (new_positions < self.volume_shape - self.margin),
            axis=1
        )

        warped_beads = {
            'positions': new_positions[in_bounds],
            'radii': radii[in_bounds],
            'intensities': intensities[in_bounds],
            'num_beads': np.sum(in_bounds),
            'original_num_beads': len(positions)
        }

        return warped_beads, np.sum(in_bounds)

    def _sample_flow_at_positions(self, flow, positions):
        """
        Sample flow field at arbitrary positions using trilinear interpolation.

        Args:
            flow: (3, D, H, W) flow field
            positions: (N, 3) array of positions

        Returns:
            displacements: (N, 3) array of sampled displacements
        """
        N = len(positions)
        displacements = np.zeros((N, 3), dtype=np.float32)

        coords = positions.T  # (3, N)
        for i in range(3):
            displacements[:, i] = map_coordinates(
                flow[i], coords, order=1, mode='nearest'
            )

        return displacements


def compute_warp_statistics(beads_original, beads_warped):
    """
    Compute statistics about the bead warping process.

    Args:
        beads_original: original bead dict
        beads_warped: warped bead dict

    Returns:
        stats: dict with warping statistics
    """
    original_count = beads_original['num_beads']
    warped_count = beads_warped['num_beads']

    visible_ratio = warped_count / original_count if original_count > 0 else 0

    if warped_count > 0:
        orig_pos = beads_original['positions'][:warped_count]
        warp_pos = beads_warped['positions']
        displacements = np.linalg.norm(warp_pos - orig_pos, axis=1)
        disp_mean = float(np.mean(displacements))
        disp_std = float(np.std(displacements))
        disp_max = float(np.max(displacements))
    else:
        disp_mean = disp_std = disp_max = 0

    stats = {
        'original_count': int(original_count),
        'warped_count': int(warped_count),
        'visible_ratio': float(visible_ratio),
        'displacement_mean': disp_mean,
        'displacement_std': disp_std,
        'displacement_max': disp_max
    }

    return stats
