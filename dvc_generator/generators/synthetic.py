"""
Pipeline A: Synthetic confocal particle dataset generator.

Generates 3D volumetric image pairs with ground-truth deformation fields
by procedurally placing fluorescent beads, applying deformations, and
simulating confocal microscopy imaging effects.

Steps per sample:
  1. Generate deformation field
  2. Generate bead positions (vol0)
  3. Render vol0
  4. Warp beads → new positions (vol1)
  5. Render vol1
  6. Apply imaging effects (PSF, noise, photobleaching)
  7. Quality check
"""

import numpy as np

from dvc_generator.generators.base import BaseGenerator
from dvc_generator.simulation.deformation import compute_flow_statistics
from dvc_generator.simulation.warping import ForwardWarper, compute_warp_statistics
from dvc_generator.simulation.beads import (
    BeadGenerator, BeadRenderer, compute_bead_statistics
)
from dvc_generator.simulation.effects import (
    ImagingSimulator, compute_imaging_statistics
)


class SyntheticGenerator(BaseGenerator):
    """Generate synthetic confocal particle datasets."""

    def __init__(self, config_path: str):
        super().__init__(config_path)

        # Pipeline A specific modules
        self.bead_gen = BeadGenerator(self.volume_shape,
                                      self.config['particles'])
        self.renderer = BeadRenderer(self.volume_shape,
                                      self.config['particles'])
        self.warper = ForwardWarper(self.volume_shape,
                                    self.config['particles'])
        self.imaging = ImagingSimulator(self.config['imaging'])

    def _generate_single_sample(self, sample_idx, split):
        """Generate a single synthetic sample.

        Returns:
            vol0, vol1, flow, metadata, passed
        """
        # 1. Generate deformation field
        flow, deform_meta = self.deform_gen.generate()

        # 2. Generate beads for vol0
        beads_vol0 = self.bead_gen.generate_beads()

        # 3. Render vol0 (clean)
        vol0_clean = self.renderer.render(beads_vol0)

        # 4. Warp beads to create vol1
        beads_vol1, num_visible = self.warper.warp_beads(beads_vol0, flow)

        # 5. Render vol1 (clean)
        vol1_clean = self.renderer.render(beads_vol1)

        # 6. Apply imaging effects
        vol0, imaging_meta0 = self.imaging.apply_imaging_effects(
            vol0_clean, apply_photobleaching=False
        )
        vol1, imaging_meta1 = self.imaging.apply_imaging_effects(
            vol1_clean, apply_photobleaching=True
        )

        # 7. Compute statistics
        flow_stats = compute_flow_statistics(flow)
        bead_stats = compute_bead_statistics(beads_vol0)
        warp_stats = compute_warp_statistics(beads_vol0, beads_vol1)

        # 8. Compile metadata
        metadata = {
            'sample_idx': sample_idx,
            'split': split,
            'source': 'synthetic',
            'deform_type': deform_meta['type'],
            'deform_params': deform_meta,
            'bead_stats': bead_stats,
            'warp_stats': warp_stats,
            'flow_stats': flow_stats,
            'imaging_vol0': imaging_meta0,
            'imaging_vol1': imaging_meta1,
        }

        # 9. Quality check
        sample_data = {
            'vol0': vol0, 'vol1': vol1,
            'flow': flow, 'metadata': metadata,
        }
        passed, issues = self.qc.check_sample(sample_data)

        if not passed:
            print(f"  Warning: Sample {sample_idx} failed QC: {issues}")

        return vol0, vol1, flow, metadata, passed
