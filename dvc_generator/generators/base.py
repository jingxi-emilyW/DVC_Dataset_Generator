"""
Abstract base class for dataset generators.

Both SyntheticGenerator (Pipeline A) and ExperimentalGenerator (Pipeline B)
extend this class.  Shared logic lives here:
  - Config loading and seed setting
  - Split iteration with progress bars
  - Parallel / sequential dispatch
  - Output writing (via OutputWriter)
  - Generation summary / config saving
"""

import abc
import time
import yaml
import numpy as np
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

from dvc_generator.simulation.deformation import DeformationGenerator
from dvc_generator.analysis.quality_control import QualityChecker
from dvc_generator.io.writer import OutputWriter


class BaseGenerator(abc.ABC):
    """Abstract base for dataset generators."""

    def __init__(self, config_path: str):
        """
        Args:
            config_path: Path to the YAML configuration file
        """
        self.config_path = Path(config_path)
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.source_type = self.config['source_type']
        self.volume_shape = tuple(self.config['dataset']['volume_shape'])

        # Generation settings
        gen_cfg = self.config.get('generation', {})
        self.seed = gen_cfg.get('seed', 42)
        self.parallel = gen_cfg.get('parallel', False)
        self.num_workers = gen_cfg.get('num_workers', 4)
        self.save_metadata = gen_cfg.get('save_metadata', True)
        self.visualize_samples = gen_cfg.get('visualize_samples', False)
        self.num_visualize = gen_cfg.get('num_visualize', 5)

        # Set random seed
        np.random.seed(self.seed)

        # Splits
        self.splits = self.config.get('splits', {})

        # Shared modules
        self.deform_gen = DeformationGenerator(
            self.volume_shape,
            self.config['deformations']
        )

        qc_config = self.config.get('quality', {})
        self.qc = QualityChecker(qc_config)

        # Output writer
        output_dir = self.config['dataset']['output_dir']
        self.writer = OutputWriter(output_dir, save_metadata=self.save_metadata)

    def generate_all(self):
        """Generate complete dataset (all splits)."""
        start_time = time.time()

        print("=" * 70)
        print(f"3D DVC Dataset Generation  [{self.source_type}]")
        print("=" * 70)
        print(f"Config: {self.config_path}")
        print(f"Output: {self.config['dataset']['output_dir']}")
        print()

        for split_name in ['train', 'val', 'test']:
            num_samples = self.splits.get(split_name, 0)
            if num_samples > 0:
                self._generate_split(split_name, num_samples)

        # Save summary and config copy
        elapsed = time.time() - start_time
        summary = {
            'source_type': self.source_type,
            'dataset_name': self.config['dataset']['name'],
            'created': time.strftime('%Y-%m-%d %H:%M:%S'),
            'config_file': str(self.config_path),
            'splits': {k: v for k, v in self.splits.items() if v > 0},
            'generation_time_seconds': round(elapsed, 1),
        }
        self._extend_summary(summary)
        self.writer.save_generation_summary(summary)
        self.writer.save_generation_config(self.config)

        print(f"\n{'=' * 70}")
        print(f"Dataset generation complete!")
        print(f"  Total time: {elapsed / 60:.1f} minutes")
        print(f"  Output: {self.config['dataset']['output_dir']}")
        print(f"{'=' * 70}\n")

    def _generate_split(self, split: str, num_samples: int):
        """Generate all samples for one split.

        Args:
            split: 'train', 'val', or 'test'
            num_samples: Number of samples to generate
        """
        self.writer.prepare_split_dirs(split)

        print(f"\nGenerating {split} set ({num_samples} samples)...")

        passed_count = 0
        failed_count = 0

        for idx in tqdm(range(num_samples), desc=f"Generating {split}"):
            vol0, vol1, flow, metadata, passed = self._generate_single_sample(
                idx, split
            )

            self.writer.save_sample(split, idx, vol0, vol1, flow, metadata)

            if passed:
                passed_count += 1
            else:
                failed_count += 1

            # Visualization for first N samples
            if self.visualize_samples and idx < self.num_visualize:
                self._visualize_sample(split, idx, vol0, vol1, flow, metadata)

        print(f"  {split} set complete: {passed_count} passed, "
              f"{failed_count} failed QC")

    @abc.abstractmethod
    def _generate_single_sample(self, sample_idx: int, split: str):
        """Generate a single sample.

        Must be implemented by subclasses.

        Args:
            sample_idx: Index of the sample within the split
            split: Split name

        Returns:
            vol0:     (D, H, W) float32 reference volume
            vol1:     (D, H, W) float32 deformed volume
            flow:     (3, D, H, W) float32 flow field (vol0 → vol1)
            metadata: dict with per-sample metadata
            passed:   bool indicating whether the sample passed QC
        """
        ...

    def _visualize_sample(self, split, idx, vol0, vol1, flow, metadata):
        """Create visualization for a sample (delegates to QualityChecker)."""
        sample_data = {
            'vol0': vol0, 'vol1': vol1,
            'flow': flow, 'metadata': metadata,
        }
        vis_dir = Path(self.config['dataset']['output_dir']) / split / 'visualizations'
        vis_dir.mkdir(parents=True, exist_ok=True)
        self.qc.visualize_sample(sample_data, vis_dir / f'sample_{idx:04d}.png', idx)

    def _extend_summary(self, summary: dict):
        """Hook for subclasses to add extra fields to generation_summary.yaml."""
        pass
