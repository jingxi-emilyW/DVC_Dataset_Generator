"""
Unified output writer for saving generated samples.

Both Pipeline A (synthetic) and Pipeline B (experimental) produce the same
output format:
  - vol0.npy  (D, H, W) float32
  - vol1.npy  (D, H, W) float32
  - flow.npy  (3, D, H, W) float32
  - metadata JSON per sample (optional)
  - generation_summary.yaml + generation_config.yaml at dataset level
"""

import json
import time
import yaml
import numpy as np
from pathlib import Path


class OutputWriter:
    """Handles all file I/O for generated datasets."""

    def __init__(self, base_dir: str, save_metadata: bool = True):
        """
        Args:
            base_dir: Root output directory for the dataset
            save_metadata: Whether to save per-sample metadata JSON files
        """
        self.base_dir = Path(base_dir)
        self.save_metadata = save_metadata

    def prepare_split_dirs(self, split: str):
        """Create directory structure for a split.

        Args:
            split: Split name ('train', 'val', 'test')
        """
        split_dir = self.base_dir / split
        for subdir in ['vol0', 'vol1', 'flow']:
            (split_dir / subdir).mkdir(parents=True, exist_ok=True)
        if self.save_metadata:
            (split_dir / 'metadata').mkdir(parents=True, exist_ok=True)

    def save_sample(self, split: str, sample_idx: int,
                    vol0: np.ndarray, vol1: np.ndarray,
                    flow: np.ndarray, metadata: dict):
        """Save a single generated sample.

        Args:
            split: Split name
            sample_idx: Sample index
            vol0: Reference volume (D, H, W) float32
            vol1: Deformed volume (D, H, W) float32
            flow: Flow field (3, D, H, W) float32
            metadata: Per-sample metadata dict
        """
        split_dir = self.base_dir / split
        fname = f'sample_{sample_idx:04d}'

        np.save(split_dir / 'vol0' / f'{fname}.npy', vol0)
        np.save(split_dir / 'vol1' / f'{fname}.npy', vol1)
        np.save(split_dir / 'flow' / f'{fname}.npy', flow)

        if self.save_metadata:
            with open(split_dir / 'metadata' / f'{fname}.json', 'w') as f:
                json.dump(metadata, f, indent=2)

    def save_generation_summary(self, summary: dict):
        """Save generation summary YAML at dataset root.

        Args:
            summary: Summary dict (timing, counts, source info, etc.)
        """
        self.base_dir.mkdir(parents=True, exist_ok=True)
        with open(self.base_dir / 'generation_summary.yaml', 'w') as f:
            yaml.dump(summary, f, default_flow_style=False)

    def save_generation_config(self, config: dict):
        """Save a copy of the generation config at dataset root.

        Args:
            config: Full config dict for reproducibility
        """
        self.base_dir.mkdir(parents=True, exist_ok=True)
        with open(self.base_dir / 'generation_config.yaml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
