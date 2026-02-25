"""
Pipeline B: Dataset generation from experimental images.

Loads real 3D volumetric images (TIF / NumPy), extracts sub-volumes,
preprocesses them, then applies synthetic deformations to create
training pairs with ground-truth flow fields.

Two warp strategies are supported (configured via ``warp_mode``):

backward_swap (default, mathematically exact)
  1. Generate flow F, treat as backward flow.
  2. Backward warp: vol_warped[q] = vol_original[q - F(q)]
  3. Swap roles: vol0 = vol_warped, vol1 = vol_original
  4. Saved flow = -F  (forward direction vol0 → vol1)

forward (approximate, trilinear splatting)
  1. Generate flow F, treat as forward flow.
  2. Forward warp: vol1 = forward_warp(vol0, F)
  3. Saved flow = F
"""

import numpy as np

from dvc_generator.generators.base import BaseGenerator
from dvc_generator.simulation.deformation import compute_flow_statistics
from dvc_generator.simulation.warping import backward_warp, forward_warp
from dvc_generator.io.extractor import VolumeExtractor
from dvc_generator.io.preprocessor import Preprocessor


class ExperimentalGenerator(BaseGenerator):
    """Generate datasets from experimental volumetric images."""

    def __init__(self, config_path: str):
        super().__init__(config_path)

        self.warp_mode = self.config.get('warp_mode', 'backward_swap')

        # These are populated during generate_all() before split generation
        self._volumes = []
        self._extraction_metadata = []

    def generate_all(self):
        """Override to add loading / extraction / preprocessing before splits."""
        # Step 1: Load source volumes
        print("=" * 70)
        print(f"3D DVC Dataset Generation  [{self.source_type}]")
        print("=" * 70)
        print(f"Config: {self.config_path}")
        print(f"Warp mode: {self.warp_mode}")
        print()

        raw_volumes, file_names = self._load_source_volumes()

        # Step 2: Extract sub-volumes
        print("\nExtracting sub-volumes...")
        print("-" * 70)
        extractor = VolumeExtractor(self.config)
        extracted, extract_meta = extractor.extract_volumes(raw_volumes, file_names)
        print()

        # Step 3: Preprocess
        print("Preprocessing volumes...")
        print("-" * 70)
        preprocessor = Preprocessor(self.config)
        processed = preprocessor.process_batch(extracted)
        print(f"  Preprocessed {len(processed)} volumes")
        print()

        # Allocate volumes to splits
        self._allocate_volumes(processed, extract_meta)

        # Now call the parent generate_all() flow (which iterates splits)
        # but we skip the header print since we already printed it
        import time
        start_time = time.time()

        for split_name in ['train', 'val', 'test']:
            num_samples = self.splits.get(split_name, 0)
            if num_samples > 0:
                self._generate_split(split_name, num_samples)

        elapsed = time.time() - start_time
        summary = {
            'source_type': self.source_type,
            'dataset_name': self.config['dataset']['name'],
            'created': time.strftime('%Y-%m-%d %H:%M:%S'),
            'config_file': str(self.config_path),
            'warp_mode': self.warp_mode,
            'num_source_files': len(file_names),
            'source_files': file_names,
            'num_extracted_volumes': len(processed),
            'splits': {k: v for k, v in self.splits.items() if v > 0},
            'generation_time_seconds': round(elapsed, 1),
        }
        self.writer.save_generation_summary(summary)
        self.writer.save_generation_config(self.config)

        print(f"\n{'=' * 70}")
        print(f"Dataset generation complete!")
        print(f"  Total time: {elapsed / 60:.1f} minutes")
        print(f"  Output: {self.config['dataset']['output_dir']}")
        print(f"{'=' * 70}\n")

    def _load_source_volumes(self):
        """Load source volumetric data (TIF or NumPy).

        Returns:
            volumes: list of np.ndarray
            file_names: list of str
        """
        input_cfg = self.config.get('input', {})
        input_format = input_cfg.get('format', 'tif')

        if input_format in ('tif', 'tiff'):
            return self._load_tif_volumes()
        elif input_format == 'numpy':
            return self._load_numpy_volumes()
        else:
            raise ValueError(
                f"Unsupported input format: '{input_format}'. "
                f"Supported: tif, numpy"
            )

    def _load_tif_volumes(self):
        """Load volumes from TIF files."""
        from dvc_generator.io.tif_loader import TifLoader

        print("Loading TIF files...")
        print("-" * 70)

        loader = TifLoader(self.config)
        summary = loader.get_summary()

        print(f"  Total files: {summary['num_total_files']}")
        print(f"  Valid files: {summary['num_valid_files']}")
        print()

        valid_files = loader.filter_valid_files()
        volumes = []
        file_names = []

        for file_path in valid_files:
            vol = loader.load_file(file_path)
            volumes.append(vol)
            file_names.append(file_path.name)
            print(f"  Loaded: {file_path.name} -- shape {vol.shape}")

        print(f"  Loaded {len(volumes)} volumes total")
        return volumes, file_names

    def _load_numpy_volumes(self):
        """Load volumes from .npy files."""
        from pathlib import Path

        input_cfg = self.config['input']
        data_dir = Path(input_cfg['raw_data_dir'])
        pattern = input_cfg.get('file_pattern', '*.npy')

        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        npy_files = sorted(data_dir.glob(pattern))
        if not npy_files:
            raise ValueError(f"No .npy files found in {data_dir}")

        print(f"Loading NumPy files from {data_dir}...")
        print("-" * 70)

        volumes = []
        file_names = []
        min_size = self.config.get('input', {}).get('min_volume_size', [0, 0, 0])

        for fpath in npy_files:
            vol = np.load(fpath).astype(np.float32)
            if vol.ndim != 3:
                print(f"  Skipping {fpath.name}: expected 3D, got {vol.ndim}D")
                continue
            if any(vol.shape[i] < min_size[i] for i in range(3)):
                print(f"  Skipping {fpath.name}: shape {vol.shape} < {min_size}")
                continue
            volumes.append(vol)
            file_names.append(fpath.name)
            print(f"  Loaded: {fpath.name} -- shape {vol.shape}")

        if not volumes:
            raise ValueError(f"No valid .npy files found in {data_dir}")

        print(f"  Loaded {len(volumes)} volumes total")
        return volumes, file_names

    def _allocate_volumes(self, volumes, extraction_metadata):
        """Allocate extracted volumes to splits.

        Shuffles volumes randomly and assigns them to train/val/test in order.
        Stores per-split slices in self._split_volumes.
        """
        total_requested = sum(
            self.splits.get(s, 0) for s in ['train', 'val', 'test']
        )
        total_available = len(volumes)

        if total_requested > total_available:
            raise ValueError(
                f"Requested {total_requested} samples but only "
                f"{total_available} volumes extracted. "
                f"Increase extraction.total_volumes or reduce split counts."
            )

        indices = np.random.permutation(len(volumes))
        self._split_volumes = {}
        idx = 0

        for split_name in ['train', 'val', 'test']:
            n = self.splits.get(split_name, 0)
            split_indices = indices[idx:idx + n]
            self._split_volumes[split_name] = {
                'volumes': [volumes[i] for i in split_indices],
                'metadata': [extraction_metadata[i] for i in split_indices],
            }
            idx += n

    def _generate_single_sample(self, sample_idx, split):
        """Generate a single experimental sample.

        Returns:
            vol0, vol1, flow, metadata, passed
        """
        split_data = self._split_volumes[split]
        vol_original = split_data['volumes'][sample_idx]
        extract_meta = split_data['metadata'][sample_idx]

        # Generate deformation
        flow_raw, deform_meta = self.deform_gen.generate()

        # Apply warp strategy
        if self.warp_mode == 'backward_swap':
            backward_flow = flow_raw
            vol_warped = backward_warp(vol_original, backward_flow)
            vol0 = vol_warped
            vol1 = vol_original
            saved_flow = -backward_flow  # forward flow: vol0 → vol1
        else:
            vol0 = vol_original
            vol1 = forward_warp(vol_original, flow_raw)
            saved_flow = flow_raw

        flow_stats = compute_flow_statistics(saved_flow)

        metadata = {
            'sample_idx': sample_idx,
            'split': split,
            'source': 'experimental',
            'warp_mode': self.warp_mode,
            'extraction': extract_meta,
            'deform_type': deform_meta['type'],
            'deform_params': deform_meta,
            'flow_stats': flow_stats,
            'volume_stats': {
                'vol0_mean': float(np.mean(vol0)),
                'vol0_std': float(np.std(vol0)),
                'vol0_min': float(np.min(vol0)),
                'vol0_max': float(np.max(vol0)),
                'vol1_mean': float(np.mean(vol1)),
                'vol1_std': float(np.std(vol1)),
            },
        }

        # Quality check
        sample_data = {
            'vol0': vol0, 'vol1': vol1,
            'flow': saved_flow, 'metadata': metadata,
        }
        passed, issues = self.qc.check_sample(sample_data)
        if not passed:
            print(f"  Warning: Sample {sample_idx} failed QC: {issues}")

        return vol0, vol1, saved_flow, metadata, passed
