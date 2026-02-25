#!/usr/bin/env python3
"""
Unified CLI entry point for 3D DVC dataset generation.

Routes to the appropriate pipeline based on ``source_type`` in the YAML config:
  - synthetic        -> SyntheticGenerator  (Pipeline A)
  - crop_from_image  -> ExperimentalGenerator (Pipeline B)

Usage:
    generate-dataset --config configs/synthetic_128_v2.yaml
    python -m dvc_generator.cli --config configs/exp_Franck_128.yaml
"""

import sys
import argparse
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(
        description='Generate 3D DVC datasets (synthetic or from experimental images)'
    )
    parser.add_argument(
        '--config', type=str, required=True,
        help='Path to YAML configuration file'
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: config file not found: {config_path}")
        # List available configs relative to cwd
        configs_dir = Path.cwd() / 'configs'
        if configs_dir.exists():
            print("Available configs:")
            for cfg in sorted(configs_dir.glob('*.yaml')):
                print(f"  - configs/{cfg.name}")
        return 1

    # Read source_type to decide which generator to use
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    source_type = config.get('source_type')
    if source_type is None:
        print("Error: config file must contain a 'source_type' field "
              "('synthetic' or 'crop_from_image')")
        return 1

    if source_type == 'synthetic':
        from dvc_generator.generators.synthetic import SyntheticGenerator
        generator = SyntheticGenerator(str(config_path))
    elif source_type == 'crop_from_image':
        from dvc_generator.generators.experimental import ExperimentalGenerator
        generator = ExperimentalGenerator(str(config_path))
    else:
        print(f"Error: unknown source_type '{source_type}'. "
              f"Expected 'synthetic' or 'crop_from_image'.")
        return 1

    generator.generate_all()
    return 0


if __name__ == '__main__':
    sys.exit(main())
