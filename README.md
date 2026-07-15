# VisionEntropy

Entropy-guided image segmentation framework.

This repository is set up for the first VisionEntropy vertical slice:

- shared Python package for datasets, entropy measures, representations, segmentation, evaluation, and pipelines
- reproducible YAML experiment configs
- testable entropy core
- React admin app for configuring runs and inspecting results
- output folders for experiment artifacts

## Quick Start

### Python framework

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pytest
```

### Admin web app

```powershell
cd apps/admin-web
npm install
npm run dev
```

## Repository Layout

```text
apps/
  admin-web/              React admin console
configs/
  experiments/            Reproducible experiment YAML files
data/
  raw/                    Downloaded or original datasets
  processed/              Prepared local data
notebooks/                Research notebooks
outputs/
  runs/                   Per-run experiment outputs
  figures/                Shared figures
  tables/                 Shared tables
  reports/                HTML/Markdown reports
src/visionentropy/
  datasets/               Dataset interfaces and loaders
  entropy/                Entropy measures and maps
  evaluation/             Segmentation metrics
  features/               Feature extraction
  pipeline/               End-to-end orchestration
  preprocessing/          Image transforms
  representations/        Pixels, color spaces, superpixels, graphs
  segmentation/           Classical and learned segmenters
tests/                    Unit and integration tests
```

## First Milestone

VE-0.1 focuses on a classical vertical slice:

1. SyntheticShapes and skimage example datasets
2. RGB, grayscale, Lab, local entropy maps, and SLIC superpixels
3. Shannon, Renyi, and Tsallis entropy
4. Thresholding and clustering baselines
5. IoU, Dice, and pixel accuracy
6. React admin workflow and later API/runner integration
