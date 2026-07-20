# Topological Deep Learning Based Image Segmentation

A controlled comparison of three segmentation systems on the ISBI 2012 electron-microscopy membrane dataset, built on a single shared U-Net backbone. The project measures the trade-off between segmentation quality, topological correctness, and computational cost when adding a topology-aware term to the training objective.

## Overview

Three models are trained and evaluated under one protocol, so the loss function is the only variable that changes:

| Model | Loss |
|---|---|
| Baseline | BCE + soft Dice |
| PH | BCE + soft Dice + Persistent Homology loss |
| DECT | BCE + soft Dice + Differentiable Euler Characteristic Transform loss |

PH and DECT do not encode identical information. PH tracks connected components and holes through persistence, while DECT measures directional Euler-characteristic curves. The comparison is therefore presented as an empirical quality–efficiency trade-off, not as proof that one descriptor universally replaces the other.

## Repository Structure

```text
topological-image-segmentation/
├── README.md
├── REPORTPROGRESS.md
├── requirements.txt
├── configs/
│   └── isbi.yaml
├── data/
│   ├── raw/isbi/
│   │   ├── train-volume.tif
│   │   └── train-labels.tif
│   ├── processed/
│   └── splits/
├── outputs/
│   ├── baseline/
│   ├── ph/
│   ├── dect/
│   ├── comparisons/
│   └── logs/
├── scripts/
│   ├── prepare_isbi.py
│   ├── train.py
│   ├── evaluate.py
│   ├── benchmark_topology.py
│   ├── compare.py
│   ├── compare_predictions.py
│   └── run_all.py
├── src/
│   ├── config.py
│   ├── data/
│   │   ├── augmentations.py
│   │   └── isbi_dataset.py
│   ├── models/
│   │   └── unet.py
│   ├── losses/
│   │   ├── segmentation.py
│   │   ├── ph_loss.py
│   │   └── dect_loss.py
│   ├── metrics/
│   │   ├── segmentation.py
│   │   └── topology.py
│   ├── engine/
│   │   ├── trainer.py
│   │   └── evaluator.py
│   └── utils/
│       ├── device.py
│       ├── seed.py
│       ├── timing.py
│       └── visualization.py
└── tests/
    ├── test_dataset.py
    ├── test_dect.py
    └── test_losses.py
```

## Requirements

- Python 3.11 or newer
- An NVIDIA GPU with a CUDA-enabled PyTorch build (the project is configured for GPU training and was developed on an RTX 3050 with 4 GB VRAM)

Python dependencies are listed in `requirements.txt`:

```text
numpy>=1.26,<3
scipy>=1.11
pandas>=2.1
PyYAML>=6.0
Pillow>=10.0
tifffile>=2023.9
matplotlib>=3.8
seaborn>=0.13
scikit-image>=0.22
scikit-learn>=1.3
tqdm>=4.66
torch>=2.2
torchvision>=0.17
gudhi>=3.9
torch-topological
pytest>=8.0
```

## Installation

Create and activate a virtual environment, then install the dependencies.

### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Windows (PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Verify CUDA

Confirm that your PyTorch build can actually use the GPU before training:

```powershell
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA runtime:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

You should see `CUDA available: True` and your GPU name. If it prints `False`, install a CUDA-enabled PyTorch wheel using the command from the official PyTorch installation selector. Note that the CUDA version reported by `nvidia-smi` is the driver's maximum supported version — your PyTorch wheel can be built against an earlier CUDA runtime and still work.

## Data

Place the ISBI 2012 challenge stacks here:

```text
data/raw/isbi/train-volume.tif
data/raw/isbi/train-labels.tif
```

Both must be 3D TIFF stacks with identical `[slices, height, width]` shapes. Prepare normalized arrays and reproducible slice-level splits:

```powershell
python -m scripts.prepare_isbi --config configs/isbi.yaml
```

This writes:

```text
data/processed/isbi/images.npy
data/processed/isbi/labels.npy
data/splits/isbi_split.json
```

The split is performed by whole slices rather than random patches to reduce leakage between training and evaluation.

## Usage

All commands are shown in the module form (`python -m scripts.<name>`), which works reliably on Windows. The `--device` flag is optional; omit it to auto-select CUDA.

### Train

```powershell
python -m scripts.train --config configs/isbi.yaml --method baseline --device cuda:0
python -m scripts.train --config configs/isbi.yaml --method dect --device cuda:0
python -m scripts.train --config configs/isbi.yaml --method ph --device cuda:0
```

Each run writes `history.csv`, `last.pt`, and `best.pt` under `outputs/<method>/`.

### Evaluate

```powershell
python -m scripts.evaluate --config configs/isbi.yaml --method baseline --device cuda:0
python -m scripts.evaluate --config configs/isbi.yaml --method dect --device cuda:0
python -m scripts.evaluate --config configs/isbi.yaml --method ph --device cuda:0
```

Evaluation reports overlap (Dice, IoU), partition (ARI, VoI), boundary (ASSD), Betti-number, and Euler-characteristic metrics, plus inference timing. Metrics are saved to `outputs/<method>/evaluation/metrics.json`, and per-slice `.npy` predictions and diagnostic `.png` figures are saved under `outputs/<method>/evaluation/predictions/`.

### Benchmark topology loss timing

```powershell
python -m scripts.benchmark_topology --config configs/isbi.yaml --method dect --device cuda:0
python -m scripts.benchmark_topology --config configs/isbi.yaml --method ph --device cuda:0
```

This measures a warmed-up topology-loss forward and backward pass and records peak VRAM usage.

### Compare models

Aggregate training histories and evaluation metrics into comparison plots and a CSV table:

```powershell
python -m scripts.compare --output-root outputs
```

Generate side-by-side prediction panels across all three models for each test slice:

```powershell
python -m scripts.compare_predictions --output-root outputs --processed-directory data/processed/isbi --split-file data/splits/isbi_split.json --threshold 0.5
```

Outputs are written to `outputs/comparisons/`.

### Full pipeline

To run preparation, training, evaluation, and the DECT benchmark in sequence:

```powershell
python -m scripts.run_all --config configs/isbi.yaml
```

Include the PH attempt with `--include-ph`. PH failures are isolated so baseline and DECT results remain usable.

## Testing

Run the unit tests (dataset, segmentation loss, DECT gradient, and PH backend) before launching a long training job:

```powershell
pytest -q
```

## Configuration

All hyperparameters live in `configs/isbi.yaml`. Key settings for a 4 GB VRAM GPU:

- `training.batch_size: 4` with `gradient_accumulation_steps: 2` gives an effective batch size of 8.
- `model.base_channels: 24` keeps the U-Net within memory.
- `dect.direction_chunk_size: 4` processes ECT directions in chunks to avoid materializing all directions at once.

If you still hit out-of-memory errors, reduce to `batch_size: 2`, `gradient_accumulation_steps: 4`, `base_channels: 16`, and `direction_chunk_size: 2`.

## Metrics

| Family | Metrics | Direction |
|---|---|---|
| Overlap | Dice, IoU | higher is better |
| Partition | ARI, VoI | ARI higher, VoI lower |
| Topology | Betti-number error, Euler error | lower is better |
| Boundary | ASSD | lower is better |
| Cost | per-epoch time, inference time, peak VRAM | lower is better |

## Notes and Limitations

- **Device split.** The GPU handles the U-Net forward/backward passes, segmentation loss, DECT loss, and AMP. The CPU handles TIFF/NumPy loading, augmentation, DataLoader workers, and the SciPy/scikit-image evaluation metrics. This division is intentional — efficient GPU use means keeping the GPU fed via pinned-memory, non-blocking transfers, not forcing every operation onto CUDA.
- **PH is a surrogate.** The PH loss uses torch-topological with persistence-sorted matching. This is differentiable but is not equivalent to optimal persistence-diagram assignment or the exact critical-point matching of Hu et al. Report it as a PH surrogate.
- **DECT weight.** After the loss-scaling correction (ECT curves normalized by patch area, with no area-squared division), the topology weight is not numerically comparable to earlier configurations. Start at 0.05 and select a final weight using validation Dice plus topology error — never the test set.
- **Reproducibility.** The configured seed initializes Python, NumPy, and PyTorch. Use `deterministic: false` for throughput and `true` for strict reproducibility checks (slower). Exact bitwise reproducibility still depends on platform, accelerator, and CUDA build.

## Acknowledgements

This work builds directly on three papers:

- Hu et al., *Topology-Preserving Deep Image Segmentation*, NeurIPS 2019.
- Nadimpalli et al., *Euler Characteristic Transform Based Topological Loss for Reconstructing 3D Images from Single 2D Slices*, CVPRW 2023.
- Röell and Rieck, *Differentiable Euler Characteristic Transforms for Shape Classification*, 2023.

The ISBI 2012 dataset is from Arganda-Carreras et al., *Crowdsourcing the Creation of Image Segmentation Algorithms for Connectomics*, Frontiers in Neuroanatomy, 2015.
