
## `REPORTPROGRESS.md`

```markdown
# Progress Report

## Objective

Compare a conventional binary segmentation objective with two
topology-aware objectives on ISBI image data:

- BCE plus soft Dice.
- BCE plus soft Dice and persistent homology.
- BCE plus soft Dice and differentiable ECT.

## Completed

- Added per-slice robust intensity normalization.
- Added deterministic train, validation, and test splits.
- Added random patch sampling with foreground-biased sampling.
- Added paired spatial and intensity augmentation.
- Added a 2D U-Net baseline.
- Added BCE and soft-Dice segmentation losses.
- Added a differentiable ECT implementation.
- Added foreground and background ECT matching.
- Added configurable topology warmup.
- Added mixed-precision baseline training.
- Added checkpoints, early stopping, and CSV histories.
- Added overlap, partition, surface, Betti, and Euler metrics.
- Added prediction visualizations.
- Added topology forward/backward benchmarks.
- Added dataset, segmentation-loss, and DECT gradient tests.

## Deliberately incomplete

The persistent-homology backend remains unbound. It fails during construction
until its adapter is implemented against the exact installed GUDHI API.

This is intentional. Replacing the training loss with a NumPy or
scikit-image Betti calculation would silently remove gradient propagation and
would not constitute persistent-homology training.

## Validation checklist

Before experiments:

1. Confirm the two TIFF stacks have identical `[N, H, W]` shapes.
2. Run data preparation.
3. Run `pytest -q`.
4. Train and evaluate the baseline.
5. Train and evaluate DECT.
6. Verify the DECT loss produces finite, nonzero gradients.
7. Bind and test the PH backend before attempting PH experiments.
8. Run each method with multiple seeds for final comparisons.

## Remaining research work

- Implement and test the GUDHI version adapter.
- Add PH unit tests with known components and holes.
- Record package, CUDA, driver, and hardware versions.
- Run multiple independent seeds.
- Report means, standard deviations, and paired comparisons.
- Profile memory as well as runtime.
- Inspect topology failures qualitatively.
- Check sensitivity to topology weight, patch size, and threshold.
