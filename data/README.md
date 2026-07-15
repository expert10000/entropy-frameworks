# VisionEntropy Data Directory

Large datasets should live here locally, but they should not be committed to Git.

## Recommended Layout

```text
data/
  raw/
    oxford_iiit_pet/
      images/
      annotations/
    bsds500/
    pascal_voc/
  processed/
    synthetic_shapes/
    cache/
```

## Dataset Modes

- `synthetic_shapes`: generated on demand, no files required.
- `skimage_examples`: built into `scikit-image`, no download required.
- `oxford_iiit_pet`: user-managed local dataset, optionally downloaded by a future runner.

Keep source archives, extracted images, masks, and heavyweight intermediate files out of the repository.
