# Dataset, training and results

## Public benchmark

The reproducible benchmark used by `scripts/train_openmatb.py` is the OpenMATB workload dataset from MInD Laboratory:

https://github.com/MInD-Laboratory/Measuring_Workload_Dynamics_in_OpenMATB

The repository provides performance-derived workload features and H/M/L experimental conditions. The training script downloads the published CSV at runtime rather than copying third-party data into this repository.

## Reproduction

```bash
python scripts/train_openmatb.py
```

Outputs:

- `results/openmatb/model_comparison.csv`
- `results/openmatb/confusion_matrix_*.png`
- `results/openmatb/run_metadata.json`
- `models/openmatb/best_model.joblib`
- `models/openmatb/feature_columns.joblib`

## Important methodological distinction

The OpenMATB benchmark is suitable for validating the ML workload-classification pipeline, but its performance/task features are **not** identical to webcam-derived EAR/MAR/head-motion features. Therefore benchmark accuracy must not be presented as webcam accuracy. The webcam system should be evaluated separately after collecting consenting webcam sessions with matching ground-truth workload labels.

No synthetic observations or fabricated accuracy values are included.
