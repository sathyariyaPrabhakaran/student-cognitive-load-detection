# Dataset

`dataset.csv` is intentionally not fabricated or committed with synthetic labels. The project requires a real cognitive-load dataset with an official workload label.

## Canonical schema

Required target: `cognitive_load`

Supported project features:
- `ear`, `mar`, `blink_count`, `yawn_count`, `head_movement`
- `typing_speed`, `mouse_speed`, `keyboard_idle`, `mouse_idle`, `mouse_clicks`, `study_time`

Optional time column for temporal analysis: `timestamp`, `datetime`, `time`, or `created_at`.

## Reproducibility policy

Do not invent measurements or labels to increase accuracy. If a public dataset does not contain a particular signal, leave that feature missing and let the documented imputation pipeline handle it, or adapt the project feature schema to the actual modalities.

`src/dataset_adapter.py` maps only columns that genuinely exist in a prepared source CSV.

After preparing real data:

```bash
python src/train_models.py
python src/temporal_analysis.py
```

Training writes model comparison and confusion matrices to `results/` and the selected model to `models/best_model.joblib`.

Check the source dataset's license/terms before redistribution.
