# Student Cognitive Load Detection

A research-oriented cognitive-load classification project using behavioral and facial features.

## Baseline vs. improved approach

The public reference baseline uses 11 engineered features and a Random Forest classifier for Low/Medium/High cognitive-load prediction.

This repository does not copy the reference repository. It recreates the modeling idea and adds a reproducible model-comparison pipeline:

1. Random Forest — baseline
2. HistGradientBoosting — proposed tabular ML model
3. RBF-SVM — comparison model
4. Stratified cross-validation
5. Accuracy, precision, recall, macro F1
6. Confusion matrices
7. Best-model selection using validation macro F1

No performance number is claimed until the actual dataset is supplied and the pipeline is executed.

## Dataset schema

Place `dataset.csv` inside `data/`.

Expected columns:

- ear
- mar
- blink_count
- yawn_count
- head_movement
- typing_speed
- mouse_speed
- keyboard_idle
- mouse_idle
- mouse_clicks
- study_time
- cognitive_load

## Run

```bash
python -m pip install -r requirements.txt
python src/train_models.py
```

The script creates model-comparison metrics, confusion matrices, and the selected model under `results/` and `models/`.

## Attribution

The feature schema and baseline concept were informed by publicly available cognitive-load detection projects. The implementation here is independently written and will be extended with feature engineering, temporal analysis, and model comparison.
