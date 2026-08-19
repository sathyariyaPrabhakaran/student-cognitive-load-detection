# Adaptive Temporal Behavioral Fusion

An AI system for estimating **workload state from time-evolving behavioural signals**. The system is designed around session-level patterns rather than a single snapshot and produces an inferred workload state, confidence, and temporal trajectory.

## What is different about this implementation?

Instead of treating every observation as an independent row, the pipeline models how behaviour changes across a session:

```text
SWELL-KW behavioural signals
        ↓
Data validation + numeric feature selection
        ↓
Rolling-window statistics
        ↓
Change / trend features
        ↓
Interaction + fatigue dynamics
        ↓
Adaptive evidence fusion
        ↓
Participant-aware ML comparison
        ↓
Workload-state inference
        ↓
Confidence + temporal trajectory
```

### Modelling components

- **Random Forest** — transparent ensemble baseline
- **Adaptive Temporal Gradient Boosting** — proposed tabular model
- **RBF-SVM** — nonlinear comparator
- Rolling mean / standard deviation
- Short-term deltas and trends
- Interaction-intensity and fatigue-pressure features
- Adaptive modality/evidence weighting
- **Stratified group cross-validation**
- **Participant-level holdout testing** to reduce identity leakage
- Macro precision, recall and F1
- Confusion matrices
- Saved model + feature schema + experiment metadata

## Dataset

The project uses the public **SWELL Knowledge Work Dataset (SWELL-KW)**, deposited by Koldijk, Sappelli, Verberne, Neerincx and Kraaij. It contains minute-level behavioural and multimodal measurements from 25 participants performing knowledge-work tasks under neutral, interruption and time-pressure conditions. The original study also collected subjective task-load and mental-effort ratings.

Source: DOI `10.17026/DANS-X55-69ZP`.

The repository deliberately downloads the approximately 4 MB preprocessed behavioural feature table rather than the approximately 7 GB raw archive. The dataset is licensed CC-BY-NC-SA-4.0. citeturn5search0turn3search0

**Important label semantics:** the current reproducible pipeline predicts the experimentally recorded **workload condition** (`relaxed`, `baseline`, `elevated`, `high`). This is used as an operational cognitive-load proxy. It is not a clinical or psychological diagnosis, and the project does not claim that the condition label is identical to a person's subjective cognitive load.

## Setup

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Download the public behavioural table:

```bash
python scripts/setup_swell_kw.py
```

Train and evaluate:

```bash
python src/train_models.py
```

Start the application:

```bash
python app.py
```

Then open `http://127.0.0.1:5000`.

## Outputs

Training creates:

```text
models/best_model.joblib
models/feature_columns.joblib
results/model_comparison.csv
results/run_metadata.json
results/confusion_matrix_*.png
```

No accuracy or F1 value is hard-coded. Metrics shown by the application are generated from the real dataset and the current training run.

## Data privacy / deployment

The core inference path works with recorded/tabular session data. A camera is **not required** for the system to operate. Any future live sensor connector is treated as a separate data-acquisition layer.
