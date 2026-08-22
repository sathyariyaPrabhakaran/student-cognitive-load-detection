# Cost-Aware Adaptive Diabetic Retinopathy Screening

A research-oriented ML system for reducing unnecessary deep-model computation in diabetic-retinopathy screening.

## Core research question

Can a lightweight retinal-image model decide which cases actually need a more expensive expert model, while preserving screening sensitivity?

This is **not** a claim that the system replaces clinical diagnosis. It is a research prototype for computationally efficient screening.

## Proposed architecture

```text
Fundus image
    |
    v
Lightweight MobileNetV3-Small
    |
    +---- confidence / entropy / margin
    +---- image quality features
    |
    v
ML routing model (Logistic Regression)
    |
    +---- easy case ---------> lightweight prediction
    |
    +---- uncertain/hard ----> EfficientNet-B0 expert
                                      |
                                      v
                              final prediction
```

The router is trained to predict when escalation is useful rather than using a hard-coded confidence rule. The operating threshold is selected on a separate calibration split under a minimum sensitivity constraint.

## What is measured

- sensitivity / recall
- specificity
- macro F1 and balanced accuracy
- percentage of cases escalated to the expert
- lightweight-model latency
- expert-model latency
- estimated inference time saved
- parameter counts
- routing confusion matrix
- performance versus escalation-rate curve

The project deliberately does **not** hard-code an accuracy or cost-saving percentage. Results must come from training on a real dataset.

## Dataset format

Use a public diabetic-retinopathy image dataset after complying with its license/terms. Organize it as:

```text
data/retina/
  train/
    0_no_dr/
    1_mild/
    2_moderate/
    3_severe/
    4_proliferative/
  ...
```

The loader also supports fewer classes if the chosen dataset is organized differently.

## Run

Install:

```bash
python -m pip install -r medical_cost_aware_dr/requirements.txt
```

Train and evaluate:

```bash
python medical_cost_aware_dr/src/train_adaptive.py --data-dir data/retina
```

Outputs are written to `medical_cost_aware_dr/results/` and trained weights to `medical_cost_aware_dr/models/`.

## Research discipline

The project must be evaluated against at least:

1. expert-only inference
2. lightweight-only inference
3. fixed confidence-threshold routing
4. learned adaptive routing

The final report should compare performance at multiple sensitivity constraints and should report compute/latency trade-offs rather than only accuracy.
