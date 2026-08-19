# Webcam-specific validation

The repository now includes `scripts/collect_webcam_data.py` for collecting project-specific labelled feature windows from consenting participants.

## Protocol

Collect multiple sessions for each label (low, medium, high) using the same task protocol. Keep participants' identity private and do not commit raw video or personally identifying data.

Example:

```bash
python scripts/collect_webcam_data.py --label low --minutes 5
python scripts/collect_webcam_data.py --label medium --minutes 5
python scripts/collect_webcam_data.py --label high --minutes 5
```

The collector stores numeric feature rows and labels in `data/webcam_sessions.csv`; it does not save video frames.

## Validation requirement

A webcam-specific accuracy/F1 score can only be reported after real labelled sessions have been collected. Until then, the repository deliberately does not claim a webcam accuracy number.

For a stronger experiment, collect multiple participants and split train/test by participant to reduce subject leakage.
