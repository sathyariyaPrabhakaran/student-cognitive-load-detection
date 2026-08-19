"""Collect consenting, labelled webcam feature windows for project-specific validation.

Usage:
  python scripts/collect_webcam_data.py --label low --minutes 5

The script stores numeric facial/behavioral features only; it does not save
video frames. Labels must be supplied by the participant/research protocol.
"""
import argparse, csv, time
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "webcam_sessions.csv"

try:
    import mediapipe as mp
except ImportError:
    raise SystemExit("Install dependencies first: pip install -r requirements.txt")

FEATURES = ["timestamp", "ear", "mar", "blink_count", "yawn_count", "head_movement", "typing_speed", "mouse_speed", "keyboard_idle", "mouse_idle", "mouse_clicks", "study_time", "cognitive_load"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--label", choices=["low", "medium", "high"], required=True)
    p.add_argument("--minutes", type=float, default=5)
    args = p.parse_args()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    exists = OUT.exists()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Could not open webcam.")
    face = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
    start = time.time(); previous_nose = None; blink = yawn = 0
    with OUT.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FEATURES)
        if not exists: writer.writeheader()
        try:
            while time.time() - start < args.minutes * 60:
                ok, frame = cap.read()
                if not ok: break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = face.process(rgb)
                ear = mar = head = np.nan
                if result.multi_face_landmarks:
                    lm = result.multi_face_landmarks[0].landmark
                    def d(a,b): return np.hypot(lm[a].x-lm[b].x, lm[a].y-lm[b].y)
                    # MediaPipe Face Mesh eye/mouth landmarks.
                    ear = (d(159,145) + d(158,144)) / max(2*d(33,133), 1e-9)
                    mar = (d(13,14) + d(82,312)) / max(2*d(78,308), 1e-9)
                    nose = np.array([lm[1].x, lm[1].y])
                    if previous_nose is not None: head = float(np.linalg.norm(nose-previous_nose))
                    previous_nose = nose
                elapsed = time.time() - start
                writer.writerow({"timestamp": time.time(), "ear": ear, "mar": mar, "blink_count": blink, "yawn_count": yawn, "head_movement": head, "typing_speed": np.nan, "mouse_speed": np.nan, "keyboard_idle": np.nan, "mouse_idle": np.nan, "mouse_clicks": np.nan, "study_time": elapsed, "cognitive_load": args.label})
                cv2.putText(frame, f"Label: {args.label} | {elapsed/60:.1f} min | ESC stop", (10,30), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,0), 2)
                cv2.imshow("Cognitive Load Data Collection", frame)
                if cv2.waitKey(1) & 0xFF == 27: break
        finally:
            cap.release(); face.close(); cv2.destroyAllWindows()
    print(f"Saved labelled session to {OUT}")

if __name__ == "__main__": main()
