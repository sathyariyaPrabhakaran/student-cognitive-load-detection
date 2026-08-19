"""MediaPipe/OpenCV feature extraction for live webcam inference."""
import math
from collections import deque


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _ratio(points, a, b, c, d):
    vertical = _dist(points[a], points[b])
    horizontal = _dist(points[c], points[d])
    return vertical / (horizontal + 1e-6)


class FaceFeatureTracker:
    """Extract stable EAR/MAR proxies and temporal blink/yawn counts.

    This module intentionally keeps computer-vision extraction separate from
    the trained classifier. If MediaPipe is unavailable, the application can
    still expose the manual /predict API.
    """
    def __init__(self):
        self.blinks = 0
        self.yawns = 0
        self.frames = 0
        self.ear_history = deque(maxlen=30)
        self.mar_history = deque(maxlen=30)

    def update(self, landmarks, width, height):
        pts = [(p.x * width, p.y * height) for p in landmarks]
        # MediaPipe Face Mesh indices for left/right eye and mouth.
        left_ear = _ratio(pts, 159, 145, 33, 133)
        right_ear = _ratio(pts, 386, 374, 362, 263)
        ear = (left_ear + right_ear) / 2
        mar = _ratio(pts, 13, 14, 61, 291)
        self.frames += 1
        self.ear_history.append(ear)
        self.mar_history.append(mar)
        if len(self.ear_history) >= 2 and self.ear_history[-2] >= 0.20 and ear < 0.20:
            self.blinks += 1
        if len(self.mar_history) >= 2 and self.mar_history[-2] <= 0.65 and mar > 0.65:
            self.yawns += 1
        return {"ear": ear, "mar": mar, "blink_count": self.blinks, "yawn_count": self.yawns}
