import cv2
import numpy as np


class MediaPipeDetector:
    """
    Hand-presence detector using MediaPipe Hands.

    Returns a 2-element float32 array [p_no_gesture, p_gesture], compatible with
    the Queue smoothing and threshold logic used by the CNN detector path so both
    backends can share the same downstream inference code.

    Detection is binary per frame; temporal smoothing is delegated to the caller's
    Queue (moving average, median, or EWMA) as with the CNN path.
    """

    def __init__(self, min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        import mediapipe as mp
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def detect(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Args:
            frame_bgr: (H, W, 3) BGR uint8 numpy array
        Returns:
            (2,) float32: [p_no_gesture, p_gesture]
        """
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)
        if result.multi_hand_landmarks:
            return np.array([0.0, 1.0], dtype=np.float32)
        return np.array([1.0, 0.0], dtype=np.float32)

    def close(self):
        self.hands.close()
