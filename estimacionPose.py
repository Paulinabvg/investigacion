import cv2
import mediapipe as mp
from .config import MEDIAPIPE_CONFIG

class PoseEstimator:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(**MEDIAPIPE_CONFIG)
        self.mp_draw = mp.solutions.drawing_utils

    def process_frame(self, frame):
        """Procesa un frame y devuelve los landmarks detectados."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        return results.pose_landmarks

    def draw_landmarks(self, frame, landmarks):
        """Dibuja los landmarks en el frame."""
        if landmarks:
            self.mp_draw.draw_landmarks(
                frame, landmarks, self.mp_pose.POSE_CONNECTIONS)
        return frame