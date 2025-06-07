# input/asl_detector.py
import cv2
import json
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

class ASLDetector:
    def __init__(self,
                 letter_model_path: str,
                 letter_map_path: str,
                 digit_model_path: str,
                 digit_map_path: str,
                 conf_threshold: float = 0.6):
        # Load letter (A-Z) model
        self.letter_model = load_model(letter_model_path)
        with open(letter_map_path, 'r') as f:
            letter_indices = json.load(f)
        self.idx2letter = {int(v): k for k, v in letter_indices.items()}

        # Load digit (0-9) model
        self.digit_model = load_model(digit_model_path)
        with open(digit_map_path, 'r') as f:
            digit_indices = json.load(f)
        self.idx2digit = {int(v): k for k, v in digit_indices.items()}

        self.conf_threshold = conf_threshold

        # MediaPipe hands setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

    def _preprocess_letter(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        if not results.multi_hand_landmarks:
            return None, None
        lm = results.multi_hand_landmarks[0].landmark
        coords = np.array([[p.x, p.y, getattr(p, 'z', 0)]
                           for p in lm], dtype=np.float32).flatten().reshape(1, -1)
        return coords, results.multi_hand_landmarks[0]

    def _preprocess_digit(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        if not results.multi_hand_landmarks:
            return None, None
        lm = results.multi_hand_landmarks[0].landmark
        # Only use x,y coordinates for digits (42 dimensions)
        coords = np.array([[p.x, p.y] for p in lm], dtype=np.float32).flatten().reshape(1, -1)
        return coords, results.multi_hand_landmarks[0]

    def recognize_letter(self, frame):
        """
        Recognize ASL letter (A-Z).
        Returns: (label:str or None, confidence:float)
        """
        coords, landmarks = self._preprocess_letter(frame)
        if coords is None:
            return None, 0.0
        
        # Draw landmarks
        self.mp_draw.draw_landmarks(
            frame,
            landmarks,
            self.mp_hands.HAND_CONNECTIONS
        )

        proba = self.letter_model.predict(coords, verbose=0)[0]
        idx = int(np.argmax(proba))
        conf = proba[idx]
        if conf < self.conf_threshold:
            return None, conf
        return self.idx2letter[idx], conf

    def recognize_digit(self, frame):
        """
        Recognize ASL digit (0-9).
        Returns: (label:str or None, confidence:float)
        """
        coords, landmarks = self._preprocess_digit(frame)
        if coords is None:
            return None, 0.0
        
        # Draw landmarks
        self.mp_draw.draw_landmarks(
            frame,
            landmarks,
            self.mp_hands.HAND_CONNECTIONS
        )

        proba = self.digit_model.predict(coords, verbose=0)[0]
        idx = int(np.argmax(proba))
        conf = proba[idx]
        if conf < self.conf_threshold:
            return None, conf
        return self.idx2digit[idx], conf