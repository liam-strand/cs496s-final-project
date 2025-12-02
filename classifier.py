from typing import Protocol
import numpy as np
import random
from onnxruntime import InferenceSession
from features import extract_all_features_with_xcorr


class StompClassifier(Protocol):
    """Interface for stomp classifiers."""

    def classify(self, stomp: np.ndarray) -> str:
        """Classify a stomp segment into a direction."""
        ...


class DummyClassifier:
    """A dummy classifier that returns random directions."""

    def __init__(self):
        self.directions = ["up", "down", "left", "right"]

    def classify(self, stomp: np.ndarray) -> str:
        if stomp.ndim > 1 and stomp.shape[1] >= 2:
            left_energy = np.sum(stomp[:, 0] ** 2)
            right_energy = np.sum(stomp[:, 1] ** 2)

            ratio = left_energy / (right_energy + 1e-6)
            if ratio > 1.5:
                return "left"
            elif ratio < 0.66:
                return "right"

        return random.choice(self.directions)


class MLPClassifier:
    def run_model(self, features: np.ndarray) -> int:
        pred_ort = self.sess.run(None, {"input": features.astype(np.float32)})[0]
        return pred_ort

    def moves(self, idx: int) -> str: ...

    def classify(self, stomp: np.ndarray) -> str:
        features = extract_all_features_with_xcorr(stomp)
        reshaped_features = np.asarray(features).reshape(1, -1)
        pred_ort = self.run_model(reshaped_features)
        return self.moves(pred_ort[0])


class LeftRightClassifier(MLPClassifier):
    """A classifier that returns left or right based on energy ratio."""

    SUPER_BASIC_MOVES = ["left", "right"]

    def __init__(self):
        with open("models/mlp_left_right.onnx", "rb") as f:
            onx = f.read()
        self.sess = InferenceSession(onx, providers=["CPUExecutionProvider"])

    def moves(self, idx: int) -> str:
        return self.SUPER_BASIC_MOVES[idx]


class FiveDirectionClassifier(MLPClassifier):
    BASIC_MOVES = ["center", "left", "right", "up", "down"]

    def __init__(self):
        with open("models/mlp_five_directions.onnx", "rb") as f:
            onx = f.read()
        self.sess = InferenceSession(onx, providers=["CPUExecutionProvider"])

    def moves(self, idx: int) -> str:
        return self.BASIC_MOVES[idx]


class ElevenDirectionClassifier(MLPClassifier):
    MOVES = [
        "center",
        "left",
        "right",
        "up",
        "down",
        "downleft",
        "downright",
        "upleft",
        "upright",
        "updown",
        "leftright",
    ]

    def __init__(self):
        with open("models/mlp_all_directions.onnx", "rb") as f:
            onx = f.read()
        self.sess = InferenceSession(onx, providers=["CPUExecutionProvider"])

    def moves(self, idx: int) -> str:
        return self.MOVES[idx]
