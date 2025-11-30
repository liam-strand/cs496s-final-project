from typing import Protocol
import numpy as np
import random


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
