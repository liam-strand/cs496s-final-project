import pyautogui
import time
from typing import Protocol


class InputController(Protocol):
    """Interface for controllers."""

    def press(self, direction: str):
        """Press the key(s) corresponding to the direction."""
        ...


class KeyboardController:
    """Controls keyboard input based on detected directions."""

    def __init__(self, verbose: bool = True, cooldown: float = 0.3):
        self.verbose = verbose
        self.cooldown = cooldown
        self.last_press_time = 0.0
        # Safety feature: fail-safe if mouse is in corner
        pyautogui.FAILSAFE = True

        self.key_map = {
            "left": ["left"],
            "right": ["right"],
            "up": ["up"],
            "down": ["down"],
            "upleft": ["up", "left"],
            "upright": ["up", "right"],
            "downleft": ["down", "left"],
            "downright": ["down", "right"],
            "leftright": ["left", "right"],
            "updown": ["up", "down"],
            "center": [],  # No action
        }

    def press(self, direction: str):
        """Press the key(s) corresponding to the direction."""
        current_time = time.time()
        if current_time - self.last_press_time < self.cooldown:
            return

        self.last_press_time = current_time

        direction = direction.lower()
        keys = self.key_map.get(direction)

        if keys:
            if self.verbose:
                print(f"InputController: Pressing {keys} for '{direction}'")

            pyautogui.press(keys)
        else:
            if self.verbose:
                print(f"InputController: Unknown direction {direction}")


class DummyController:
    """Controls keyboard input based on detected directions."""

    def __init__(self, cooldown: float = 0.3):
        self.cooldown = cooldown
        self.last_press_time = 0.0

    def press(self, direction: str):
        """Press the key(s) corresponding to the direction."""
        current_time = time.time()
        if current_time - self.last_press_time < self.cooldown:
            return

        self.last_press_time = current_time
        print(f"DummyController: Pressing '{direction}'")
