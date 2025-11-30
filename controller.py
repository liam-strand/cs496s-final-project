import pyautogui
import time


class InputController:
    """Controls keyboard input based on detected directions."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
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
            "center": [],  # No action? Or maybe space?
        }

    def press(self, direction: str):
        """Press the key(s) corresponding to the direction."""
        direction = direction.lower()
        keys = self.key_map.get(direction)

        if keys:
            if self.verbose:
                print(f"InputController: Pressing {keys} for '{direction}'")
            for key in keys:
                pyautogui.keyDown(key)

            time.sleep(0.05)

            for key in keys:
                pyautogui.keyUp(key)
        else:
            if self.verbose:
                print(f"InputController: Unknown direction {direction}")
