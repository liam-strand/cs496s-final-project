import pyautogui
import time


def main():
    print("Starting to press left arrow every second...")
    print("Press Ctrl+C to stop")

    try:
        while True:
            pyautogui.press("left")
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopped pressing left arrow")


if __name__ == "__main__":
    main()
