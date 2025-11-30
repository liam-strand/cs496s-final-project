import argparse
import sys
import numpy as np
import sounddevice as sd  # type: ignore
from stomp_detector import StompDetector
from classifier import DummyClassifier
from controller import InputController


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device", type=int, default=None, help="Audio input device index"
    )
    parser.add_argument("--sr", type=int, default=44100, help="Sampling rate")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Energy threshold for stomp detection",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("Initializing Stomp Detection System...")
    print(f"Device: {args.device}, SR: {args.sr}, Threshold: {args.threshold}")

    # Parameters
    sr = args.sr
    window_ms = 200
    step_ms = 50

    window_frames = int((window_ms / 1000.0) * sr)
    step_frames = int((step_ms / 1000.0) * sr)
    channels = 2  # Assuming stereo

    # Initialize components
    try:
        detector = StompDetector(sr=sr, energy_threshold=args.threshold)
        classifier = DummyClassifier()
        controller = InputController(verbose=True)

        # Buffer to hold the rolling window
        audio_buffer = np.zeros((window_frames, channels), dtype=np.float32)

        print("Listening... Press Ctrl+C to stop.")

        # Open stream

        with sd.InputStream(
            samplerate=sr,
            blocksize=step_frames,
            device=args.device,
            channels=channels,
            dtype="float32",
        ) as stream:
            while True:
                try:
                    # Read 'step' frames
                    chunk, overflow = stream.read(step_frames)

                    if overflow:
                        print("Warning: Audio overflow", file=sys.stderr)

                    # Update rolling buffer
                    audio_buffer = np.roll(audio_buffer, -step_frames, axis=0)
                    audio_buffer[-step_frames:] = chunk

                    # Detect on the full window
                    stomps = detector.detect(audio_buffer)

                    for stomp in stomps:
                        direction = classifier.classify(stomp)
                        print(f"Detected Stomp! Classified as: {direction}")
                        controller.press(direction)

                except KeyboardInterrupt:
                    print("\nStopping...")
                    break
                except Exception as e:
                    print(f"Error in loop: {e}", file=sys.stderr)
                    break

    except Exception as e:
        print(f"Initialization error: {e}", file=sys.stderr)
    finally:
        print("System stopped.")


if __name__ == "__main__":
    main()
