import argparse
import sys
import time
import numpy as np
import sounddevice as sd  # type: ignore
from stomp_detector import StompDetector
from classifier import FiveDirectionClassifier as Classifier
from controller import KeyboardController as Controller
from file_stream import FileStream


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device", type=int, default=None, help="Audio input device index"
    )
    parser.add_argument(
        "--input-file", type=str, default=None, help="Path to input audio file"
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio devices and exit",
    )
    parser.add_argument(
        "--select", action="store_true", help="Interactively select an audio device"
    )
    return parser.parse_args()


def list_audio_devices():
    """Prints available audio input devices."""
    print("\nAvailable Audio Input Devices:")
    print(sd.query_devices())
    print("\n")


def select_audio_device():
    """Interactively selects an audio device."""
    list_audio_devices()
    while True:
        try:
            selection = input(
                "Enter the ID of the input device to use (or press Enter for default): "
            )
            if not selection:
                return None
            device_id = int(selection)
            # Validate device ID (simple check)
            try:
                sd.query_devices(device_id, "input")
                return device_id
            except Exception:
                print("Invalid device ID. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")


def calibrate(stream, step_frames, duration=3.0):
    """Calibrates the noise floor and returns the max noise RMS."""
    print(f"Calibrating background noise for {duration} seconds...")
    print("Please remain silent...")

    max_energy = 0.0
    start_time = time.time()

    while time.time() - start_time < duration:
        chunk, overflow = stream.read(step_frames)
        if overflow:
            print("Warning: Audio overflow during calibration", file=sys.stderr)

        # Calculate RMS energy of the chunk
        # Ensure mono
        if chunk.ndim > 1:
            y = np.mean(chunk, axis=1)
        else:
            y = chunk

        rms = np.sqrt(np.mean(y**2))
        if rms > max_energy:
            max_energy = rms

    print(f"Calibration complete. Max noise RMS: {max_energy:.5f}")
    return max_energy


def main():
    args = parse_args()

    if args.list_devices:
        list_audio_devices()
        return

    print("Initializing Stomp Detection System...")

    device_id = args.device
    if args.select:
        device_id = select_audio_device()

    # Determine sample rate
    try:
        if device_id is not None:
            device_info = sd.query_devices(device_id, "input")
            sr = int(device_info["default_samplerate"])
        else:
            # If using default device, query it
            device_info = sd.query_devices(kind="input")
            sr = int(device_info["default_samplerate"])
    except Exception as e:
        print(
            f"Warning: Could not determine default sample rate: {e}",
            file=sys.stderr,
        )
        sr = 48000

    print(f"Device: {device_id if device_id is not None else 'Default'}, SR: {sr}")

    # Parameters
    window_ms = 200
    step_ms = 100

    window_frames = int((window_ms / 1000.0) * sr)
    step_frames = int((step_ms / 1000.0) * sr)
    channels = 2  # Assuming stereo

    # Initialize components
    try:
        # We will set the threshold after calibration
        classifier = Classifier()
        controller = Controller()

        # Buffer to hold the rolling window
        audio_buffer = np.zeros((window_frames, channels), dtype=np.float32)

        print("Listening... Press Ctrl+C to stop.")

        # Open stream
        if args.input_file:
            stream_ctx = FileStream(args.input_file, step_frames)
            # Update sr to match file if not overridden
            if args.sr is None:
                sr = stream_ctx.sr
                window_frames = int((window_ms / 1000.0) * sr)
                audio_buffer = np.zeros((window_frames, channels), dtype=np.float32)

        else:
            stream_ctx = sd.InputStream(
                samplerate=sr,
                blocksize=step_frames,
                device=device_id,
                channels=channels,
                dtype="float32",
            )

        detector = StompDetector(sr=sr, energy_threshold=7.0)

        with stream_ctx as stream:
            if args.input_file:
                print(
                    "Using file input. Skipping calibration and using default/provided threshold."
                )
            else:
                noise_level = calibrate(stream, step_frames)
                # Initialize noise level slightly higher to avoid immediate triggers
                detector.noise_level = max(noise_level, 0.001)
                print(f"Setting initial noise level to: {detector.noise_level:.5f}")

            while True:
                try:
                    # Check for end of file
                    if args.input_file and stream.finished:
                        print("End of file reached.")
                        break

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
                        controller.press(direction)

                except KeyboardInterrupt:
                    print("\nStopping...")
                    break
    finally:
        print("System stopped.")


if __name__ == "__main__":
    main()
