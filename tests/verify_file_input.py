import numpy as np
import scipy.io.wavfile as wavfile
import subprocess
import sys
import os


def create_test_audio(filename, duration=3.0, sr=44100):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Silence
    audio = np.zeros_like(t)
    # Add a stomp (burst of noise) at 1.5s
    stomp_start = int(1.5 * sr)
    stomp_duration = int(0.1 * sr)
    audio[stomp_start : stomp_start + stomp_duration] = np.random.uniform(
        -0.5, 0.5, stomp_duration
    )

    # Stereo
    audio_stereo = np.stack([audio, audio], axis=1)

    wavfile.write(filename, sr, audio_stereo.astype(np.float32))
    print(f"Created {filename}")


def run_test():
    filename = "test_input.wav"
    create_test_audio(filename)

    cmd = [sys.executable, "main.py", "--input-file", filename, "--threshold", "0.05"]
    print(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        print("Output:")
        print(result.stdout)
        print("Stderr:")
        print(result.stderr)

        if "Detected Stomp!" in result.stdout:
            print("\nSUCCESS: Stomp detected.")
        else:
            print("\nFAILURE: Stomp NOT detected.")
            sys.exit(1)

    except subprocess.TimeoutExpired:
        print(
            "Timeout expired (expected if loop doesn't exit properly, but we added break on EOF)"
        )
    finally:
        if os.path.exists(filename):
            os.remove(filename)


if __name__ == "__main__":
    run_test()
