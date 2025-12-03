"""CLI utility for collecting paired stereo recordings with sounddevice."""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path
from typing import Iterable, List

import numpy as np
import sounddevice as sd  # type: ignore
from scipy.io import wavfile


BASE_DIRECTIONS = ["up", "down", "left", "right"]
# PAIR_DIRECTIONS = ["".join(pair) for pair in itertools.combinations(BASE_DIRECTIONS, 2)]
ALL_DIRECTIONS = BASE_DIRECTIONS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record stereo WAV files for each direction and paired direction."
    )
    parser.add_argument(
        "output_dir", type=Path, help="Directory where the WAV files will be written."
    )
    parser.add_argument(
        "--samplerate",
        type=int,
        default=44100,
        help="Sampling rate to use while recording (default: 44100 Hz).",
    )
    parser.add_argument(
        "--blocksize",
        type=int,
        default=0,
        help="Optional blocksize passed to sounddevice.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional sounddevice input device identifier.",
    )
    return parser.parse_args()


def prompt_for_name() -> str:
    while True:
        name = input("Enter a shared NAME for this recording session: ").strip()
        if name:
            return name
        print("NAME cannot be empty. Please try again.")


def record_until_enter(
    samplerate: int, blocksize: int, device: str | None, channels: int = 2
) -> np.ndarray:
    frames: List[np.ndarray] = []

    def callback(indata, _frames, _time, status):
        if status:
            print(f"Recorder warning: {status}", file=sys.stderr)
        frames.append(indata.copy())

    # Keep the stream open while the user decides when to stop the take.
    with sd.InputStream(
        samplerate=samplerate,
        blocksize=blocksize,
        device=device,
        channels=channels,
        dtype="float32",
        callback=callback,
    ):
        input("Recording... press Enter to stop.")

    if not frames:
        return np.empty((0, channels), dtype="float32")

    return np.concatenate(frames, axis=0)


def iterate_directions() -> Iterable[str]:
    yield from ALL_DIRECTIONS


def save_wav(path: Path, samplerate: int, data: np.ndarray) -> None:
    normalized = np.clip(data, -1.0, 1.0)
    wavfile.write(path, samplerate, normalized.astype(np.float32))


def record_direction(
    name: str,
    direction: str,
    output_dir: Path,
    samplerate: int,
    blocksize: int,
    device: str | None,
) -> None:
    print(f"\nPreparing to record '{direction}' for NAME '{name}'.")
    input("Press Enter when you're ready to start recording.")
    take = record_until_enter(samplerate=samplerate, blocksize=blocksize, device=device)

    if take.size == 0:
        print("No audio captured; skipping file write.")
        return

    filepath = output_dir / f"{name}_{direction}.wav"
    save_wav(filepath, samplerate, take)
    duration = take.shape[0] / samplerate
    print(f"Saved {filepath} ({duration:.1f}s).")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Stereo recording session initialized.")
    print(f"Files will be saved to: {output_dir}\n")

    name = prompt_for_name()

    try:
        for direction in iterate_directions():
            record_direction(
                name=name,
                direction=direction,
                output_dir=output_dir,
                samplerate=args.samplerate,
                blocksize=args.blocksize,
                device=args.device,
            )
    except KeyboardInterrupt:
        print("\nSession interrupted by user. Any completed takes have been saved.")


if __name__ == "__main__":
    main()
