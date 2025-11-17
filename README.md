## Stereo Recording Utility

This project provides a command-line helper for capturing stereo audio snippets for a fixed set of directions and direction pairs. Each take is recorded until you press Enter, then saved as `{NAME}_{DIRECTION}.wav` inside a destination folder of your choice.

### Requirements

- Python 3.13+
- `sounddevice`, `numpy`, and `scipy` (installed automatically via `pip install -e .` or `uv pip install .`)
- An audio interface with at least two input channels

### Usage

1. Activate your virtual environment and install dependencies:

```bash
uv pip install .
```

2. Run the recorder, passing the directory where WAV files should be stored:

```bash
python main.py recordings/
```

3. Enter the shared NAME for the session when prompted. For each direction you will:
	- Press Enter to start
	- Speak or play the audio cue
	- Press Enter again to stop and save the take

You can stop the session at any time with `Ctrl+C`; completed takes remain on disk.
