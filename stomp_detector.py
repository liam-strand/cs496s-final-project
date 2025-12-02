import numpy as np
import librosa


class StompDetector:
    """Real-time stomp detector."""

    def __init__(
        self,
        sr: int = 48000,
        win_ms: int = 200,
        frame_ms: int = 20,
        hop_ms: int = 10,
        energy_threshold: float = 0.3,
        min_stomp_sep_ms: int = 250,
    ):
        """
        Args:
            sr: Sampling rate.
            win_ms: Length of audio segment to return around each stomp (ms).
            frame_ms: Analysis frame length (ms).
            hop_ms: Analysis hop length (ms).
            energy_threshold: Minimum normalized energy for a peak.
            min_stomp_sep_ms: Minimum time between stomps (ms).
        """
        self.sr = sr
        self.win_ms = win_ms
        self.frame_ms = frame_ms
        self.hop_ms = hop_ms
        self.energy_threshold = energy_threshold
        self.min_stomp_sep_ms = min_stomp_sep_ms

        self.last_stomp_time = 0.0

        # Derived parameters
        self.frame_len = int((frame_ms / 1000.0) * sr)
        self.hop_len = int((hop_ms / 1000.0) * sr)
        self.half_win = int((win_ms / 1000.0) * sr // 2)

    def detect(self, audio: np.ndarray) -> list[np.ndarray]:
        """Detect stomps in the provided audio chunk."""
        # Ensure mono for energy calculation
        if audio.ndim > 1:
            y = np.mean(audio, axis=1)
        else:
            y = audio

        if len(y) < self.frame_len:
            return []

        # Compute RMS
        energy = librosa.feature.rms(
            y=y, frame_length=self.frame_len, hop_length=self.hop_len, center=True
        )[0]

        resampled_audio = librosa.resample(
            audio, orig_sr=self.sr, target_sr=16000, axis=0
        )

        if np.max(energy) >= self.energy_threshold:
            return [resampled_audio]
        else:
            return []
