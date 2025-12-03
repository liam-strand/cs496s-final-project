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
        energy_threshold: float = 5.0,
        alpha: float = 0.05,
    ):
        """
        Args:
            sr: Sampling rate.
            win_ms: Length of audio segment to return around each stomp (ms).
            frame_ms: Analysis frame length (ms).
            hop_ms: Analysis hop length (ms).
            energy_threshold: Multiplier for noise floor to trigger detection.
            alpha: Smoothing factor for noise floor update (0 < alpha < 1).
            cooldown_ms: Minimum time between stomps (ms).
        """
        self.sr = sr
        self.win_ms = win_ms
        self.frame_ms = frame_ms
        self.hop_ms = hop_ms
        self.energy_threshold = energy_threshold
        self.alpha = alpha

        # State
        self.noise_level = 0.001  # Initial small value
        self.cooldown = 0

        # Derived parameters
        self.frame_len = int((frame_ms / 1000.0) * sr)
        self.hop_len = int((hop_ms / 1000.0) * sr)
        self.half_win = int((win_ms / 1000.0) * sr // 2)

    def detect(self, audio: np.ndarray) -> list[np.ndarray]:
        """Detect stomps in the provided audio chunk."""
        if self.cooldown > 0:
            self.cooldown -= 1
            return []

        # print(self.noise_level)

        # Ensure mono for energy calculation
        if audio.ndim > 1:
            y = np.mean(audio, axis=1)
        else:
            y = audio

        if len(y) < self.frame_len:
            return []

        # Calculate indices for the middle 100ms
        mid_start = self.half_win // 2
        mid_end = mid_start + self.half_win

        # Extract middle segment
        y_mid = y[mid_start:mid_end]

        # Compute RMS on the middle segment
        if len(y_mid) < self.frame_len:
            return []

        energy = librosa.feature.rms(
            y=y_mid, frame_length=self.frame_len, hop_length=self.hop_len, center=True
        )[0]

        # Use max energy in the segment for detection
        segment_energy = np.max(energy)
        # Use mean energy for noise floor update
        avg_energy = np.mean(energy)

        # Check for detection
        is_stomp = False
        if segment_energy > self.noise_level * self.energy_threshold:
            is_stomp = True
            self.cooldown = 2

        if is_stomp:
            # Resample only on detection
            resampled_audio = librosa.resample(
                audio, orig_sr=self.sr, target_sr=16000, axis=0
            )
            return [resampled_audio]
        else:
            self.noise_level = (1 - self.alpha) * self.noise_level + self.alpha * avg_energy
            return []
