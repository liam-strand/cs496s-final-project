import numpy as np
import librosa
from scipy.signal import find_peaks
import time


class StompDetector:
    """Real-time stomp detector."""

    def __init__(
        self,
        sr: int = 44100,
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
        rms = librosa.feature.rms(
            y=y, frame_length=self.frame_len, hop_length=self.hop_len, center=True
        )[0]

        # Normalize
        max_rms = np.max(rms)
        if max_rms > 1e-4:  # Avoid amplifying silence
            energy = rms / max_rms
        else:
            energy = rms  # It's all zeros/noise

        # Find peaks
        peaks, _ = find_peaks(energy, height=self.energy_threshold)

        detected_stomps = []
        current_time = time.time()

        # Calculate time of each peak relative to NOW
        # audio ends at current_time
        for peak_frame in peaks:
            center_sample = peak_frame * self.hop_len

            # Skip if peak is too close to edges (cannot extract full window)
            start = center_sample - self.half_win
            end = center_sample + self.half_win

            if start < 0 or end > len(y):
                continue

            # Calculate absolute time of this peak
            samples_from_end = len(y) - center_sample
            peak_time = current_time - (samples_from_end / self.sr)

            # Check separation
            if (peak_time - self.last_stomp_time) * 1000.0 >= self.min_stomp_sep_ms:
                # It's a new stomp
                self.last_stomp_time = peak_time

                # Extract segment
                segment = audio[start:end]
                detected_stomps.append(segment)

        return detected_stomps
