import numpy as np
import pytest
from stomp_detector import StompDetector

def test_adaptive_noise_floor():
    sr = 1000
    detector = StompDetector(
        sr=sr, win_ms=200, energy_threshold=1000.0, alpha=0.5
    )
    # Initial noise level is 0.001
    assert detector.noise_level == 0.001

    # Feed silence (zeros)
    # RMS of zeros is 0.
    # New noise = 0.5 * 0.001 + 0.5 * 0 = 0.0005
    detector.detect(np.zeros(200))
    assert detector.noise_level == 0.0005

    # Feed constant noise
    # RMS of 0.1 is 0.1
    # New noise = 0.5 * 0.0005 + 0.5 * 0.1 = 0.00025 + 0.05 = 0.05025
    noise_chunk = np.ones(200) * 0.1
    detector.detect(noise_chunk)
    assert detector.noise_level == pytest.approx(0.05025, abs=0.01)

def test_detection_thresholding():
    sr = 1000
    detector = StompDetector(
        sr=sr, win_ms=200, energy_threshold=2.0, alpha=0.0 # Disable adaptation for this test
    )
    detector.noise_level = 0.1
    
    # Threshold is 0.1 * 2.0 = 0.2

    # Case 1: Signal below threshold
    # Pulse 0.15
    audio_low = np.zeros(200)
    audio_low[90:110] = 0.15
    stomps = detector.detect(audio_low)
    assert len(stomps) == 0

    # Case 2: Signal above threshold
    # Pulse 0.25
    audio_high = np.zeros(200)
    audio_high[90:110] = 0.25
    stomps = detector.detect(audio_high)
    assert len(stomps) == 1
