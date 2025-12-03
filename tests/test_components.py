import numpy as np
from stomp_detector import StompDetector
from controller import KeyboardController


def test_detection():
    sr = 1000
    # Initialize with a known noise level for testing if possible, or rely on default
    detector = StompDetector(
        sr=sr, win_ms=200, energy_threshold=2.0, cooldown_ms=100
    )
    # Manually set noise level to avoid adaptation during this specific test
    detector.noise_level = 0.1

    # Create synthetic signal: Silence -> Pulse -> Silence
    t = np.linspace(0, 1, sr)
    audio = np.zeros_like(t)

    # Add pulse at 0.5s (middle of 1s buffer, but we pass chunks usually)
    # Let's pass a chunk that has the pulse in the middle
    # 200ms chunk
    audio_chunk = np.zeros(200)
    # Pulse in middle 100ms (50-150)
    audio_chunk[90:110] = 1.0 
    
    # Detect
    stomps = detector.detect(audio_chunk)
    assert len(stomps) == 1
    
    # Check if stomp is centered roughly (resampled length check)
    # 16000 Hz * 0.2s = 3200 samples
    assert len(stomps[0]) == int(16000 * 0.2)


def test_deduplication(mocker):
    mock_time = mocker.patch("time.time")
    mock_time.return_value = 100.0

    sr = 1000
    detector = StompDetector(sr=sr, win_ms=200, cooldown_ms=200, energy_threshold=2.0)
    detector.noise_level = 0.1

    # Create signal with pulse
    audio_pulse = np.zeros(200)
    audio_pulse[90:110] = 1.0

    # First pass: detect
    stomps1 = detector.detect(audio_pulse)
    assert len(stomps1) == 1

    # Immediate second pass: should be ignored due to cooldown
    # Advance time slightly
    mock_time.return_value = 100.1 # +100ms
    stomps2 = detector.detect(audio_pulse)
    assert len(stomps2) == 0

    # Third pass: after cooldown
    mock_time.return_value = 100.3 # +300ms (from start)
    stomps3 = detector.detect(audio_pulse)
    assert len(stomps3) == 1


def test_controller(mocker):
    from controller import KeyboardController

    # Mock pyautogui.press
    mock_press = mocker.patch("controller.pyautogui.press")
    mock_time = mocker.patch("time.time", return_value=1000.0)
    
    controller = KeyboardController(verbose=False)

    # Test single direction
    controller.press("left")
    mock_press.assert_called_with(["left"])

    # Test combined direction
    mock_press.reset_mock()
    mock_time.return_value = 1001.0
    controller.press("upleft")
    mock_press.assert_called_with(["up", "left"])

    # Test invalid
    mock_press.reset_mock()
    controller.press("invalid")
    mock_press.assert_not_called()
