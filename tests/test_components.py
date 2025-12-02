import numpy as np
from stomp_detector import StompDetector


def test_detection():
    sr = 1000
    detector = StompDetector(
        sr=sr, win_ms=200, energy_threshold=0.5, min_stomp_sep_ms=100
    )

    # Create synthetic signal: Silence -> Pulse -> Silence
    t = np.linspace(0, 1, sr)
    audio = np.zeros_like(t)

    # Add pulse at 0.5s
    audio[475:525] = 1.0

    # Detect
    stomps = detector.detect(audio)
    assert len(stomps) == 1

    # Check if stomp is centered roughly
    assert len(stomps[0]) == int(0.2 * sr)

    # Peak should be in the middle of the segment
    peak_idx = np.argmax(stomps[0])
    assert 70 < peak_idx < 130, f"Peak index {peak_idx} not in range"


def test_deduplication(mocker):
    mock_time = mocker.patch("time.time")
    mock_time.return_value = 100.0

    sr = 1000
    detector = StompDetector(sr=sr, win_ms=200, min_stomp_sep_ms=200)

    # Create signal with pulse at 0.5s
    t = np.linspace(0, 1, sr)
    audio = np.zeros_like(t)
    audio[490:510] = 1.0

    # First pass: full signal
    stomps1 = detector.detect(audio)
    assert len(stomps1) == 1

    # Window 1: Same event
    audio1 = np.zeros(1000)
    audio1[500] = 1.0
    # Peak time 99.5. Diff 0. Should be skipped.
    stomps = detector.detect(audio1)
    assert len(stomps) == 0

    # Window 2: Shifted
    mock_time.return_value = 100.1
    audio2 = np.zeros(1000)
    audio2[400] = 1.0
    # Peak time 99.5. Diff 0. Should be skipped.
    stomps = detector.detect(audio2)
    assert len(stomps) == 0

    # Window 3: New event
    audio2[690:710] = 1.0
    # Peak time 99.8. Diff 0.3. Should be detected.
    stomps = detector.detect(audio2)
    assert len(stomps) == 1


def test_controller(mocker):
    from controller import KeyboarfdController

    # Mock pyautogui
    mock_down = mocker.patch("pyautogui.keyDown")
    mock_up = mocker.patch("pyautogui.keyUp")
    mocker.patch("time.sleep")

    controller = KeyboarfdController(verbose=False)

    # Test single direction
    controller.press("left")
    mock_down.assert_called_with("left")
    mock_up.assert_called_with("left")

    # Test combined direction
    mock_down.reset_mock()
    mock_up.reset_mock()

    controller.press("upleft")
    # Should press both
    assert mock_down.call_count == 2
    mock_down.assert_any_call("up")
    mock_down.assert_any_call("left")

    assert mock_up.call_count == 2
    mock_up.assert_any_call("up")
    mock_up.assert_any_call("left")

    # Test invalid
    mock_down.reset_mock()
    controller.press("invalid")
    mock_down.assert_not_called()
