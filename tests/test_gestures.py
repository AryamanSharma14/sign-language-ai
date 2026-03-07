from core.gestures import GESTURE_LABELS, GESTURE_PIN_MAP, COMMAND_MAP

def test_all_labels_have_pin():
    for label in GESTURE_LABELS:
        assert label in GESTURE_PIN_MAP, f"{label} missing from GESTURE_PIN_MAP"

def test_all_labels_have_command():
    for label in GESTURE_LABELS:
        assert label in COMMAND_MAP, f"{label} missing from COMMAND_MAP"

def test_pins_are_unique():
    pins = list(GESTURE_PIN_MAP.values())
    assert len(pins) == len(set(pins)), "Duplicate GPIO pins detected"

def test_gesture_count():
    assert len(GESTURE_LABELS) == 11
