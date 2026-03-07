"""Single source of truth for gesture labels, GPIO pins, and command translations."""

GESTURE_LABELS = [
    "A", "B", "C", "D", "L", "Y",
    "PEACE", "OK", "STOP", "HELP", "THANK_YOU",
]

GESTURE_PIN_MAP = {
    "A":         17,
    "B":         27,
    "C":         22,
    "D":          5,
    "L":          6,
    "Y":         13,
    "PEACE":     23,
    "OK":        24,
    "STOP":      19,
    "HELP":      26,
    "THANK_YOU": 21,
}

COMMAND_MAP = {
    "A":         "YES",
    "B":         "NO",
    "C":         "CONFIRM",
    "D":         "DOWN",
    "L":         "LETTER_L",
    "Y":         "LETTER_Y",
    "PEACE":     "HELLO",
    "OK":        "OK",
    "STOP":      "STOP",
    "HELP":      "HELP",
    "THANK_YOU": "THANKS",
}
