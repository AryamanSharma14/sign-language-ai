# hardware/gpio_controller.py
"""
GPIO controller for Raspberry Pi.
Auto-falls back to terminal simulation on non-Pi systems.
"""

from core.gestures import GESTURE_PIN_MAP


class GpioController:
    def __init__(self, simulate: bool = False):
        self.simulate = simulate
        self._gpio = None
        if not simulate:
            try:
                import RPi.GPIO as GPIO
                self._gpio = GPIO
            except ImportError:
                self.simulate = True

    def setup(self):
        if not self.simulate:
            self._gpio.setmode(self._gpio.BCM)
            for pin in GESTURE_PIN_MAP.values():
                self._gpio.setup(pin, self._gpio.OUT)
                self._gpio.output(pin, self._gpio.LOW)
        pins = list(GESTURE_PIN_MAP.values())
        status = "(simulated)" if self.simulate else ""
        print(f"[GPIO] Pins {pins} configured as OUTPUT {status}")

    def set_gesture(self, label: str):
        pin = GESTURE_PIN_MAP.get(label)
        if pin is None:
            return
        if not self.simulate:
            for g, p in GESTURE_PIN_MAP.items():
                val = self._gpio.HIGH if g == label else self._gpio.LOW
                self._gpio.output(p, val)
        sim = " (sim)" if self.simulate else ""
        print(f"[GPIO] Pin {pin} HIGH → {label}{sim}")

    def all_low(self):
        if not self.simulate:
            for pin in GESTURE_PIN_MAP.values():
                self._gpio.output(pin, self._gpio.LOW)

    def cleanup(self):
        self.all_low()
        if not self.simulate:
            self._gpio.cleanup()
        print("[GPIO] Cleanup done")
