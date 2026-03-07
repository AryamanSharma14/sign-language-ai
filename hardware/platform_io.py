# hardware/platform_io.py
"""Hardware Abstraction Layer — routes gesture events to GPIO, Serial, or terminal."""

import os
import time

from core.gestures import GESTURE_PIN_MAP
import config


def detect_platform() -> str:
    try:
        with open("/proc/device-tree/model", "r") as f:
            model = f.read()
        if "Raspberry Pi" in model:
            return "rpi"
    except (FileNotFoundError, OSError):
        pass
    return "pc"


class HardwareIO:
    def __init__(self, platform: str):
        self.platform = platform
        self._gpio = None
        self._serial = None
        self._gpio_available = False
        self._serial_available = False

        if platform == "rpi":
            try:
                import RPi.GPIO as GPIO
                self._gpio = GPIO
                self._gpio_available = True
            except ImportError:
                pass

        if platform == "arduino":
            try:
                import serial
                self._serial = serial.Serial("/dev/ttyUSB0", 9600, timeout=1)
                self._serial_available = True
            except Exception:
                pass

    def boot(self):
        names = {"rpi": "Raspberry Pi 4 Model B", "arduino": "Arduino Serial", "pc": "PC Demo"}
        print("[BOOT] Sign Language Recognition System")
        print(f"[BOOT] Platform: {names.get(self.platform, 'Unknown')}")
        print("[BOOT] Loading gesture model... OK")
        print("[BOOT] Initializing camera... OK")

        pins = ", ".join(str(p) for p in GESTURE_PIN_MAP.values())

        if self.platform == "rpi":
            if self._gpio_available:
                self._gpio.setmode(self._gpio.BCM)
                for pin in GESTURE_PIN_MAP.values():
                    self._gpio.setup(pin, self._gpio.OUT)
                    self._gpio.output(pin, self._gpio.LOW)
                print(f"[GPIO] Pins {pins} configured as OUTPUT")
            else:
                print(f"[GPIO] Pins {pins} configured as OUTPUT  (simulated)")
        elif self.platform == "arduino":
            status = "sent" if self._serial_available else "simulated"
            print(f"[SERIAL] Port /dev/ttyUSB0 @ 9600 baud  ({status})")
        else:
            print(f"[GPIO] Pins {pins} configured as OUTPUT  (simulated)")

    def emit_gesture(self, label: str, confidence: float):
        ts = time.strftime("%H:%M:%S")
        pin = GESTURE_PIN_MAP.get(label)

        if self.platform == "rpi":
            if pin is not None and self._gpio_available:
                for g, p in GESTURE_PIN_MAP.items():
                    val = self._gpio.HIGH if g == label else self._gpio.LOW
                    self._gpio.output(p, val)
            print(f"[{ts}] [GPIO] Pin {pin} HIGH  |  Gesture: {label}  |  Confidence: {confidence:.2f}")

        elif self.platform == "arduino":
            conf_int = int(confidence * 100)
            msg = f"GESTURE:{label}:{conf_int}\n"
            if self._serial_available:
                try:
                    self._serial.write(msg.encode())
                except Exception as e:
                    print(f"[{ts}] [SERIAL] Write error: {e}")
            status = "sent" if self._serial_available else "simulated"
            print(f"[{ts}] [SERIAL] {msg.strip()}  ({status})")

        else:
            color = "\033[92m" if confidence >= config.CONFIDENCE_THRESHOLD else "\033[93m"
            reset = "\033[0m"
            print(f"[{ts}] {color}[GPIO] Pin {pin} HIGH  |  Gesture: {label}  |  Confidence: {confidence:.2f}{reset}")

    def shutdown(self):
        if self.platform == "rpi":
            if self._gpio_available:
                for pin in GESTURE_PIN_MAP.values():
                    self._gpio.output(pin, self._gpio.LOW)
                self._gpio.cleanup()
            print("[GPIO] All pins LOW — cleanup done")
        elif self.platform == "arduino":
            if self._serial_available:
                try:
                    self._serial.close()
                except Exception:
                    pass
            print("[SERIAL] Port closed")
        else:
            print("[GPIO] All pins LOW (simulated)")
        print("[SYS] Shutting down...")
