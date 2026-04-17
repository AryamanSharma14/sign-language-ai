# web/app.py
"""
Flask web dashboard for sign language recognition.

Routes:
    GET  /            → HTML dashboard
    GET  /video_feed  → MJPEG stream
    WS   /ws          → WebSocket gesture events (JSON)

Run:
    python -m web.app
"""

import json
import time
import sys
from flask import Flask, Response, render_template
from flask_sock import Sock

import config
from setup_helper import ensure_assets
from web import stream as cam

# Ensure assets exist before starting
if not ensure_assets():
    print("[Error] Required assets missing. Run 'python setup.py' first.")
    sys.exit(1)

app = Flask(__name__, template_folder="templates", static_folder="static")
sock = Sock(app)

# Start camera background thread when module is imported
cam.start()


def _mjpeg_generator():
    while True:
        frame = cam.get_frame()
        if frame:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            )
        time.sleep(0.03)  # ~30 FPS ceiling for the HTTP stream


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        _mjpeg_generator(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@sock.route("/ws")
def websocket(ws):
    last_sent = {}
    while True:
        event = cam.get_latest_event()
        if event and event != last_sent:
            ws.send(json.dumps(event))
            last_sent = event
        time.sleep(config.WS_EVENT_RATE)


if __name__ == "__main__":
    app.run(host=config.WEB_HOST, port=config.WEB_PORT, debug=False)
