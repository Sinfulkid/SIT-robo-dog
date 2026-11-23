import io
import socket
import threading
import time
import av
import cv2
import numpy as np
from flask import Flask, Response, render_template

app = Flask(__name__)

latest_slam_jpg = None
slam_lock = threading.Lock()

MAP_SIZE = 600
SCALE = 200  # 50 pixels = 1 meter (covers 12m x 12m)
slam_canvas = np.zeros((MAP_SIZE, MAP_SIZE, 3), dtype=np.uint8)


def udp_lidar_listener():
    global latest_slam_jpg, slam_canvas

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", 9876))
    print("[SLAM UDP] Listening on port 9876...")

    while True:
        data, _ = sock.recvfrom(65535)
        xyz = np.frombuffer(data, dtype=np.float32).reshape(-1, 3)

        px = (MAP_SIZE // 2 + (xyz[:, 0] * SCALE)).astype(np.int32)
        py = (MAP_SIZE // 2 - (xyz[:, 1] * SCALE)).astype(np.int32)

        valid = (px >= 0) & (px < MAP_SIZE) & (py >= 0) & (py < MAP_SIZE)
        px, py = px[valid], py[valid]

        # Fade previous scan slightly to create a smooth live radar/SLAM trail
        slam_canvas = (slam_canvas * 0.85).astype(np.uint8)
        slam_canvas[py, px] = (102, 255, 0)  # BGR green

        display = slam_canvas.copy()
        # Range rings (1m, 2m, 3m, 4m, 5m)
        for r in range(1, 6):
            cv2.circle(display, (MAP_SIZE // 2, MAP_SIZE // 2), r * SCALE, (35, 45, 58), 1)
        cv2.line(display, (MAP_SIZE // 2, 0), (MAP_SIZE // 2, MAP_SIZE), (45, 55, 72), 1)
        cv2.line(display, (0, MAP_SIZE // 2), (MAP_SIZE, MAP_SIZE // 2), (45, 55, 72), 1)
        # Robot position & forward heading arrow
        cv2.circle(display, (MAP_SIZE // 2, MAP_SIZE // 2), 6, (0, 200, 255), -1)
        cv2.arrowedLine(display, (MAP_SIZE // 2, MAP_SIZE // 2),
                        (MAP_SIZE // 2 + 22, MAP_SIZE // 2), (0, 200, 255), 2)

        ret, buffer = cv2.imencode('.jpg', display)
        if ret:
            with slam_lock:
                latest_slam_jpg = buffer.tobytes()


def generate_camera_frames():
    sdp_config = (
        "v=0\n"
        "o=- 0 0 IN IP4 127.0.0.1\n"
        "s=Unitree Go2 Camera Stream\n"
        "c=IN IP4 230.1.1.1\n"
        "t=0 0\n"
        "m=video 1720 RTP/AVP 96\n"
        "a=rtpmap:96 H264/90000\n"
    )
    sdp_file = io.BytesIO(sdp_config.encode('utf-8'))

    try:
        container = av.open(
            sdp_file,
            format='sdp',
            options={
                "protocol_whitelist": "file,udp,rtp",
                "localaddr": "192.168.123.50"
            }
        )
        stream = container.streams.video
        for frame in container.decode(stream):
            img = frame.to_ndarray(format='bgr24')
            ret, buffer = cv2.imencode('.jpg', img)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    except Exception as e:
        print(f"Camera streaming error: {e}")


def generate_slam_frames():
    blank = np.zeros((MAP_SIZE, MAP_SIZE, 3), dtype=np.uint8)
    cv2.putText(blank, "Waiting for Go2 LiDAR UDP (9876)...", (95, 300),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 102), 2)
    _, blank_jpg = cv2.imencode('.jpg', blank)

    while True:
        with slam_lock:
            frame_bytes = latest_slam_jpg if latest_slam_jpg is not None else blank_jpg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.06)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_camera_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/slam_feed')
def slam_feed():
    return Response(generate_slam_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    threading.Thread(target=udp_lidar_listener, daemon=True).start()
    app.run(host='0.0.0.0', port=5000, threaded=True)
