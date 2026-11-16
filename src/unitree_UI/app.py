import io
import socket
import threading
import av
import cv2
from flask import Flask, Response, render_template

app = Flask(__name__)

latest_points_bytes = b""
points_lock = threading.Lock()


def udp_lidar_listener():
    global latest_points_bytes
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", 9876))
    print("[3D SLAM UDP] Listening on port 9876...")

    scan_history = []
    while True:
        data, _ = sock.recvfrom(65535)
        scan_history.append(data)
        if len(scan_history) > 5:  # Merge last 5 packets (~15,000-20,000 3D points)
            scan_history.pop(0)
        with points_lock:
            latest_points_bytes = b"".join(scan_history)


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


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_camera_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/slam_points')
def slam_points():
    with points_lock:
        data = latest_points_bytes
    return Response(data, mimetype='application/octet-stream')


if __name__ == '__main__':
    threading.Thread(target=udp_lidar_listener, daemon=True).start()
    app.run(host='0.0.0.0', port=5000, threaded=True)
