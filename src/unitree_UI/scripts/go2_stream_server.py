import cv2
from flask import Flask, Response, jsonify

app = Flask(__name__)

# Unitree Go2 default RTP/H.264 multicast stream via GStreamer
GST_PIPELINE = (
    "udpsrc address=230.1.1.1 port=1720 multicast-iface=eth10 ! "
    "application/x-rtp, media=video, encoding-name=H264 ! "
    "rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! "
    "video/x-raw, format=BGR ! appsink drop=1"
)

def generate_mjpeg_stream():
    # Open video stream using GStreamer backend
    cap = cv2.VideoCapture(GST_PIPELINE, cv2.CAP_GSTREAMER)

    # Fallback to physical camera device if multicast fails
    if not cap.isOpened():
        print("[WARN] GStreamer pipeline failed. Falling back to VideoCapture(0)")
        cap = cv2.VideoCapture(0)

    print("[INFO] Stream server active.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue
        
        # Compress frame to JPEG for fast HTTP transfer
        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ret:
            continue

        frame_bytes = buffer.tobytes()

        # Yield frame as part of an MJPEG continuous stream
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_mjpeg_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status')
def status():
    return jsonify({"status": "online", "robot": "Unitree Go2 EDU"})

if __name__ == '__main__':
    # Host on all network interfaces on port 5000
    app.run(host='0.0.0.0', port=5000, threaded=True)
