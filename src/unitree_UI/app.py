import av
import cv2
import io
from flask import Flask, Response, render_template, jsonify

app = Flask(__name__)

def generate_frames():
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
        container = av.open(sdp_file, format='sdp', options={"protocol_whitelist": "file,udp,rtp"})
        stream = container.streams.video
        for frame in container.decode(stream):
            img = frame.to_ndarray(format='bgr24')
            ret, buffer = cv2.imencode('.jpg', img)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    except Exception as e:
        print(f"Streaming error: {e}")

@app.route('/')
def index():
    # Renders the index.html layout file straight out of the templates folder
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# --- NEW: THIS API RECEIVES THE BUTTON CLICKS FROM THE WEB INTERFACE ---
@app.route('/api/command/<action>')
def handle_command(action):
    print(f"[EXECUTING] Received request to command Go2 to: {action.upper()}")
    
    # 💡 This is exactly where you will eventually import Unitree's SDK 
    # or execution packages to physically send commands to the Go2.
    
    return jsonify({"status": f"Command '{action}' processed successfully"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
