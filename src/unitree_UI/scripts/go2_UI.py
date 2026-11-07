import av
import cv2
import io
import sys
from flask import Flask, Response, render_template_string

app = Flask(__name__)

# Basic clean HTML Dashboard layout for your UI
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Unitree Go2 Live Control Panel</title>
    <style>
        body { font-family: Arial, sans-serif; background: #121212; color: white; text-align: center; margin: 0; padding: 20px; }
        h1 { color: #00ff00; }
        .video-container { margin: 20px auto; width: 80%; max-width: 800px; border: 4px solid #333; border-radius: 8px; overflow: hidden; background: #000; }
        img { width: 100%; height: auto; display: block; }
    </style>
</head>
<body>
    <h1>Unitree Go2 Live Camera Interface</h1>
    <p>WSL Video Pipeline Status: <span style="color:#00ff00; font-weight:bold;">STREAMING LIVE</span></p>
    <div class="video-container">
        <!-- The live feed image element pointing to our video stream route -->
        <img src="/video_feed" />
    </div>
</body>
</html>
"""

def generate_frames():
    # SDP descriptor mapping the Go2's multicast camera parameters
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
        # Listen globally to incoming UDP stream packets
        container = av.open(sdp_file, format='sdp', options={"protocol_whitelist": "file,udp,rtp"})
        stream = container.streams.video
        
        for frame in container.decode(stream):
            # Convert frame directly to standard JPEG format for web transfer
            img = frame.to_ndarray(format='bgr24')
            ret, buffer = cv2.imencode('.jpg', img)
            frame_bytes = buffer.tobytes()
            
            # Pack frames into a continuous multipart JPEG web stream chunk
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    except Exception as e:
        print(f"Stream backend error: {e}")

@app.route('/')
def index():
    # Return the HTML structure of your UI dashboard
    return render_template_string(HTML_TEMPLATE)

@app.route('/video_feed')
def video_feed():
    # Stream the frames dynamically using HTTP server push configurations
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("Launching UI Live Video Dashboard...")
    print("Open your Windows browser and go to: http://localhost:5000")
    # Bind server to all network interfaces on port 5000
    app.run(host='0.0.0.0', port=5000, threaded=True)
