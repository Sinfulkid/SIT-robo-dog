import rclpy
from rclpy.node import Node
from unitree_go.msg import Go2FrontVideoData
import cv2
import numpy as np
import threading
from flask import Flask, Response

app = Flask(__name__)
latest_frame = None
frame_lock = threading.Lock()

class VideoSubscriber(Node):
    def __init__(self):
        super().__init__('go2_web_video_subscriber')
        self.subscription = self.create_subscription(
            Go2FrontVideoData,
            '/frontvideostream',
            self.listener_callback,
            10
        )

    def listener_callback(self, msg):
        global latest_frame
        try:
            # Convert raw byte payload from Go2FrontVideoData to numpy array and decode JPEG/H264 frame
            np_arr = np.frombuffer(msg.data, dtype=np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is not None:
                with frame_lock:
                    latest_frame = frame.copy()
        except Exception as e:
            pass

def ros_thread():
    rclpy.init()
    node = VideoSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            with frame_lock:
                if latest_frame is None:
                    continue
                success, encoded = cv2.imencode('.jpg', latest_frame)
                if not success:
                    continue
                frame_bytes = encoded.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    t = threading.Thread(target=ros_thread, daemon=True)
    t.start()
    app.run(host='0.0.0.0', port=5000)
