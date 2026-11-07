import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from unitree_go.msg import Go2FrontVideoData
import cv2
import numpy as np
import threading
import time
from flask import Flask, Response

app = Flask(__name__)
latest_frame = None
frame_lock = threading.Lock()

class VideoSubscriber(Node):
    def __init__(self):
        super().__init__('go2_web_video_subscriber')
        
        # MANDATORY: Match the exact Best Effort QoS profile of the robot's camera
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=1
        )
        
        self.subscription = self.create_subscription(
            Go2FrontVideoData,
            '/frontvideostream',
            self.listener_callback,
            qos_profile
        )
        self.get_logger().info("Go2 camera subscriber node initialized successfully!")

    def listener_callback(self, msg):
        global latest_frame
        try:
            # FIX: Unitree uses specific video stream arrays (video720p, video360p, video180p)
            np_arr = np.frombuffer(msg.video720p, dtype=np.uint8)
            if len(np_arr) == 0:
                return
                
            # Decode the JPEG binary byte array back into an OpenCV visual frame
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is not None:
                with frame_lock:
                    latest_frame = frame.copy()
        except Exception as e:
            self.get_logger().error(f"Error handling video frame: {str(e)}")

def ros_thread():
    rclpy.init()
    node = VideoSubscriber()
    try:
        rclpy.spin(node)
    except Exception as e:
        print(f"ROS thread interrupted: {e}")
    finally:
        node.destroy_node()
        rclpy.shutdown()

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            with frame_lock:
                if latest_frame is not None:
                    success, encoded = cv2.imencode('.jpg', latest_frame)
                    if success:
                        frame_bytes = encoded.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.03)  # Yield the CPU loop slightly (~30 frames per second)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Fire up the ROS 2 node listener in the background
    t = threading.Thread(target=ros_thread, daemon=True)
    t.start()
    
    # Run the streaming web dashboard on port 5000
    app.run(host='0.0.0.0', port=5000, debug=False)
