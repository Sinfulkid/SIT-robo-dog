import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import onnxruntime as ort
from std_msgs.msg import String
import os
from ament_index_python.packages import get_package_share_directory

class AnomalyInferenceNode(Node):
    def __init__(self):
        super().__init__('anomaly_inference_node')
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.listener_callback,
            10
        )
        self.publisher_ = self.create_publisher(
            String,
            'anomaly_results',
            10
        )
        self.bridge = CvBridge()
        
        # Locate and load the ONNX model session
        pkg_dir = get_package_share_directory('anomaly_detector')
        model_path = os.path.expanduser('~/ros2_ws/src/anomaly_detector/models/model.onnx')

        try:
            self.ort_session = ort.InferenceSession(model_path)
            self.input_name = self.ort_session.get_inputs()[0].name
            self.get_logger().info(f"Loaded ONNX model successfully from {model_path}")
        except Exception as e:
            self.get_logger().error(f"Failed to load ONNX model: {e}")

        self.get_logger().info("Anomaly Inference Node initialized and listening...")

    def listener_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
        # 1. Preprocess the frame for MobileNetV2 (Resize to 224x224)
        resized = cv2.resize(frame, (224, 224))
        input_data = resized.astype(np.float32) / 255.0
        input_data = np.expand_dims(input_data, axis=0)  # Add batch dimension (1, H, W, C)
        
        # MobileNetV2 expects NCHW format instead of NHWC, transpose the axes:
        input_data = np.transpose(input_data, (0, 3, 1, 2))

        # 2. Run ONNX inference
        try:
            outputs = self.ort_session.run(None, {self.input_name: input_data})
            prediction = np.argmax(outputs[0])
            result_text = f"Class ID detected: {prediction}"
        except Exception as e:
            result_text = f"Inference Error: {str(e)}"
        
        # 3. Publish the result
        result_msg = String()
        result_msg.data = result_text
        self.publisher_.publish(result_msg)
        
        self.get_logger().info(f"Published result: {result_msg.data}")

def main(args=None):
    rclpy.init(args=args)
    node = AnomalyInferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
