#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from ultralytics import YOLO
import os

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

        # Load the fine-tuned YOLOv8 model weights from train-3
        model_path = os.path.expanduser('~/ros2_ws/runs/detect/train-3/weights/best.pt')
        try:
            self.model = YOLO(model_path)
            self.get_logger().info(f"Loaded YOLO model successfully from {model_path}")
        except Exception as e:
            self.get_logger().error(f"Failed to load YOLO model: {e}")

        self.get_logger().info("Anomaly Inference Node initialized and listening...")

    def listener_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Run YOLO inference
        try:
            results = self.model(frame, conf=0.25, verbose=False)
            detections = results[0].boxes
            count = len(detections)
            
            result_text = f"Detected {count} wall crack(s)"
        except Exception as e:
            result_text = f"Inference Error: {str(e)}"

        # Publish results
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
