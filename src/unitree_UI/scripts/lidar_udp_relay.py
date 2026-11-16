import socket
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2

LAPTOP_IP = "192.168.123.50"
LAPTOP_PORT = 9876


class LidarUdpRelay(Node):
    def __init__(self):
        super().__init__('lidar_udp_relay')
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
        self.sub = self.create_subscription(PointCloud2, '/utlidar/cloud', self.cb, qos)
        self.get_logger().info(f"Streaming /utlidar/cloud -> {LAPTOP_IP}:{LAPTOP_PORT}")

    def cb(self, msg: PointCloud2):
        step = msg.point_step
        raw = bytes(msg.data)
        n = len(raw) // step
        if n == 0:
            return
        pts = np.frombuffer(raw, dtype=np.uint8).reshape(n, step)
        xyz = np.frombuffer(pts[:, :12].tobytes(), dtype=np.float32).reshape(n, 3)

        # Keep obstacle height (0.10m to 1.50m) & cap to 1500 points so it fits in 1 UDP packet
        mask = (xyz[:, 2] > 0.10) & (xyz[:, 2] < 1.50)
        filtered = xyz[mask][::2][:1500].astype(np.float32)
        if len(filtered) > 0:
            self.sock.sendto(filtered.tobytes(), (LAPTOP_IP, LAPTOP_PORT))


def main():
    rclpy.init()
    node = LidarUdpRelay()
    rclpy.spin(node)


if __name__ == '__main__':
    main()
