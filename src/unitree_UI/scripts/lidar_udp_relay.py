import socket
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2

LAPTOP_IP = "192.168.123.50"
LAPTOP_PORT = 9876

# Obstacle height slice (meters) and max points per UDP packet (3000 * 12 bytes = 36 KB)
Z_MIN = 0.05
Z_MAX = 1.40
MAX_POINTS = 3000


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

        mask = (xyz[:, 2] > Z_MIN) & (xyz[:, 2] < Z_MAX)
        filtered = xyz[mask][:MAX_POINTS].astype(np.float32)

        if len(filtered) > 0:
            self.sock.sendto(filtered.tobytes(), (LAPTOP_IP, LAPTOP_PORT))


def main():
    rclpy.init()
    node = LidarUdpRelay()
    rclpy.spin(node)


if __name__ == '__main__':
    main()
