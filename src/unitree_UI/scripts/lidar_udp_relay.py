import socket
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2

LAPTOP_IP = "192.168.123.50"
LAPTOP_PORT = 9876


class Lidar3DUdpRelay(Node):
    def __init__(self):
        super().__init__('lidar_3d_udp_relay')
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
        self.sub = self.create_subscription(PointCloud2, '/utlidar/cloud', self.send_cloud, qos)
        self.get_logger().info(f"Streaming 3D /utlidar/cloud -> {LAPTOP_IP}:{LAPTOP_PORT}")

    def send_cloud(self, msg: PointCloud2):
        step = msg.point_step
        raw = bytes(msg.data)
        n = len(raw) // step
        if n == 0:
            return
        pts = np.frombuffer(raw, dtype=np.uint8).reshape(n, step)
        xyz = np.frombuffer(pts[:, :12].tobytes(), dtype=np.float32).reshape(n, 3)

        # Keep floor (-0.55m) through tall walls (2.0m) within 8m radius; cap to 4000 pts per packet
        dist2 = xyz[:, 0]**2 + xyz[:, 1]**2
        mask = (xyz[:, 2] > -0.55) & (xyz[:, 2] < 2.0) & (dist2 < 64.0)
        filtered = xyz[mask][::2][:4000].astype(np.float32)

        if len(filtered) > 0:
            self.sock.sendto(filtered.tobytes(), (LAPTOP_IP, LAPTOP_PORT))


def main():
    rclpy.init()
    node = Lidar3DUdpRelay()
    rclpy.spin(node)


if __name__ == '__main__':
    main()
