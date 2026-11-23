import ipaddress
import math
import socket
import sys
from typing import List
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2

DEFAULT_LAPTOP_IP = "192.168.123.50"
LAPTOP_PORT = 9876

# Obstacle height slice (meters) and max points per UDP packet (3000 * 12 bytes = 36 KB)
Z_MIN = -1.0
Z_MAX = 1.0
MAX_POINTS = 3000
LIDAR_INVERTED = True
LIDAR_YAW_RAD = 0.5


def parse_laptop_ip(argv: List[str], default_ip: str) -> str:
    if len(argv) < 2:
        return default_ip

    if argv[1] in ("-h", "--help"):
        print(f"Usage: python3 {argv[0]} [laptop_ip]")
        print(f"Default laptop IP: {default_ip}")
        sys.exit(0)

    candidate = argv[1].strip()
    try:
        ipaddress.ip_address(candidate)
        return candidate
    except ValueError:
        print(f"Error: '{candidate}' is not a valid IP address.")
        print(f"Usage: python3 {argv[0]} [laptop_ip]")
        sys.exit(1)


class LidarUdpRelay(Node):
    def __init__(self, laptop_ip: str, laptop_port: int):
        super().__init__('lidar_udp_relay')
        self.laptop_ip = laptop_ip
        self.laptop_port = laptop_port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self.sub = self.create_subscription(PointCloud2, '/utlidar/cloud', self.cb, qos)
        self.get_logger().info(
            f"Streaming /utlidar/cloud (yaw={LIDAR_YAW_RAD:+.4f} rad, inverted={LIDAR_INVERTED}, "
            f"z=[{Z_MIN}, {Z_MAX}]) -> {self.laptop_ip}:{self.laptop_port}"
        )

    def cb(self, msg: PointCloud2):
        step = msg.point_step
        raw = bytes(msg.data)
        n = len(raw) // step if step > 0 else 0
        if n == 0:
            return

        pts = np.frombuffer(raw, dtype=np.uint8).reshape(n, step)
        xyz = np.frombuffer(pts[:, :12].tobytes(), dtype=np.float32).reshape(n, 3).copy()

        # 180-degree roll around forward axis flips lateral (y) and vertical (z)
        if LIDAR_INVERTED:
            xyz[:, 1] = -xyz[:, 1]
            xyz[:, 2] = -xyz[:, 2]

        # Apply yaw offset in the horizontal plane before height filtering
        if LIDAR_YAW_RAD != 0.0:
            c = math.cos(LIDAR_YAW_RAD)
            s = math.sin(LIDAR_YAW_RAD)
            x_rot = xyz[:, 0] * c - xyz[:, 1] * s
            y_rot = xyz[:, 0] * s + xyz[:, 1] * c
            xyz[:, 0] = x_rot
            xyz[:, 1] = y_rot

        mask = (
            np.isfinite(xyz[:, 0])
            & np.isfinite(xyz[:, 1])
            & (xyz[:, 2] > Z_MIN)
            & (xyz[:, 2] < Z_MAX)
        )
        filtered = xyz[mask][:MAX_POINTS].astype(np.float32)
        if len(filtered) > 0:
            self.sock.sendto(filtered.tobytes(), (self.laptop_ip, self.laptop_port))

    def destroy_node(self):
        self.sock.close()
        super().destroy_node()


def main():
    laptop_ip = parse_laptop_ip(sys.argv, DEFAULT_LAPTOP_IP)
    rclpy.init()
    node = LidarUdpRelay(laptop_ip, LAPTOP_PORT)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
