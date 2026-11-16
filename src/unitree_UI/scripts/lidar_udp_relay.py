import socket
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2
from unitree_go.msg import LowState

LAPTOP_IP = "192.168.123.50"
LAPTOP_PORT = 9876


class LidarImuUdpRelay(Node):
    def __init__(self):
        super().__init__('lidar_imu_udp_relay')
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # [w, x, y, z]

        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
        best_effort = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)

        self.sub_imu = self.create_subscription(LowState, '/lf/lowstate', self.imu_cb, best_effort)
        self.sub_lidar = self.create_subscription(PointCloud2, '/utlidar/cloud', self.lidar_cb, qos)
        self.get_logger().info(f"Streaming 3D LiDAR + IMU -> {LAPTOP_IP}:{LAPTOP_PORT}")

    def imu_cb(self, msg: LowState):
        self.quat = np.array(msg.imu_state.quaternion, dtype=np.float32)

    def lidar_cb(self, msg: PointCloud2):
        step = msg.point_step
        raw = bytes(msg.data)
        n = len(raw) // step
        if n == 0:
            return
        pts = np.frombuffer(raw, dtype=np.uint8).reshape(n, step)
        xyz = np.frombuffer(pts[:, :12].tobytes(), dtype=np.float32).reshape(n, 3)

        dist2 = xyz[:, 0]**2 + xyz[:, 1]**2
        mask = (xyz[:, 2] > -0.55) & (xyz[:, 2] < 2.0) & (dist2 < 64.0)
        filtered = xyz[mask][::2][:4000].astype(np.float32)

        if len(filtered) > 0:
            packet = self.quat.tobytes() + filtered.tobytes()
            self.sock.sendto(packet, (LAPTOP_IP, LAPTOP_PORT))


def main():
    rclpy.init()
    node = LidarImuUdpRelay()
    rclpy.spin(node)


if __name__ == '__main__':
    main()

