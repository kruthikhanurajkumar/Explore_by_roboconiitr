#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TwistStamped
from std_msgs.msg import Header

class TwistToStampedNode(Node):
    def __init__(self):
        super().__init__('twist_to_stamped_bridge')
        self.sub = self.create_subscription(Twist, '/cmd_vel_raw', self.callback, 10)
        self.pub = self.create_publisher(TwistStamped, '/panther/cmd_vel', 10)
        self.get_logger().info('Bridge Node Started: /cmd_vel_raw (Twist) → /cmd_vel (TwistStamped)')

    def callback(self, msg):
        stamped_msg = TwistStamped()
        stamped_msg.header.stamp = self.get_clock().now().to_msg()
        stamped_msg.twist = msg
        self.pub.publish(stamped_msg)

def main(args=None):
    rclpy.init(args=args)
    node = TwistToStampedNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()