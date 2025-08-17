#!/usr/bin/env python3
from ultralytics import YOLO
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class YOLOv8Node(Node):
    def __init__(self):
        super().__init__('yolov8_node')
        self.bridge = CvBridge()

        # Load your trained model
        self.get_logger().info("Loading YOLOv8 model...")
        self.model = YOLO("/home/kruthik/husarion_ws/src/explore/yolo_aruco/data/runs/detect/train/weights/best.pt")  # change to actual path
        self.get_logger().info("YOLOv8 model loaded successfully.")
        self.subscription = self.create_subscription(
            Image,
            '/front_cam/color/image_raw',
            self.image_callback,
            10
        )

    def image_callback(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            results = self.model(img)[0]

            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf)
                cls = int(box.cls)
                label = self.model.names[cls]

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f"{label} ({conf:.2f})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            cv2.imshow("YOLOv8 Detection", img)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"Inference failed: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = YOLOv8Node()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()