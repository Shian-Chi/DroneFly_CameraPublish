import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
    

class CameraSubscriberNode(Node):
    def __init__(self):
        super().__init__('camera_subscriber_node')
        self.show = self.create_subscription(Image, 'camera_image', self.image_show, 10)
        self.show  # prevent unused variable warning
        self.cv_bridge = CvBridge()
    
    def image_show(self, msg):
        try:
            # 將ROS圖像消息轉換為OpenCV圖像
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, 'bgr8')
            cv2.imshow('Camera Image', cv_image)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"Error processing image: {str(e)}")
    
    
    
    

def main(args=None):
    rclpy.init(args=args)
    camera_subscriber_node = CameraSubscriberNode()
    try:
        rclpy.spin(camera_subscriber_node)
        camera_subscriber_node.destroy_node()
        rclpy.shutdown()
    finally:
        camera_subscriber_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
