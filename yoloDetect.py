import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
from queue import LifoQueue
from threading import Thread
from yolo.visionDetect import YOLO
from rclpy.executors import MultiThreadedExecutor
from tutorial_interfaces.msg import Img, Motor
import sys
sys.path.append('/home/ubuntu/camera_pub/yolo') 

import os
os.environ['DISPLAY'] = ':0'

motor = Motor()
img = Img()

cv_image_queue = LifoQueue(maxsize=1)

class CameraSubscriberNode(Node):
    def __init__(self):
        super().__init__('camera_subscriber_node')
        self.show = self.create_subscription(Image, "camera_image", self.image_show, 10)
        self.show  # prevent unused variable warning
        self.cv_bridge = CvBridge()

    def image_show(self, msg):
        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, 'bgr8')
            cv_image = cv2.resize(cv_image, (1280, 720))
            cv_image_queue.put(cv_image)  # Put image into queue
        except Exception as e:
            self.get_logger().error(f"Error processing image: {str(e)}")

class YOLO_Publisher_Node(Node):
    def __init__(self):
        super().__init__("YOLO_publisher_node")
        self.motor_publish = self.create_publisher(Motor, "Gimbal_Topic", 10)
        self.img_publish = self.create_publisher(Img, "YOLO_Topic", 10)
        
        track_timer = 0.1 # sec
        self.motor_timer = self.create_timer(track_timer,self.tracker_publish)
        
    def tracker_publish(self):
        global motor, img
        self.motor_publish.publish(motor)
        self.img_publish.publish(img)

class tracking():
    def __init__(self):
        self.yolo = YOLO("yolo/tennisv7.pt")
        print("track initialized complete")
        
    def run_yolo(self,queue:LifoQueue):
        while True:
            global img, motor
            f = queue.get(timeout=1)
            img.detect_status, img.target_center_status, motor.yaw, motor.pitch =self.yolo.run(f)


def _spinThread(pub, sub):
    executor = MultiThreadedExecutor()
    executor.add_node(sub)
    executor.add_node(pub)
    executor.spin()


def main(args=None):
    rclpy.init(args=args)
    cam_subscriber_node = CameraSubscriberNode()
    AI_publisher_node = YOLO_Publisher_Node()
    spinThread = Thread(target=_spinThread, args=(cam_subscriber_node, AI_publisher_node))
  
    tracker = tracking()
    _tracker_thread = Thread(target=tracker.run_yolo, args=(cv_image_queue,), daemon = True)
 
    try:
        spinThread.start()
        _tracker_thread.start()
    except Exception as e:
        print(e)
        _tracker_thread.join()
        cam_subscriber_node.destroy_node()
        AI_publisher_node.destroy_node()
        rclpy.shutdown()
    except KeyboardInterrupt:
        cam_subscriber_node.destroy_node()
        AI_publisher_node.destroy_node()
        rclpy.shutdown()

    

if __name__ == '__main__':
    main()
