import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Image
from std_msgs.msg import Header
import cv2
from cv_bridge import CvBridge
from threading import Thread
from queue import Queue
import time

cam_ret = False
frame_queue = Queue()

def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1280,
    capture_height=720,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
        )
    )


def getFrameQueue():
    global frame_queue
    data = frame_queue.get()
    return data["image"]

def getCaptureFPS():
    global frame_queue
    data = frame_queue.get()
    return data["fps"]


class Camera:
    def __init__(self, frame_queue):
        # MIPI CSI Camera (Gstreamer)
        self.cam = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
        
        # USB Camera
        '''
        self.cam = cv2.VideoCapture(-1)
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH,1920)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT,1080)
        '''
        
        self.frame_queue = frame_queue
        self.is_running = False
        self.fps = 0.0
        self._t_last = time.time() * 1000
        self._data = {}

    def capture_queue(self):
        global cam_ret
        self._t_last = time.time() * 1000
        while self.is_running and self.cam.isOpened():
            cam_ret, frame = self.cam.read()
            if not cam_ret:
                continue
            if self.frame_queue.qsize() < 1:
                t = time.time() * 1000
                t_span = t - self._t_last
                self.fps = int(1000.0 / t_span)
                self._data["image"] = frame.copy()
                self._data["fps"] = self.fps
                self.frame_queue.put(self._data)
                self._t_last = t

    def run(self):
        self.is_running = True
        self.thread_capture = Thread(target=self.capture_queue)
        self.thread_capture.start()

    def stop(self):
        self.is_running = False
        self.thread_capture.join()  # Wait for the thread to finish
        self.cam.release()

class CameraPublisherNode(Node):
    def __init__(self):
        super().__init__('camera_publisher_node')
        self.camera_pub = self.create_publisher(Image, 'camera_image', 10)
        self.cv_bridge = CvBridge()  # Initialize CvBridge
        timer_period = 1 / 60  # seconds
        self.image_publish_timer = self.create_timer(timer_period, self.image_publish_callback)

    def image_publish_callback(self):
        if cam_ret:
            frame = getFrameQueue()
            image_msg = self.cv_bridge.cv2_to_imgmsg(frame, 'bgr8')
            image_msg.header = Header(stamp=self.get_clock().now().to_msg())
            self.camera_pub.publish(image_msg)

def _thread_ros2_communication(pub):
    executor = MultiThreadedExecutor()
    executor.add_node(pub)
    try:
        executor.spin()
    except Exception as e:
        print(e)
    finally:
        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    
    cam = Camera(frame_queue)
    cam.run()
    
    camera_publisher_node = CameraPublisherNode()
    try:
        _thread_ros2_communication(camera_publisher_node)
        print("Start Publisher Image")
    except Exception as e:
        print(e)
    finally:
        cam.stop()
        camera_publisher_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
