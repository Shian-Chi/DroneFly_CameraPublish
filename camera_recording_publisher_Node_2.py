import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Header
import cv2
from cv_bridge import CvBridge
from threading import Thread
from queue import Queue
import os, time
from yolo.utils.general import check_imshow
from rclpy.executors import MultiThreadedExecutor

publish_name = 'camera_image'

timeOutValue = 60 # sec


frame_queue = Queue()
cam_ret = False

def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
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

class ImagePublish(Node):
    def __init__(self):
        super().__init__('camera_publisher_node')
        self.camera_pub = self.create_publisher(Image, publish_name, 10)
        timer_period = 1/30  # seconds
        self.timer = self.create_timer(timer_period, self.cameraPublish)
        
        self.cv_bridge = CvBridge()
        print("Start publishing camera data")
        
    def cameraPublish(self):
        global frame_queue
        if cam_ret:
            frame = getFrameQueue()
            image_msg = self.cv_bridge.cv2_to_imgmsg(frame, 'bgr8')
            image_msg.header = Header(stamp=self.get_clock().now().to_msg())
            
            self.camera_pub.publish(image_msg)


class Camera:
    def __init__(self, frame_q):
        # Capture
        self.cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER) 
        self.frame = frame_q
        self.is_running = False  
        self.fps = 0.0  
        self._t_last = time.time() * 1000
        self._data = {} 
        
        # Recording
        # File name confirmation
        self.file_count = 1
        self.name = f'output{self.file_count}.avi'
        self.fileName()
        
        # Settings recording parameters
        self.frame_width = 1280  # Set the width of the video
        self.frame_height = 720  # Set film height
        self.fps = 30 # Set the frame rate of the video
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID') # *.avi
        self.videoSaveSettings()

        # Check whether the screen can be displayed 
        self.view_img = False
        if self.view_img:
            self.view_img = check_imshow()

        print("Start recording")

        self.start_time = time.time()
        self.end_time = timeOutValue
        self.current_time = 0
        self.elapsed_time = 0
    
    # ------------------------Recording------------------------
    def videoSaveSettings(self):
        self.out = cv2.VideoWriter(self.name, self.fourcc, self.fps, (self.frame_width, self.frame_height))

    def isFileExist(self):
        return os.path.exists(self.name) # Check if the file exists

    def setFileName(self):
        self.file_count = self.file_count + 1
        self.name = f'output{self.file_count}.avi'
        print(self.name)
    
    def fileName(self):
        while self.isFileExist():
            self.setFileName()
    
    def stop_recording(self):
        self.out.release()
        if self.view_img:
            cv2.destroyAllWindows()
        self.get_logger().info("Recording stopped.")
        self.destroy_node()
        rclpy.shutdown()

    def recording(self, frame):
        try:
            # Resize the image and write it to the video file
            cv_image = frame
            if cv_image is not None:
                cv_image = cv2.resize(cv_image, (self.frame_width, self.frame_height))
                self.out.write(cv_image)
            else:
                print("No Img")

            if self.view_img:
                cv2.imshow('Camera Image', cv_image)
                cv2.waitKey(1)
                
        except KeyboardInterrupt:
            self.out.release()
            print("Caught KeyboardInterrupt")
        except Exception as e:
            self.out.release() 
            print(f"Error Info: \n{e}")

    def timing(self, frame):
        # Calculate elapsed time since recording start
        self.current_time = time.time()
        self.elapsed_time = self.current_time - self.start_time

        # Check if elapsed time exceeds the timeout value
        if self.elapsed_time >= self.end_time:
            print("Time out") 
            self.fileName()
            self.out.release()
            self.videoSaveSettings()
            self.start_time = time.time()
        self.recording(frame)
    # ---------------------------------------------------------
     
    def capture_queue(self):
        global cam_ret
        self._t_last = time.time() * 1000
        try:
            while self.is_running and self.cap.isOpened():
                cam_ret, frame = self.cap.read()

                if frame is None:
                    print("Failed to capture frame from the camera.")
                    continue  # Skip processing and continue capturing frames.

                self.timing(frame)  # Recording

                if self.frame.qsize() < 1:
                    t = time.time() * 1000
                    t_span = t - self._t_last
                    self.fps = int(1000.0 / t_span)
                    self._data["image"] = frame.copy()
                    self._data["fps"] = self.fps
                    self.frame.put(self._data)
                    self._t_last = t
        except KeyboardInterrupt:
            self.stop()
            frame_queue.task_done()
        except Exception as e:
            print(e)
            self.stop()
            frame_queue.task_done()
        
    def run(self):
        global frame_queue
        self.is_running = True
        self.thread_capture = Thread(target=self.capture_queue)
        self.thread_capture.start()
 
    def stop(self):
        self.is_running = False 
        self.cap.release()
        self.out.release()
        frame_queue.task_done()

def _thread_ros2_communication(sub):
    executor = MultiThreadedExecutor()
    executor.add_node(sub)
    try:
        executor.spin()
    except Exception as e:
        print(e)
    finally:
        rclpy.shutdown()

def main(args=None):
    # ROS2 init
    rclpy.init(args=args)
    try:
        # Camera
        cam = Camera(frame_q=frame_queue)
        cam.run() # run camera thread
        
        imgPub = ImagePublish()
        _thread_ros2_communication(imgPub)
    except KeyboardInterrupt:
        cam.stop()
        cam.thread_capture.join()
        print("Turn off camera")
        frame_queue.task_done()

        imgPub.destroy_node()
        rclpy.shutdown()
    except Exception as e:
        print(e)
        cam.stop()
        cam.thread_capture.join()
        print("Turn off camera")
        frame_queue.task_done()

        imgPub.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()