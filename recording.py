import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2 as cv
from cv_bridge import CvBridge
from yolo.utils.general import check_imshow
import time, os, datetime


# Define a timeout value for recording
timeOutMinutes = 1
timeOutSeconds = 0

class recordingNode(Node):
    def __init__(self):
        super().__init__('recording')
        
        # Timed video
        self.subscription = self.create_subscription(Image, 'camera_image', self.timing, 10)
 
        # Keep recording
#        self.subscription = self.create_subscription(Image, 'camera_image', self.recording, 10)       
        
        # File name confirmation
        self.file_count = 1
        self.name = f'output{self.file_count}.avi'
        self.fileName()
        
        # Settings recording parameters
        self.cv_bridge = CvBridge()
        self.frame_width = 1280  # Set the width of the video
        self.frame_height = 720  # Set film height
        self.fps = 30  # Set the frame rate of the video
        self.fourcc = cv.VideoWriter_fourcc(*'XVID') # *.avi
        self.videoSaveSettings()

        # Check whether the screen can be displayed 
        self.view_img = False
        if self.view_img:
            self.view_img = check_imshow()

        print("Start recording")

        self.start_time = datetime.datetime.now()
        
        
        self.target_duration = datetime.timedelta(minutes=timeOutMinutes,seconds=timeOutSeconds)
        print(self.start_time,self.target_duration)
        self.current_time = 0
        self.elapsed_time = 0
    
    def videoSaveSettings(self):
        self.out = cv.VideoWriter(self.name, self.fourcc, self.fps, (self.frame_width, self.frame_height))
        
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
            cv.destroyAllWindows()
        self.get_logger().info("Recording stopped.")
        self.destroy_node()
        rclpy.shutdown()

    def recording(self, msg):
        try:
            # Convert ROS image messages to OpenCV image
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, 'bgr8')
            
            # Resize the image and write it to the video file
            if cv_image is not None:

                cv_image = cv.resize(cv_image, (self.frame_width, self.frame_height))
                self.out.write(cv_image)
            else:
                print("No Img")

            if self.view_img:
                cv.imshow('Camera Image', cv_image)
                cv.waitKey(1)
                
        except KeyboardInterrupt:
            self.out.release()
            print("Caught KeyboardInterrupt")
        except Exception as e:
            self.out.release() 
            print(f"Error Info: \n{e}")

    # Timing
    def timing(self, msg):
        # Calculate elapsed time since recording start
        self.current_time = datetime.datetime.now()
        self.elapsed_time = self.current_time - self.start_time

        # Check if elapsed time exceeds the timeout value
        if self.elapsed_time >= self.target_duration:
            print("Time out") 
            self.fileName()
            self.out.release()
            self.videoSaveSettings()
            self.start_time = datetime.datetime.now()
        self.recording(msg)
            
    
def main(args=None):
    rclpy.init(args=args)
    recording = recordingNode()
    try:
        rclpy.spin(recording)
    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Stopping recording.")
        recording.stop_recording()
    except Exception as e:
        print("An unexpected error occurred:", str(e))
        recording.stop_recording()

if __name__ == '__main__':
    main()
