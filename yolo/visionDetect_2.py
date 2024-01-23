import torch
import torch.backends.cudnn as cudnn
import numpy as np

from yolo.models import experimental
from yolo.utils.general import check_img_size, check_imshow, non_max_suppression, set_logging
from yolo.utils.torch_utils import select_device
from yolo.myProject.parameter import Parameters
import cv2
from colorama import init, Fore
para = Parameters()

init(autoreset=True)

class YOLO():
    def __init__(self,weights):        
        self.weights = weights
        self.img_size = 640
        self.conf_thres = 0.6
        self.iou_thres = 0.4
        self.agnostic_nms = False
        self.augment = False
        self.classes = None
        
        # Initialize
        set_logging()
        self.device = select_device('0')
        self.half = self.device.type != 'cpu'
        # Load model
        self.model = experimental.attempt_load(self.weights, map_location=self.device)  # load FP32 model
        stride = int(self.model.stride.max())  # model stride
        imgsz = check_img_size(self.img_size, s=stride)  # check img_size
        if self.half:
            self.model.half()  # to FP16
        
        cudnn.benchmark = True  # set True to speed up constant image size inference
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, imgsz, imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once
        print(Fore.GREEN+"YOLO initialization complete")
        
    def yolo(self,img):
        if(len(np.shape(img)) == 3):
            img = np.transpose(img,(2,0,1))
        elif(len(np.shape(img)) == 4):
            img = np.transpose(img,(0,3,1,2))
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        
        if img.ndimension() == 3:
            img = img.unsqueeze(0)


        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = self.model(img, augment=self.augment)[0]
        # Apply NMS
        pred = non_max_suppression(
            pred, self.conf_thres, self.iou_thres, classes=self.classes, agnostic=self.agnostic_nms)

        return pred
    

    def run(self,frame):
        bbox = None
        # '''YOLO Searching'''
        frame = cv2.resize(frame,(640,640))
        bbox = self.yolo(frame)
        for object in bbox[0]:        
            print(object)

'''
if __name__ == '__main__':
    args = get_Argument()
    Tracker = Yolo(args)
    Tracker.run()
'''

 