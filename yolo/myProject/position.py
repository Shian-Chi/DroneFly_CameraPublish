import numpy as np
from math import *
from mavros_msgs.msg import Altitude, GlobalPositionTarget

def matrix_calcul(n):
    matrix = np.mat(n)
    ans = (np.linalg.det((matrix)))
    print(ans)
    return ans


'''
n = np.array([[1, 3], [1, 5]])
matrix_calcul(n)
'''


class vector():
    def __init__(self):
        self.x = np.double(0.0)
        self.y = np.double(0.0)
        self.z = np.double(0.0)
    

class horizontalTargetPositioning():
    def __init__(self):
        self.zeroPos = vector()
        self.firPos = vector()
        self.groundTargetPos = vector()
        self.targetHeading = vector()

    def groundTargetPostion(self):
        self.groundTargetPos.x = (tan(self.firPos.y)*self.firPos.x - tan(self.zeroPos.y)*self.zeroPos.x) / tan(self.firPos.y) - tan(self.zeroPos.y)
        self.groundTargetPos.y = (tan(self.firPos.y)*self.firPos.y - tan(self.zeroPos.y)*self.zeroPos.y) / tan(self.firPos.y) - tan(self.zeroPos.y)
        self.groundTargetPos.z = self.zeroPos.z - self.zeroPos.y/cos(self.zeroPos.z) - self.groundTargetPos.x - self.zeroPos.x

    def pitch_yaw_degreesAdd(self,dronePitch, droneYaw, motorPitch, motorYaw):
        self.targetHeading.y = dronePitch + motorPitch
        self.targetHeading.z = droneYaw + motorYaw
        return self.targetHeading.y, self.targetHeading.z
    
        def posUpdata(x, y, z):
        self.zero.x = self.firPos.x
        self.zero.y = self.firPos.y
        self.zero.z = self.firPos.z
        self.firPos.x = x
        self.firPos.y = y
        self.firPos.z = z
    
    def GimbalAngleUpdata(x, y, z):
        self.zeroTargetAngles.x = self.firTargetAngles.x
        self.zeroTargetAngles.y = self.firTargetAngles.y
        self.zeroTargetAngles.z = self.firTargetAngles.z
        self.firTargetAngles.x = x
        self.firTargetAngles.y = y
        self.firTargetAngles.z = z
        
        
class verticalTargetPositioning():
    def __init__(self):
        self.zeroPos = vector()
        self.firPos = vector()
        self.groundTargetPos = vector()
        self.zeroTargetAngles = vector()
        self.firTargetAngles = vector()
        self.D_xy = (self.firPos - self.zeroPos) / (tan(self.firTargetAngles.y) - tan(self.zeroTargetAngles.y))
        
    def groundTargetPostion(self):
        self.groundTargetPos.x = self.firPos.x + D_xy * cos(self.zeroTargetAngles.z)
        self.groundTargetPos.y = self.firPos.y + D_xy * sin(self.zeroTargetAngles.z)
        self.groundTargetPos.z = self.firPos.z - D_xy * tan(self.firTargetAngles.y)
        
    def posUpdata(x, y, z):
        self.zero.x = self.firPos.x
        self.zero.y = self.firPos.y
        self.zero.z = self.firPos.z
        self.firPos.x = x
        self.firPos.y = y
        self.firPos.z = z
    
    def GimbalAngleUpdata(x, y, z):
        self.zeroTargetAngles.x = self.firTargetAngles.x
        self.zeroTargetAngles.y = self.firTargetAngles.y
        self.zeroTargetAngles.z = self.firTargetAngles.z
        self.firTargetAngles.x = x
        self.firTargetAngles.y = y
        self.firTargetAngles.z = z