from piper_sdk.interface.piper_interface_v2 import C_PiperInterface_V2
import time
import cv2 
import cv2.aruco as aruco
import oak_pointcloud as oakp
import numpy
import math

INTERFACE       = "can0"
SPEED_PERCENT   = 10
HOLD_SECONDS    = 3.0
REST_JOINTS_DEG = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

def rotx(angle):
    angle = math.radians(angle)
    mat = [[1, 0, 0],
           [0, math.cos(angle), math.sin(-angle)],
           [0, math.sin(angle), math.cos(angle)]]
    return numpy.array(mat)

def roty(angle):
    angle = math.radians(angle)
    mat = [[math.cos(angle), 0, math.sin(angle)],
           [0, 1, 0],
           [math.sin(-angle), 0, math.cos(angle)]]
    return numpy.array(mat)

def rotz(angle):
    angle = math.radians(angle)
    mat = [[math.cos(angle), math.sin(-angle), 0],
           [math.sin(angle), math.cos(angle), 0],
           [0, 0, 1]]
    return numpy.array(mat)
    

from piper_rest_position import can_interface_state, wait_for_joint_feedback, wait_for_enable
piper = C_PiperInterface_V2("can0", judge_flag=False)

# piper.CreateCanBus("can0", "socketcan", 1000000)
piper.ConnectPort()
piper.EnableArm(7,0X02)
wait_for_enable(piper, 3.0)
piper.ModeCtrl(0x01, 0x01, 50, 0x00)
piper.JointCtrl(0,0,0,0,0,0)

# end = piper.GetArmEndPoseMsgs()
# # print(end)
# # piper.ModeCtrl(0x01, 0x00, 50, 0x00)
# # piper.EndPoseCtrl(300000,100000,300000,0,90000,0)
# camx, camy, camz = oakp.main()
# print(f"x:{camx}, y:{camy}, z:{camz}")
# print(end)

# target_cam = numpy.array([[camx*1000], [camy*1000], [camz*1000], [1]])

# id_mat = numpy.eye(4)

# camera_ee = numpy.eye(4)
# camera_ee[:3, 3] = [-80, 0, 0]

# ee_tcp = numpy.eye(4)
# ee_tcp[:3, 3] = [0, 0, -95]

# tcp_ee = numpy.eye(4)
# tcp_ee[:3, 3] = [0, 0, 95]

# tcp_base = numpy.eye(4)
# rot_mat = roty(85)
# trans_mat = [56.127, 0, 213.266+95]

# camera_tcp = numpy.eye(4)
# camera_tcp[:3, 3] = [-80, 0, -95]

# tcp_base[:3, :3] = rot_mat
# tcp_base[:3, 3] = trans_mat

# target_base = tcp_base @ camera_tcp @ target_cam
# print(target_base)
# target_base= (target_base*1000)
# x_final = int(target_base[0])
# y_final = int(target_base[1])
# z_final = int(target_base[2])
# print(target_base)

# piper.ModeCtrl(0x01, 0x00, 30, 0x00)
# piper.EndPoseCtrl(x_final,y_final,z_final,0,84999,0)
# time.sleep(10)
# piper.ModeCtrl(0x01, 0x01, 50, 0x00)
# piper.JointCtrl(0,0,0,0,0,0)

piper.ModeCtrl(0x01, 0x00, 30, 0x00)
piper.EndPoseCtrl(150000,0,213000,0,84999,0)
time.sleep(20)
piper.EndPoseCtrl(250000,0,213000,0,84999,0)
time.sleep(20)
piper.EndPoseCtrl(350000,0,213000,0,84999,0)
time.sleep(20)
piper.EndPoseCtrl(450000,0,213000,0,84999,0)
time.sleep(20)
piper.EndPoseCtrl(550000,0,213000,0,84999,0)
time.sleep(20)
piper.EndPoseCtrl(650000,0,213000,0,84999,0)
time.sleep(20)
piper.ModeCtrl(0x01, 0x01, 50, 0x00)
piper.JointCtrl(0,0,0,0,0,0)