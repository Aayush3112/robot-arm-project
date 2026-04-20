from piper_sdk import C_PiperInterface
import time

robot = C_PiperInterface("can0")
robot.ConnectPort()

print("Waiting for feedback...")
for _ in range(20):
    msg = robot.GetArmEndPoseMsgs()
    if msg and msg.end_pose:
        print("End Pose:", msg.end_pose)
        break
    time.sleep(0.1)
else:
    print("No feedback received.")
