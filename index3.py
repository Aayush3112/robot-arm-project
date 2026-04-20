from piper_sdk import C_PiperInterface
import time

robot = C_PiperInterface("can0")

robot.EnableArm()
print("✅ Robot Enabled")

time.sleep(2)

for i in range(200):
    robot.MotionCtrl_2(0x01, 500, 0, 0, 0, 0)
    time.sleep(0.01)

print("✅ Motion sent")