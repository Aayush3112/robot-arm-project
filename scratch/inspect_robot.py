from piper_sdk import C_PiperInterface
import inspect
robot = C_PiperInterface("can0")
print("EnableArm signature:", inspect.signature(robot.EnableArm))
