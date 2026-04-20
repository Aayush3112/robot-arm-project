#!/usr/bin/env python3

import sys
import time

import oak_pointcloud as oakp
from piper_sdk import C_PiperInterface

from piper_rest_position import (
    INTERFACE,
    REST_JOINTS_DEG,
    SPEED_PERCENT,
    can_interface_state,
    wait_for_enable,
    wait_for_joint_feedback,
)


MOVE_SETTLE_SECONDS = 2.5
GRIPPER_SETTLE_SECONDS = 1.5
TARGET_OFFSET_X_M = 0.0
TARGET_OFFSET_Y_M = 0.0
TARGET_OFFSET_Z_M = 0.0
GRIPPER_OPEN_CMD = 80000
GRIPPER_CLOSE_CMD = 0


def meters_to_sdk_units(value_m: float) -> int:
    return int(round(value_m * 1_000_000.0))


def connect_robot() -> C_PiperInterface:
    if can_interface_state(INTERFACE) != "up":
        raise RuntimeError(f"CAN interface '{INTERFACE}' is not up.")

    robot = C_PiperInterface(INTERFACE, judge_flag=False)
    robot.ConnectPort()

    if wait_for_joint_feedback(robot, timeout_s=3.0) is None:
        robot.DisconnectPort()
        raise RuntimeError("No joint feedback from arm. Check power and CAN.")

    robot.EnableArm(7, 0x02)
    wait_for_enable(robot, timeout_s=2.0)
    return robot


def move_to_cartesian(robot: C_PiperInterface, x_m: float, y_m: float, z_m: float) -> None:
    print(f"Moving to target: X={x_m:.3f} m, Y={y_m:.3f} m, Z={z_m:.3f} m")
    robot.MotionCtrl_2(0x01, 0x02, SPEED_PERCENT, 0x00, 0, 0x00)
    time.sleep(0.2)
    robot.EndPoseCtrl(
        meters_to_sdk_units(x_m),
        meters_to_sdk_units(y_m),
        meters_to_sdk_units(z_m),
        0,
        0,
        0,
    )
    time.sleep(MOVE_SETTLE_SECONDS)


def set_gripper(robot: C_PiperInterface, command: int, label: str) -> None:
    print(label)
    robot.GripperCtrl(command)
    time.sleep(GRIPPER_SETTLE_SECONDS)


def move_to_rest(robot: C_PiperInterface) -> None:
    print("Returning to rest position")
    target_sdk = [int(round(value * 1000.0)) for value in REST_JOINTS_DEG]
    robot.MotionCtrl_2(0x01, 0x01, SPEED_PERCENT, 0x00, 0, 0x00)
    time.sleep(0.2)

    deadline = time.time() + 3.0
    while time.time() < deadline:
        robot.JointCtrl(*target_sdk)
        time.sleep(0.05)


def main() -> int:
    print("Detecting AprilTag pose from OAK-D")
    pose = oakp.main(return_on_detection=True, stable_frames=5)
    if pose is None:
        print("No valid 3D tag position was returned.")
        return 1

    x, y, z = pose
    target_x = x + TARGET_OFFSET_X_M
    target_y = y + TARGET_OFFSET_Y_M
    target_z = z + TARGET_OFFSET_Z_M

    print(
        "Using target position "
        f"X={target_x:.3f} m, Y={target_y:.3f} m, Z={target_z:.3f} m"
    )

    robot = None
    try:
        robot = connect_robot()
        move_to_cartesian(robot, target_x, target_y, target_z)
        set_gripper(robot, GRIPPER_OPEN_CMD, "Opening gripper")
        set_gripper(robot, GRIPPER_CLOSE_CMD, "Closing gripper")
        move_to_rest(robot)
        print("Grasp sequence complete")
        return 0
    except Exception as exc:
        print(f"Sequence failed: {exc}")
        return 1
    finally:
        if robot is not None:
            robot.DisconnectPort()


if __name__ == "__main__":
    sys.exit(main())
