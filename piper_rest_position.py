#!/usr/bin/env python3
"""
piper_rest_position.py
Run with:  python3 piper_rest_position.py
Moves the Piper arm to its zero/rest position [0, 0, 0, 0, 0, 0] deg.
"""

import sys
import time

from piper_sdk import C_PiperInterface

INTERFACE       = "can0"
SPEED_PERCENT   = 10
HOLD_SECONDS    = 3.0
REST_JOINTS_DEG = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def can_interface_state(interface: str) -> str:
    try:
        with open(f"/sys/class/net/{interface}/operstate", "r") as f:
            return f.read().strip()
    except OSError:
        return "missing"


def wait_for_joint_feedback(robot: C_PiperInterface, timeout_s: float):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        msg = robot.GetArmJointMsgs()
        if getattr(msg, "time_stamp", 0) > 0:
            return msg
        time.sleep(0.05)
    return None


def wait_for_enable(robot: C_PiperInterface, timeout_s: float):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        status = robot.GetArmEnableStatus()
        if status and all(status):
            return
        time.sleep(0.1)


def main() -> int:
    if can_interface_state(INTERFACE) != "up":
        print(f"ERROR: CAN interface '{INTERFACE}' is not up.")
        return 1

    try:
        robot = C_PiperInterface(INTERFACE, judge_flag=False)
    except Exception as exc:
        print(f"ERROR: Could not open CAN interface: {exc}")
        return 1

    robot.ConnectPort()

    try:
        if wait_for_joint_feedback(robot, timeout_s=3.0) is None:
            print("ERROR: No joint feedback from arm — is it powered on?")
            return 1

        print("Enabling arm...")
        robot.EnableArm(7, 0x02)
        wait_for_enable(robot, timeout_s=2.0)

        target_sdk = [int(round(d * 1000.0)) for d in REST_JOINTS_DEG]

        print("Moving to rest position...")
        robot.MotionCtrl_2(0x01, 0x01, SPEED_PERCENT, 0x00, 0, 0x00)
        time.sleep(0.2)

        deadline = time.time() + HOLD_SECONDS
        while time.time() < deadline:
            robot.JointCtrl(*target_sdk)
            time.sleep(0.05)

        print("Done — arm is at rest position.")
        return 0

    finally:
        robot.DisconnectPort()


if __name__ == "__main__":
    sys.exit(main())
