#!/usr/bin/env python3

import argparse
import os
import sys
import time

from piper_sdk import C_PiperInterface


JOINT_LIMITS_DEG = {
    1: (-150.0, 150.0),
    2: (0.0, 180.0),
    3: (-170.0, 0.0),
    4: (-100.0, 100.0),
    5: (-70.0, 70.0),
    6: (-120.0, 120.0),
}


def deg_to_sdk_units(value_deg: float) -> int:
    return int(round(value_deg * 1000.0))


def sdk_units_to_deg(value_sdk: int) -> float:
    return value_sdk / 1000.0


def clamp_joint_deg(joint_index: int, value_deg: float) -> float:
    low, high = JOINT_LIMITS_DEG[joint_index]
    return max(low, min(high, value_deg))


def can_interface_state(interface: str) -> str:
    path = f"/sys/class/net/{interface}/operstate"
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return handle.read().strip()
    except OSError:
        return "missing"


def wait_for_joint_feedback(robot: C_PiperInterface, timeout_s: float) -> object | None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        joint_msg = robot.GetArmJointMsgs()
        if getattr(joint_msg, "time_stamp", 0) > 0:
            return joint_msg
        time.sleep(0.05)
    return None


def wait_for_enable(robot: C_PiperInterface, timeout_s: float) -> list[bool] | None:
    deadline = time.time() + timeout_s
    last_status = None
    while time.time() < deadline:
        last_status = robot.GetArmEnableStatus()
        if last_status and all(last_status):
            return last_status
        time.sleep(0.1)
    return last_status


def joint_state_to_list(joint_msg: object) -> list[int]:
    state = joint_msg.joint_state
    return [
        int(state.joint_1),
        int(state.joint_2),
        int(state.joint_3),
        int(state.joint_4),
        int(state.joint_5),
        int(state.joint_6),
    ]


def stream_joint_command(
    robot: C_PiperInterface,
    joint_targets_sdk: list[int],
    duration_s: float,
    period_s: float,
) -> None:
    deadline = time.time() + duration_s
    while time.time() < deadline:
        robot.JointCtrl(*joint_targets_sdk)
        time.sleep(period_s)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Small, reversible one-joint motion test for a Piper arm over SocketCAN."
    )
    parser.add_argument("--interface", default="can0", help="CAN interface name, default: can0")
    parser.add_argument("--joint", type=int, default=1, choices=range(1, 7), help="Joint index 1-6")
    parser.add_argument(
        "--delta-deg",
        type=float,
        default=3.0,
        help="How far to nudge the joint in degrees, default: 3.0",
    )
    parser.add_argument(
        "--speed-percent",
        type=int,
        default=10,
        help="Joint move speed percentage 0-100, default: 10",
    )
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=1.5,
        help="How long to keep sending the target command, default: 1.5",
    )
    parser.add_argument(
        "--return-seconds",
        type=float,
        default=1.5,
        help="How long to keep sending the return command, default: 1.5",
    )
    parser.add_argument(
        "--feedback-timeout",
        type=float,
        default=3.0,
        help="Seconds to wait for live joint feedback, default: 3.0",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually send the motion. Without this flag the script only prints the plan.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print("Piper single-joint motion test")
    print(f"CAN interface: {args.interface}")
    print(f"Joint under test: J{args.joint}")
    print(f"Requested delta: {args.delta_deg:.3f} deg")
    print()

    interface_state = can_interface_state(args.interface)
    if interface_state == "missing":
        print(f"CAN interface {args.interface} was not found.")
        return 1
    if interface_state != "up":
        print(f"CAN interface {args.interface} is {interface_state}, not up.")
        print(f"Bring it up first, for example: sudo ip link set {args.interface} up type can bitrate 1000000")
        return 1

    # Skip the SDK's eager bitrate probe and rely on live feedback instead.
    try:
        robot = C_PiperInterface(args.interface, judge_flag=False)
    except Exception as exc:
        print(f"Failed to open Piper CAN interface on {args.interface}: {exc}")
        print("If you are running locally, confirm can0 is up and your user can access SocketCAN.")
        return 1
    robot.ConnectPort()

    try:
        joint_msg = wait_for_joint_feedback(robot, args.feedback_timeout)
        if joint_msg is None:
            print("No live joint feedback arrived from the arm.")
            print("This means we should not send motion commands yet.")
            return 1

        current_joints_sdk = joint_state_to_list(joint_msg)
        current_joints_deg = [sdk_units_to_deg(value) for value in current_joints_sdk]
        enable_status = robot.GetArmEnableStatus()

        print("Current joint angles:")
        for index, value_deg in enumerate(current_joints_deg, start=1):
            print(f"  J{index}: {value_deg:.3f} deg")
        print(f"Motor enable state: {enable_status}")

        joint_idx = args.joint - 1
        requested_target_deg = current_joints_deg[joint_idx] + args.delta_deg
        target_deg = clamp_joint_deg(args.joint, requested_target_deg)
        if target_deg != requested_target_deg:
            print(f"Requested target was outside SDK joint limits, clamped to {target_deg:.3f} deg.")

        target_joints_sdk = current_joints_sdk[:]
        target_joints_sdk[joint_idx] = deg_to_sdk_units(target_deg)

        print()
        print("Planned motion:")
        print(f"  Start J{args.joint}: {current_joints_deg[joint_idx]:.3f} deg")
        print(f"  Target J{args.joint}: {target_deg:.3f} deg")
        print(f"  Speed percent: {args.speed_percent}")
        print(f"  Hold seconds: {args.hold_seconds}")
        print(f"  Return seconds: {args.return_seconds}")

        if not args.execute:
            print()
            print("Dry run only. Re-run with --execute to send the motion.")
            return 0

        print()
        print("Enabling the arm and switching to CAN joint-control mode...")
        robot.EnableArm(7, 0x02)
        enabled = wait_for_enable(robot, 2.0)
        print(f"Enable state after request: {enabled}")

        robot.MotionCtrl_2(0x01, 0x01, args.speed_percent, 0x00, 0, 0x00)
        time.sleep(0.2)

        print(f"Sending small move on J{args.joint}...")
        stream_joint_command(robot, target_joints_sdk, args.hold_seconds, 0.05)
        time.sleep(0.3)

        after_move = wait_for_joint_feedback(robot, 1.0)
        if after_move is not None:
            moved_joints_deg = [sdk_units_to_deg(value) for value in joint_state_to_list(after_move)]
            print(f"Feedback J{args.joint} after move: {moved_joints_deg[joint_idx]:.3f} deg")

        print("Returning to the starting joint angle...")
        stream_joint_command(robot, current_joints_sdk, args.return_seconds, 0.05)
        time.sleep(0.3)

        after_return = wait_for_joint_feedback(robot, 1.0)
        if after_return is not None:
            returned_joints_deg = [sdk_units_to_deg(value) for value in joint_state_to_list(after_return)]
            print(f"Feedback J{args.joint} after return: {returned_joints_deg[joint_idx]:.3f} deg")

        print()
        print("Motion sequence completed.")
        print("If the joint angle changed in feedback and the arm physically moved, CAN control is working.")
        return 0
    finally:
        robot.DisconnectPort()


if __name__ == "__main__":
    sys.exit(main())
