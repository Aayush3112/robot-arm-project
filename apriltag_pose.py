#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np


# Stereo calibration from index2.py
MTX_L = np.array([
    [417.88845642, 0.0, 316.4677369],
    [0.0, 416.47191776, 198.93029587],
    [0.0, 0.0, 1.0],
], dtype=np.float32)

DIST_L = np.array(
    [[0.10995314, -0.47603549, -0.00240217, -0.00083497, 0.4770429]],
    dtype=np.float32,
)

MTX_R = np.array([
    [411.74092405, 0.0, 319.71428904],
    [0.0, 410.75476209, 195.5214454],
    [0.0, 0.0, 1.0],
], dtype=np.float32)

DIST_R = np.array(
    [[0.12898212, -0.59878115, -0.0007473, -0.00209199, 0.61215407]],
    dtype=np.float32,
)

R = np.array([
    [9.99480321e-01, -6.25381209e-04, -3.22288075e-02],
    [8.36121545e-04, 9.99978357e-01, 6.52581876e-03],
    [3.22240289e-02, -6.54937463e-03, 9.99459213e-01],
], dtype=np.float32)

T = np.array([[-0.08243488], [0.00011714], [-0.00438158]], dtype=np.float32)

IMAGE_SIZE = (640, 400)
BASELINE_M = float(np.linalg.norm(T))


def build_rectification():
    r1, r2, p1, p2, _, _, _ = cv2.stereoRectify(
        MTX_L,
        DIST_L,
        MTX_R,
        DIST_R,
        IMAGE_SIZE,
        R,
        T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0,
    )

    map_l_x, map_l_y = cv2.initUndistortRectifyMap(
        MTX_L, DIST_L, r1, p1, IMAGE_SIZE, cv2.CV_32FC1
    )
    map_r_x, map_r_y = cv2.initUndistortRectifyMap(
        MTX_R, DIST_R, r2, p2, IMAGE_SIZE, cv2.CV_32FC1
    )
    return p1, map_l_x, map_l_y, map_r_x, map_r_y


def build_stereo_matcher():
    num_disparities = 16 * 8
    block_size = 7
    matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=8 * block_size * block_size,
        P2=32 * block_size * block_size,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=2,
        preFilterCap=31,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    return matcher


def build_detector():
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detectors = []
    for family_name, family_id in [
        ("36h11", cv2.aruco.DICT_APRILTAG_36h11),
        ("36h10", cv2.aruco.DICT_APRILTAG_36h10),
        ("25h9", cv2.aruco.DICT_APRILTAG_25h9),
        ("16h5", cv2.aruco.DICT_APRILTAG_16h5),
    ]:
        dictionary = cv2.aruco.getPredefinedDictionary(family_id)
        detectors.append((family_name, cv2.aruco.ArucoDetector(dictionary, params)))
    return detectors


def detect_apriltag(gray, detectors):
    equalized = cv2.equalizeHist(gray)
    for candidate in (gray, equalized):
        for family_name, detector in detectors:
            corners, ids, _ = detector.detectMarkers(candidate)
            if ids is not None and len(ids) > 0:
                return family_name, corners, ids
    return None


def compute_xyz_from_disparity(corners, disparity, projection_matrix):
    polygon = np.round(corners).astype(np.int32)
    x_min = max(0, int(np.min(polygon[:, 0])))
    x_max = min(disparity.shape[1] - 1, int(np.max(polygon[:, 0])))
    y_min = max(0, int(np.min(polygon[:, 1])))
    y_max = min(disparity.shape[0] - 1, int(np.max(polygon[:, 1])))
    if x_min >= x_max or y_min >= y_max:
        return None

    patch = disparity[y_min:y_max + 1, x_min:x_max + 1]
    mask = np.zeros((y_max - y_min + 1, x_max - x_min + 1), dtype=np.uint8)
    shifted = polygon - np.array([x_min, y_min], dtype=np.int32)
    cv2.fillConvexPoly(mask, shifted, 1)

    disp_vals = patch[mask == 1]
    valid = np.isfinite(disp_vals) & (disp_vals > 1.0)
    if not np.any(valid):
        return None

    disparity_px = float(np.median(disp_vals[valid]))
    fx = float(projection_matrix[0, 0])
    fy = float(projection_matrix[1, 1])
    cx = float(projection_matrix[0, 2])
    cy = float(projection_matrix[1, 2])

    u = float(np.mean(corners[:, 0]))
    v = float(np.mean(corners[:, 1]))

    z = fx * BASELINE_M / disparity_px
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return x, y, z, disparity_px


def build_pipeline():
    pipeline = dai.Pipeline()

    mono_left = pipeline.create(dai.node.MonoCamera)
    mono_right = pipeline.create(dai.node.MonoCamera)
    mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
    mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

    xout_left = pipeline.create(dai.node.XLinkOut)
    xout_right = pipeline.create(dai.node.XLinkOut)
    xout_left.setStreamName("left")
    xout_right.setStreamName("right")

    mono_left.out.link(xout_left.input)
    mono_right.out.link(xout_right.input)
    return pipeline


def main():
    projection_matrix, map_l_x, map_l_y, map_r_x, map_r_y = build_rectification()
    matcher = build_stereo_matcher()
    detectors = build_detector()

    pipeline = build_pipeline()
    cv2.namedWindow("AprilTag Stereo Depth", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("AprilTag Stereo Depth", 1280, 800)

    with dai.Device(pipeline) as device:
        left_q = device.getOutputQueue(name="left", maxSize=4, blocking=False)
        right_q = device.getOutputQueue(name="right", maxSize=4, blocking=False)

        print("Using custom stereo calibration from index2.py")
        print("Press 'q' to quit")

        smoothed_positions = {}

        while True:
            left_frame = left_q.get().getCvFrame()
            right_frame = right_q.get().getCvFrame()

            rect_left = cv2.remap(left_frame, map_l_x, map_l_y, cv2.INTER_LINEAR)
            rect_right = cv2.remap(right_frame, map_r_x, map_r_y, cv2.INTER_LINEAR)

            disparity = matcher.compute(rect_left, rect_right).astype(np.float32) / 16.0
            detection = detect_apriltag(rect_left, detectors)

            display = cv2.cvtColor(rect_left, cv2.COLOR_GRAY2BGR)
            cv2.putText(
                display,
                "Custom stereo calibration",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
            )

            if detection is None:
                cv2.putText(
                    display,
                    "No AprilTag detected",
                    (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 165, 255),
                    2,
                )
            else:
                family_name, corners, ids = detection
                cv2.putText(
                    display,
                    f"Family: {family_name}",
                    (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2,
                )

                for i, corner in enumerate(corners):
                    pts = corner[0]
                    tag_id = int(ids[i][0])
                    result = compute_xyz_from_disparity(pts, disparity, projection_matrix)
                    if result is None:
                        label = f"ID:{tag_id} NO DEPTH"
                        color = (0, 0, 255)
                    else:
                        x, y, z, disp = result
                        prev = smoothed_positions.get(tag_id)
                        if prev is not None:
                            alpha = 0.35
                            x = alpha * x + (1.0 - alpha) * prev[0]
                            y = alpha * y + (1.0 - alpha) * prev[1]
                            z = alpha * z + (1.0 - alpha) * prev[2]
                        smoothed_positions[tag_id] = (x, y, z)
                        distance = float(np.sqrt(x * x + y * y + z * z))
                        label = (
                            f"ID:{tag_id} X:{x:.2f} Y:{y:.2f} "
                            f"Z:{z:.2f} D:{distance:.2f}"
                        )
                        color = (0, 255, 0)
                        print(
                            f"Tag[{tag_id}] -> X:{x:.3f} m, Y:{y:.3f} m, "
                            f"Z:{z:.3f} m, disparity:{disp:.2f}px"
                        )

                    center = tuple(np.mean(pts, axis=0).astype(np.int32))
                    cv2.polylines(display, [pts.astype(np.int32)], True, color, 2)
                    cv2.circle(display, center, 4, color, -1)
                    cv2.putText(
                        display,
                        label,
                        (max(center[0] - 150, 10), max(center[1] - 12, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2,
                    )

            cv2.imshow("AprilTag Stereo Depth", display)
            if cv2.waitKey(1) == ord("q"):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
