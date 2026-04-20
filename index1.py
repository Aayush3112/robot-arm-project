import depthai as dai
import cv2
import os

# Create folder
os.makedirs("calib_images", exist_ok=True)

pipeline = dai.Pipeline()

monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)

monoLeft.setBoardSocket(dai.CameraBoardSocket.CAM_B)
monoRight.setBoardSocket(dai.CameraBoardSocket.CAM_C)

monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

xoutLeft = pipeline.create(dai.node.XLinkOut)
xoutRight = pipeline.create(dai.node.XLinkOut)

xoutLeft.setStreamName("left")
xoutRight.setStreamName("right")

monoLeft.out.link(xoutLeft.input)
monoRight.out.link(xoutRight.input)

count = 0

with dai.Device(pipeline) as device:

    leftQ = device.getOutputQueue(name="left", maxSize=4, blocking=False)
    rightQ = device.getOutputQueue(name="right", maxSize=4, blocking=False)

    print("Press 's' to save (only when BOTH True)")
    print("Press 'q' to quit")

    while True:
        left = leftQ.get().getCvFrame()
        right = rightQ.get().getCvFrame()

        # ✅ Save RAW before drawing
        left_raw = left.copy()
        right_raw = right.copy()

        # 👉 Already grayscale (no conversion needed)
        grayL = left
        grayR = right

        # Detect checkerboard
        retL, cornersL = cv2.findChessboardCorners(grayL, (8,6))
        retR, cornersR = cv2.findChessboardCorners(grayR, (8,6))

        # Draw for display only
        if retL:
            cv2.drawChessboardCorners(left, (8,6), cornersL, retL)
        if retR:
            cv2.drawChessboardCorners(right, (8,6), cornersR, retR)

        # Status text
        status = f"L:{retL} R:{retR} Count:{count}"
        cv2.putText(left, status, (20,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        cv2.imshow("Left", left)
        cv2.imshow("Right", right)

        key = cv2.waitKey(1)

        # Save only valid pairs
        if key == ord('s'):
            if retL and retR:
                cv2.imwrite(f"calib_images/left_{count}.png", left_raw)
                cv2.imwrite(f"calib_images/right_{count}.png", right_raw)
                print(f"✅ Saved pair {count}")
                count += 1
            else:
                print("❌ Not saved")

        if key == ord('q'):
            break

cv2.destroyAllWindows()