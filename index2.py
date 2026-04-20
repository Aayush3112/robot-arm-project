import cv2
import numpy as np
import glob

# Checkerboard settings (YOU HAVE 9x7 squares → 8x6 inner corners)
CHECKERBOARD = (8, 6)
square_size = 0.025  # meters

# Prepare object points
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= square_size

objpoints = []
imgpoints_l = []
imgpoints_r = []

# Load images
left_images = sorted(glob.glob('calib_images/left_*.png'))
right_images = sorted(glob.glob('calib_images/right_*.png'))

print("Total pairs:", len(left_images))

for imgL_path, imgR_path in zip(left_images, right_images):

    # Load as grayscale (VERY IMPORTANT)
    imgL = cv2.imread(imgL_path, 0)
    imgR = cv2.imread(imgR_path, 0)

    if imgL is None or imgR is None:
        print("❌ Failed to load image:", imgL_path)
        continue

    # Detect corners
    retL, cornersL = cv2.findChessboardCorners(imgL, CHECKERBOARD)
    retR, cornersR = cv2.findChessboardCorners(imgR, CHECKERBOARD)

    if retL and retR:
        # Improve corner accuracy
        cornersL = cv2.cornerSubPix(
            imgL, cornersL, (11,11), (-1,-1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )
        cornersR = cv2.cornerSubPix(
            imgR, cornersR, (11,11), (-1,-1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )

        objpoints.append(objp)
        imgpoints_l.append(cornersL)
        imgpoints_r.append(cornersR)

print("Valid pairs used:", len(objpoints))

# ❌ If zero → stop
if len(objpoints) == 0:
    print("❌ No valid pairs found. Check your dataset.")
    exit()

image_size = imgL.shape[::-1]

# ===============================
# STEP 1: Calibrate each camera
# ===============================

retL, mtxL, distL, _, _ = cv2.calibrateCamera(
    objpoints, imgpoints_l, image_size, None, None
)

retR, mtxR, distR, _, _ = cv2.calibrateCamera(
    objpoints, imgpoints_r, image_size, None, None
)

# ===============================
# STEP 2: Stereo calibration
# ===============================

ret, mtxL, distL, mtxR, distR, R, T, E, F = cv2.stereoCalibrate(
    objpoints,
    imgpoints_l,
    imgpoints_r,
    mtxL,
    distL,
    mtxR,
    distR,
    image_size,
    flags=cv2.CALIB_FIX_INTRINSIC
)

# ===============================
# PRINT RESULTS
# ===============================

print("\n===== FINAL RESULTS =====")

print("\nLeft Camera Matrix:\n", mtxL)
print("\nRight Camera Matrix:\n", mtxR)

print("\nLeft Distortion:\n", distL)
print("\nRight Distortion:\n", distR)

print("\nTranslation (T):\n", T)
print("\nRotation (R):\n", R)