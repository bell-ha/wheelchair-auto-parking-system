import cv2
import numpy as np
import time
import math

# =========================
# UI 설정 그대로
# =========================
ROWS = 8
COLS = 11
CHECKER_WIDTH_MM = 15
DICT = cv2.aruco.DICT_5X5_250
START_ID = 0

# UI에 없어서 "가정": 보드 출력 도구에서 marker 크기를 알면 그 값으로 바꾸세요.
MARKER_LENGTH_MM = 11  # <- 여기만 보드와 동일하게!

# 단위: meters (아무 단위여도 되지만 일관되게)
SQUARE_LENGTH = 0.02
MARKER_LENGTH = 0.014

# =========================
# 캡처 설정
# =========================
CAM_INDEX = 0
FRAME_W = 1280
FRAME_H = 720

MIN_CHARUCO_CORNERS = 6   # 이 이상 잡히면 저장 허용
MIN_SAMPLES = 25           # 최소 샘플(권장 40~80)

# fisheye calibrate flags
FISHEYE_FLAGS = (
    cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC +
    cv2.fisheye.CALIB_FIX_SKEW
)
CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-7)

def now_str():
    return time.strftime("%Y-%m-%d %H:%M:%S")

def format_mat(M):
    return np.array2string(M, separator=', ')

def compute_reproj_error_fisheye(objp_list, imgp_list, rvecs, tvecs, K, D):
    per_view = []
    total_sum = 0.0
    total_cnt = 0

    for i in range(len(objp_list)):
        objp = objp_list[i].reshape(-1, 1, 3).astype(np.float64)  # (N,1,3)
        imgp = imgp_list[i].reshape(-1, 1, 2).astype(np.float64)  # (N,1,2)
        proj, _ = cv2.fisheye.projectPoints(objp, rvecs[i], tvecs[i], K, D)
        err = np.linalg.norm(proj - imgp, axis=2)  # (N,1)
        mean_err = float(np.mean(err))
        per_view.append(mean_err)
        total_sum += float(np.sum(err))
        total_cnt += err.shape[0]

    mean_all = total_sum / max(total_cnt, 1)
    return mean_all, per_view

# =========================
# 보드/디텍터 준비
# =========================
dictionary = cv2.aruco.getPredefinedDictionary(DICT)

try:
    board = cv2.aruco.CharucoBoard((COLS, ROWS), SQUARE_LENGTH, MARKER_LENGTH, dictionary)
except Exception:
    board = cv2.aruco.CharucoBoard_create(COLS, ROWS, SQUARE_LENGTH, MARKER_LENGTH, dictionary)

# StartId(기본 0이면 그대로)
if START_ID != 0:
    try:
        board.ids = (board.ids + START_ID).astype(board.ids.dtype)
    except Exception:
        pass

params = cv2.aruco.DetectorParameters()
use_new_detector = hasattr(cv2.aruco, "ArucoDetector")
detector = cv2.aruco.ArucoDetector(dictionary, params) if use_new_detector else None

# =========================
# 캡처 시작
# =========================
cap = cv2.VideoCapture(CAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

objpoints = []
imgpoints = []
img_size = None
count = 0

print("====================================================")
print("[CHARUCO FISHEYE CALIBRATION - CAPTURE MODE]")
print(f"- time              : {now_str()}")
print(f"- cam index         : {CAM_INDEX}")
print(f"- frame size        : {FRAME_W} x {FRAME_H}")
print("----------------------------------------------------")
print("[BOARD PARAMS]")
print(f"- board(mm)         : 200 x 150 (참고용)")
print(f"- rows x cols       : {ROWS} x {COLS}")
print(f"- square(mm)        : {CHECKER_WIDTH_MM}")
print(f"- marker(mm)        : {MARKER_LENGTH_MM}  (보드와 동일해야 함)")
print(f"- dict              : DICT_5X5_250")
print(f"- start id          : {START_ID}")
print("----------------------------------------------------")
print("[KEY GUIDE]")
print("  Space : capture (if enough charuco corners)")
print("  r     : reset captured samples")
print("  ESC   : finish & calibrate")
print("====================================================")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] camera read failed")
        break

    if img_size is None:
        img_size = (frame.shape[1], frame.shape[0])  # (w,h)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 1) 마커 검출
    if use_new_detector:
        markerCorners, markerIds, rejected = detector.detectMarkers(gray)
    else:
        markerCorners, markerIds, rejected = cv2.aruco.detectMarkers(gray, dictionary, parameters=params)

    disp = frame.copy()
    n_markers = 0 if markerIds is None else len(markerIds)

    charucoCorners, charucoIds = None, None
    n_charuco = 0

    if markerIds is not None and len(markerIds) > 0:
        cv2.aruco.drawDetectedMarkers(disp, markerCorners, markerIds)

        # 2) 차루코 코너 보간
        retval, charucoCorners, charucoIds = cv2.aruco.interpolateCornersCharuco(
            markerCorners, markerIds, gray, board
        )
        if charucoIds is not None:
            n_charuco = len(charucoIds)
        if charucoCorners is not None and charucoIds is not None and n_charuco > 0:
            cv2.aruco.drawDetectedCornersCharuco(disp, charucoCorners, charucoIds, (0,255,0))

    # HUD
    cv2.putText(disp, f"Markers:{n_markers}  CharucoCorners:{n_charuco}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)

    ready = (n_charuco >= MIN_CHARUCO_CORNERS)
    if ready:
        cv2.putText(disp, "Ready (SPACE to capture)", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)
    else:
        cv2.putText(disp, f"Need >= {MIN_CHARUCO_CORNERS} corners", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

    cv2.imshow("Charuco Capture", disp)
    key = cv2.waitKey(1) & 0xFF

    # Space 저장
    if key == ord(' '):
        print("----------------------------------------------------")
        print("[CAPTURE LOG]")
        print(f"- time              : {now_str()}")
        if ready and charucoCorners is not None and charucoIds is not None:
            ids = charucoIds.flatten().astype(int)
            objp = board.chessboardCorners[ids, :].astype(np.float32)   # (Nc,3)
            imgp = charucoCorners.astype(np.float32)                    # (Nc,1,2)

            objpoints.append(objp.reshape(1, -1, 3))
            imgpoints.append(imgp.reshape(-1, 1, 2))
            count += 1

            print(f"[OK] sample index      : {count}")
            print(f"- markers detected     : {n_markers}")
            print(f"- charuco corners used : {len(ids)}")
        else:
            print("[FAILED] not enough charuco corners / no detection")
        print("----------------------------------------------------")

    # reset
    elif key == ord('r'):
        objpoints.clear()
        imgpoints.clear()
        count = 0
        print("----------------------------------------------------")
        print("[CAPTURE LOG]")
        print(f"- time              : {now_str()}")
        print("[RESET] all samples cleared")
        print("----------------------------------------------------")

    # ESC 종료
    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()

# =========================
# 캘리브레이션
# =========================
print("====================================================")
print("[CALIBRATION LOG]")
print(f"- time               : {now_str()}")
print(f"- captured samples   : {count}")
print(f"- min required       : {MIN_SAMPLES}")
print("====================================================")

if count < MIN_SAMPLES:
    print("[FAILED] Not enough samples. Capture more frames.")
    raise SystemExit(0)

K = np.zeros((3, 3), dtype=np.float64)
D = np.zeros((4, 1), dtype=np.float64)
rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(count)]
tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(count)]

image_size = (img_size[0], img_size[1])  # (w,h)

print("----------------------------------------------------")
print("[CALIBRATION LOG]")
print(f"- image_size (w,h)    : {image_size}")
print(f"- fisheye flags       : {FISHEYE_FLAGS}")
print(f"- criteria            : {CRITERIA}")
print("----------------------------------------------------")

rms, _, _, _, _ = cv2.fisheye.calibrate(
    objpoints,
    imgpoints,
    image_size,
    K,
    D,
    rvecs,
    tvecs,
    FISHEYE_FLAGS,
    CRITERIA
)

mean_all, per_view = compute_reproj_error_fisheye(objpoints, imgpoints, rvecs, tvecs, K, D)

print("====================================================")
print("[PER-VIEW ERROR LOG] (mean pixel error per sample)")
for i, e in enumerate(per_view, start=1):
    print(f"- view {i:03d} : {e:.4f} px")
print("====================================================")

print("====================================================")
print("[FINAL RESULT]")
print(f"- RMS Error (fisheye.calibrate) : {rms:.6f}")
print(f"- Mean Reprojection Error       : {mean_all:.6f} px")
print("")
print("[Camera Matrix K]")
print(format_mat(K))
print("")
print("[Distortion D] (k1,k2,k3,k4)")
print(format_mat(D))
print("====================================================")

# 복붙용 출력 (당신 코드 스타일)
print("\n[PASTE THIS INTO YOUR CODE]")
print("K = np.array(" + format_mat(K.astype(np.float32)) + ", dtype=np.float32)")
print("D = np.array(" + format_mat(D.reshape(-1).astype(np.float32)) + ", dtype=np.float32)")
