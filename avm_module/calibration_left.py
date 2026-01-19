import numpy as np
import cv2
import os

# --- [추가: 아르코 마커 및 화질 개선 설정] ---
def get_aruco_detector():
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco.DetectorParameters()
    parameters.adaptiveThreshWinSizeMin = 3
    parameters.adaptiveThreshWinSizeMax = 23
    parameters.minMarkerPerimeterRate = 0.01 # 찌그러진 마커도 잡도록 완화
    return cv2.aruco.ArucoDetector(aruco_dict, parameters)

def enhance_image(img):
    # CLAHE 대비 강화 (외곽 어두운 부분 보정)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    img = cv2.merge((l, a, b))
    img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
    
    # 강력한 샤프닝 (뭉개진 경계 복구)
    gaussian = cv2.GaussianBlur(img, (0, 0), 2.0)
    img = cv2.addWeighted(img, 1.5, gaussian, -0.5, 0)
    return img

detector = get_aruco_detector()

# --- 설정 (기본 유지) ---
CHECKERBOARD = (6, 9)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
SAVE_PATH = 'data/calib_left.npz'

if not os.path.exists('data'):
    os.makedirs('data')

objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

objpoints = [] 
imgpoints = [] 

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# --- 1단계: 데이터 수집 (원본 유지) ---
print("1단계: 데이터 수집 진행 중...")
K = np.array([[500, 0, 640], [0, 500, 360], [0, 0, 1]], dtype=np.float32)
D = np.array([[0.0], [0.0], [0.0], [0.0]], dtype=np.float32)

while True:
    ret, frame = cap.read()
    if not ret: break
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret_corners, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, 
                                                     cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
    display_frame = frame.copy()
    if ret_corners:
        cv2.drawChessboardCorners(display_frame, CHECKERBOARD, corners, ret_corners)
    cv2.imshow('Step 1: Calibration (Press C to Skip)', display_frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):
        if ret_corners:
            corners2 = cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners2.reshape(1, -1, 2))
            print(f"📷 데이터 추가됨! 현재 수: {len(imgpoints)}")
    elif key == ord('c'):
        if len(imgpoints) >= 10:
            rms, K, D, _, _ = cv2.fisheye.calibrate(objpoints, imgpoints, gray.shape[::-1], None, None,
                flags=cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_FIX_SKEW, criteria=criteria)
        else:
            K = np.array([[400.0, 0, 640.0], [0, 400.0, 360.0], [0, 0, 1]], dtype=np.float32)
            D = np.array([[-0.05], [0.0], [0.0], [0.0]], dtype=np.float32)
        cv2.destroyAllWindows()
        break
    elif key == ord('q'):
        cap.release(); cv2.destroyAllWindows(); exit()

# --- 2단계: 수동 튜닝 모드 (마커 감지 및 화질 개선 통합) ---
win_name = 'Step 2: Fine Tuner (ArUco Check)'
cv2.namedWindow(win_name)

initial_f, initial_cx, initial_cy = int(K[0, 0]), int(K[0, 2]), int(K[1, 2])
initial_k1 = int(D[0, 0] * 100 + 500) 

cv2.createTrackbar('f_scale', win_name, initial_f, 2000, lambda x: None)
cv2.createTrackbar('cx', win_name, initial_cx, 1280, lambda x: None)
cv2.createTrackbar('cy', win_name, initial_cy, 720, lambda x: None)
cv2.createTrackbar('k1', win_name, initial_k1, 1000, lambda x: None)
cv2.createTrackbar('balance', win_name, 0, 100, lambda x: None) # 인식률 위해 0 권장

while True:
    ret, frame = cap.read()
    if not ret: break

    f = max(1, cv2.getTrackbarPos('f_scale', win_name))
    cx = cv2.getTrackbarPos('cx', win_name)
    cy = cv2.getTrackbarPos('cy', win_name)
    k1 = (cv2.getTrackbarPos('k1', win_name) - 500) / 100.0
    bal = cv2.getTrackbarPos('balance', win_name) / 100.0

    K_tuned = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float32)
    D_tuned = np.array([[k1], [0], [0], [0]], dtype=np.float32)

    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K_tuned, D_tuned, (w, h), np.eye(3), balance=bal)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K_tuned, D_tuned, np.eye(3), new_K, (w, h), cv2.CV_16SC2)
    undistorted = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)

    # --- [화질 개선 적용 및 마커 감지] ---
    enhanced = enhance_image(undistorted) # 1. 화질 개선
    corners, ids, _ = detector.detectMarkers(enhanced) # 2. 개선된 이미지에서 감지
    
    display_img = undistorted.copy()
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(display_img, corners, ids) # 3. 보정된 화면에 표시
        cv2.putText(display_img, f"ArUco OK: {len(ids)}", (50, 50), 1, 2, (0, 255, 0), 2)

    # 격자 및 가이드
    for x in range(0, w, 80): cv2.line(display_img, (x, 0), (x, h), (100, 100, 100), 1)
    for y in range(0, h, 80): cv2.line(display_img, (0, y), (w, y), (100, 100, 100), 1)
    cv2.line(display_img, (w//2, 0), (w//2, h), (0, 0, 255), 2)
    cv2.line(display_img, (0, h//2), (w, h//2), (0, 0, 255), 2)

    # 화면 결합 및 출력
    res_orig = cv2.resize(frame, (640, 360))
    res_tuned = cv2.resize(display_img, (640, 360))
    cv2.imshow(win_name, np.hstack((res_orig, res_tuned)))

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        np.savez(SAVE_PATH, mtx=K_tuned, dist=D_tuned, new_mtx=new_K)
        print(f"💾 {SAVE_PATH} 저장 완료")
        break
    elif key == ord('q'): break

cap.release(); cv2.destroyAllWindows()