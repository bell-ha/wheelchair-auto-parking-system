import numpy as np
import cv2
import os

# --- 설정 ---
CHECKERBOARD = (6, 9)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
SAVE_PATH = 'data/calib_right.npz'

if not os.path.exists('data'):
    os.makedirs('data')

objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

objpoints = [] 
imgpoints = [] 

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("1단계: 데이터 수집 (생략 가능)")
print(" - 'Space': 체커보드 캡처")
print(" - 'c': 수동 튜닝 모드 강제 진입 (데이터 없어도 가능)")
print(" - 'q': 종료")

# 자동 캘리브레이션 초기값
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
        else:
            print("❌ 체커보드가 감지되지 않습니다.")

    elif key == ord('c'):
        if len(imgpoints) >= 10:
            print("⏳ 자동 계산 중...")
            rms, K, D, _, _ = cv2.fisheye.calibrate(
                objpoints, imgpoints, gray.shape[::-1], None, None,
                flags=cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_FIX_SKEW,
                criteria=criteria
            )
            print(f"✅ 자동 계산 완료! RMS: {rms:.4f}")
        else:
            print("⚠️ 데이터 부족으로 기본값 모드로 진입합니다.")
            # 기본값 설정 (1280x720 해상도 기준)
            K = np.array([[400.0, 0, 640.0], [0, 400.0, 360.0], [0, 0, 1]], dtype=np.float32)
            D = np.array([[-0.05], [0.0], [0.0], [0.0]], dtype=np.float32)
        
        cv2.destroyAllWindows()
        break

    elif key == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        exit()

# --- 2단계: 수동 튜닝 모드 (에러 방지 강화) ---
win_name = 'Step 2: Fine Tuner'
cv2.namedWindow(win_name)

# 초기 슬라이더 값 계산
initial_f = int(K[0, 0])
initial_cx = int(K[0, 2])
initial_cy = int(K[1, 2])
initial_k1 = int(D[0, 0] * 100 + 500) 

cv2.createTrackbar('f_scale', win_name, initial_f, 2000, lambda x: None)
cv2.createTrackbar('cx', win_name, initial_cx, 1280, lambda x: None)
cv2.createTrackbar('cy', win_name, initial_cy, 720, lambda x: None)
cv2.createTrackbar('k1', win_name, initial_k1, 1000, lambda x: None)
cv2.createTrackbar('balance', win_name, 50, 100, lambda x: None)

print("\n💡 수동 튜닝 팁: 빨간 십자선이 화면 중앙에 오게 하고, 녹색 선이 직선이 되도록 조절하세요.")

while True:
    ret, frame = cap.read()
    if not ret: break

    f = max(1, cv2.getTrackbarPos('f_scale', win_name)) # 0 방지
    cx = cv2.getTrackbarPos('cx', win_name)
    cy = cv2.getTrackbarPos('cy', win_name)
    k1 = (cv2.getTrackbarPos('k1', win_name) - 500) / 100.0
    bal = cv2.getTrackbarPos('balance', win_name) / 100.0

    K_tuned = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float32)
    D_tuned = np.array([[k1], [0], [0], [0]], dtype=np.float32) # 어안 핵심 k1만 조정

    # 보정 적용
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K_tuned, D_tuned, (w, h), np.eye(3), balance=bal)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K_tuned, D_tuned, np.eye(3), new_K, (w, h), cv2.CV_16SC2)
    undistorted = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)

    # 격자 그리기
    grid_img = undistorted.copy()
    for x in range(0, w, 80): cv2.line(grid_img, (x, 0), (x, h), (0, 255, 0), 1)
    for y in range(0, h, 80): cv2.line(grid_img, (0, y), (w, y), (0, 255, 0), 1)
    cv2.line(grid_img, (w//2, 0), (w//2, h), (0, 0, 255), 2)
    cv2.line(grid_img, (0, h//2), (w, h//2), (0, 0, 255), 2)

    # 화면 결합
    res_orig = cv2.resize(frame, (640, 360))
    res_tuned = cv2.resize(grid_img, (640, 360))
    display = np.hstack((res_orig, res_tuned))
    cv2.imshow(win_name, display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        np.savez(SAVE_PATH, mtx=K_tuned, dist=D_tuned, new_mtx=new_K)
        print(f"💾 저장 완료: {SAVE_PATH}")
        break
    elif key == ord('q'): break

cap.release()
cv2.destroyAllWindows()