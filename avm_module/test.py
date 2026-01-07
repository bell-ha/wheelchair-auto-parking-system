import numpy as np
import cv2
import os

# --- 설정 ---
CHECKERBOARD = (6, 9)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
SAVE_PATH = 'data/calib_result.npz'

if not os.path.exists('data'):
    os.makedirs('data')

objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

objpoints = [] 
imgpoints = [] 

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("1단계: 자동 캘리브레이션 데이터 수집")
print(" - 'Space': 프레임 캡처")
print(" - 'c': 캘리브레이션 계산 및 수동 튜닝 모드 진입")
print(" - 'q': 종료")

# 자동 캘리브레이션 변수
K, D = None, None
calibrated = False

while True:
    ret, frame = cap.read()
    if not ret: break
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 체커보드 찾기
    ret_corners, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, 
                                                     cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

    display_frame = frame.copy()
    if ret_corners:
        cv2.drawChessboardCorners(display_frame, CHECKERBOARD, corners, ret_corners)

    cv2.imshow('Step 1: Auto Calibration (Press Space to Capture)', display_frame)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord(' '):
        if ret_corners:
            corners2 = cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners2.reshape(1, -1, 2))
            print(f"📷 캡처 완료! 현재 데이터 수: {len(imgpoints)}")
        else:
            print("❌ 체커보드가 보이지 않습니다.")

    elif key == ord('c'):
        if len(imgpoints) > 10:
            print("⏳ 어안 렌즈 캘리브레이션 계산 중...")
            K = np.zeros((3, 3))
            D = np.zeros((4, 1))
            rms, _, _, _, _ = cv2.fisheye.calibrate(
                objpoints, imgpoints, gray.shape[::-1], K, D,
                flags=cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_FIX_SKEW,
                criteria=criteria
            )
            print(f"✅ 보정 완료! RMS Error: {rms:.4f}")
            cv2.destroyWindow('Step 1: Auto Calibration (Press Space to Capture)')
            break
        else:
            print(f"❌ 데이터가 부족합니다. (현재 {len(imgpoints)}/10개 최소 필요)")

    elif key == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        exit()

# --- 2단계: 수동 튜닝 모드 ---
win_name = 'Step 2: Fine Tuner (S: Save | Q: Quit)'
cv2.namedWindow(win_name)

# 초기값 설정
initial_f = int(K[0, 0])
initial_cx = int(K[0, 2])
initial_cy = int(K[1, 2])
initial_k1 = int(D[0, 0] * 100 + 500) # -5.0 ~ 5.0 범위를 0 ~ 1000 슬라이더로 매핑

cv2.createTrackbar('f_scale', win_name, initial_f, 2000, lambda x: None)
cv2.createTrackbar('cx', win_name, initial_cx, 1280, lambda x: None)
cv2.createTrackbar('cy', win_name, initial_cy, 720, lambda x: None)
cv2.createTrackbar('k1', win_name, initial_k1, 1000, lambda x: None)
cv2.createTrackbar('balance', win_name, 50, 100, lambda x: None)

print("\n2단계: 수동 튜닝 모드 진입")
print("💡 슬라이더를 조절하여 격자가 수평/수직이 되도록 맞추세요.")
print("💡 's'를 눌러 최종 결과 저장 및 종료")

while True:
    ret, frame = cap.read()
    if not ret: break

    # 슬라이더 값 읽기
    f = cv2.getTrackbarPos('f_scale', win_name)
    cx = cv2.getTrackbarPos('cx', win_name)
    cy = cv2.getTrackbarPos('cy', win_name)
    k1 = (cv2.getTrackbarPos('k1', win_name) - 500) / 100.0
    bal = cv2.getTrackbarPos('balance', win_name) / 100.0

    # 튜닝된 파라미터 적용
    K_tuned = K.copy()
    K_tuned[0, 0], K_tuned[1, 1] = f, f
    K_tuned[0, 2], K_tuned[1, 2] = cx, cy
    
    D_tuned = D.copy()
    D_tuned[0, 0] = k1

    # 보정 맵 생성
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K_tuned, D_tuned, (w, h), np.eye(3), balance=bal
    )
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K_tuned, D_tuned, np.eye(3), new_K, (w, h), cv2.CV_16SC2
    )
    undistorted = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)

    # 격자 그리기
    grid_img = undistorted.copy()
    for x in range(0, w, 80):
        cv2.line(grid_img, (x, 0), (x, h), (0, 255, 0), 1)
    for y in range(0, h, 80):
        cv2.line(grid_img, (0, y), (w, y), (0, 255, 0), 1)
    cv2.line(grid_img, (w//2, 0), (w//2, h), (0, 0, 255), 2) # 중앙 십자선
    cv2.line(grid_img, (0, h//2), (w, h//2), (0, 0, 255), 2)

    # 화면 결합 및 정보 표시
    res_orig = cv2.resize(frame, (640, 360))
    res_tuned = cv2.resize(grid_img, (640, 360))
    display = np.hstack((res_orig, res_tuned))
    
    cv2.putText(display, f"F:{f} K1:{k1:.2f} Bal:{bal:.2f}", (660, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow(win_name, display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        np.savez(SAVE_PATH, mtx=K_tuned, dist=D_tuned, new_mtx=new_K)
        print(f"💾 최종 보정 파라미터 저장 완료! -> {SAVE_PATH}")
        break
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()