import cv2
import numpy as np

# ==========================================
# 1. 설정: 5x6 체커보드 (내부 코너 기준 4x5)
# ==========================================
# [근거] PDF가 5x6 칸이므로 교차점(코너)은 (가로-1, 세로-1)인 (4, 5)입니다.
CHECKERBOARD = (4, 5) 
subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)

# 3D 실제 세계 좌표 정의 (Z=0)
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

objpoints = [] # 실제 세계 3D 점
imgpoints = [] # 이미지 평면 2D 점

# 카메라 설정
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("--- [일반 체스보드] 캘리브레이션 모드 ---")
print("1. Spacebar: 캡처 (보드에 빨간색/색상 선이 나타날 때)")
print("2. ESC: 촬영 종료 및 계산")
print("---------------------------------------")

while True:
    ret, frame = cap.read()
    if not ret: break
    
    display_frame = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 체스보드 코너 찾기
    # [참고] 광각 렌즈는 CALIB_CB_ADAPTIVE_THRESH 플래그가 도움이 됩니다.
    ret_find, corners = cv2.findChessboardCorners(
        gray, CHECKERBOARD, 
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    
    if ret_find:
        # 코너 정밀화
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), subpix_criteria)
        # 화면에 그리기
        cv2.drawChessboardCorners(display_frame, CHECKERBOARD, corners2, ret_find)
        cv2.putText(display_frame, f"Captured: {len(objpoints)}", (30, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('Chessboard Calibration', display_frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):
        if ret_find:
            objpoints.append(objp)
            imgpoints.append(corners2)
            print(f"[성공] {len(objpoints)}번째 샘플 저장")
        else:
            print("[실패] 보드 코너를 찾을 수 없습니다.")
    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()

# ==========================================
# 2. 어안 렌즈(Fisheye) 모델 계산
# ==========================================
if len(objpoints) >= 15:
    print("\n[계산 중] 광각 렌즈 모델로 계산합니다...")
    
    N_OK = len(objpoints)
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    rvecs = [np.zeros((1, 1, 3), dtype=np.float32) for i in range(N_OK)]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float32) for i in range(N_OK)]

    try:
        # 광각 렌즈(어안) 전용 calibrate 함수
        rms, _, _, _, _ = cv2.fisheye.calibrate(
            objpoints,
            imgpoints,
            gray.shape[::-1],
            K,
            D,
            rvecs,
            tvecs,
            cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_FIX_SKEW,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
        )

        print("\n" + "="*50)
        print("✅ 캘리브레이션 완료")
        print(f"RMS Error: {rms:.4f}")
        print("\nCamera Matrix (K):")
        print(np.array2string(K, separator=', '))
        print("\nDistortion Coefficients (D):")
        print(np.array2string(D, separator=', '))
        print("="*50)
    except Exception as e:
        print(f"오류 발생: {e}")
else:
    print("데이터가 부족합니다 (최소 15장 필요).")