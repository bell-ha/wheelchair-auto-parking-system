import cv2
import numpy as np

# ==========================================
# 1. 설정: 5x6 체커보드 (내부 코너 기준이므로 4x5)
# ==========================================
# 주의: 칸 수가 5x6이면, 내부 교차점(코너)은 가로-1, 세로-1인 4x5개입니다.
CHECKERBOARD = (5, 4) 
subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)

# 3D 실제 세계 좌표 정의
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

objpoints = [] # 실제 세계 3D 점
imgpoints = [] # 이미지 평면 2D 점

# 카메라 연결 (0번)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("--- 캘리브레이션 촬영 모드 ---")
print("1. Spacebar: 사진 캡처 (보드가 인식될 때만 저장됨)")
print("2. ESC: 촬영 종료 및 행렬 계산 시작")
print("----------------------------")

count = 0
while True:
    ret, frame = cap.read()
    if not ret: break
    
    display_frame = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 실시간으로 보드가 잘 보이는지 확인용 그리기
    ret_find, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
    
    if ret_find:
        cv2.drawChessboardCorners(display_frame, CHECKERBOARD, corners, ret_find)
        cv2.putText(display_frame, "Ready to Capture!", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Calibration Capture', display_frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    # Space 키: 현재 프레임 저장
    if key == ord(' '):
        if ret_find:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), subpix_criteria)
            imgpoints.append(corners2)
            count += 1
            print(f"[성공] {count}번째 데이터 저장 완료")
        else:
            print("[실패] 체커보드가 인식되지 않았습니다. 각도를 조절하세요.")
            
    # ESC 키: 루프 종료 및 계산
    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()

# ==========================================
# 2. 어안 렌즈(Fisheye) 캘리브레이션 계산
# ==========================================
if count > 10:
    print("\n계산 중... 잠시만 기다려주세요.")
    N_OK = len(objpoints)
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    rvecs = [np.zeros((1, 1, 3), dtype=np.float32) for i in range(N_OK)]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float32) for i in range(N_OK)]

    try:
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

        print("\n" + "="*30)
        print("최종 캘리브레이션 결과")
        print("="*30)
        print(f"RMS Error: {rms:.4f}")
        print("\nCamera Matrix (K):")
        print(np.array2string(K, separator=', '))
        print("\nDistortion Coefficients (D):")
        print(np.array2string(D, separator=', '))
        print("="*30)
        print("\n이 값을 복사해서 ArUco 코드에 넣으세요.")
        
    except Exception as e:
        print(f"\n[오류] 캘리브레이션 계산 실패: {e}")
else:
    print(f"\n[실패] 데이터가 부족합니다 (현재 {count}장). 최소 10장 이상 캡처하세요.")