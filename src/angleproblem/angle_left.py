import cv2
import numpy as np

# ==========================================
# 1. 고정 데이터 (K, D, 제어 계수)
# ==========================================
K = np.array([[601.71923257, 0.0, 630.47700714], 
              [0.0, 601.34529853, 367.21223657], 
              [0.0, 0.0, 1.0]], dtype=np.float32)

D = np.array([-0.18495647, 0.02541005, -0.01068433, 0.00321714], dtype=np.float32)

MARKER_SIZE = 0.25
SENSITIVITY = 1.6        # 검증된 감도
INSTALL_ANGLE = 113.0    # 카메라 설치 위치 보정
INSTALL_OFFSET = 50.84   # 사용자 측정 영점 보정값 (휠체어 북쪽=0)

# ArUco 설정
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
obj_points = np.array([[-0.125, 0.125, 0], [0.125, 0.125, 0], 
                       [0.125, -0.125, 0], [-0.125, -0.125, 0]], dtype=np.float32)

cap = cv2.VideoCapture("90_l.mp4") # 혹은 동영상 파일 경로

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is not None:
        for i in range(len(ids)):
            # [Step 1] 어안 렌즈 좌표 평면 복원
            undistorted = cv2.fisheye.undistortPoints(corners[i].reshape(-1, 1, 2), K, D, P=K)
            
            # [Step 2] 포즈 추정
            retval, rvec, tvec = cv2.solvePnP(obj_points, undistorted, K, None, flags=cv2.SOLVEPNP_ITERATIVE)
            
            # [Step 3] 오일러 각도 추출
            rmat, _ = cv2.Rodrigues(rvec)
            sy = np.sqrt(rmat[0,0]**2 + rmat[1,0]**2)
            raw_yaw = np.arctan2(-rmat[2,0], sy) * 180 / np.pi
            
            # [Step 4] 통합 보정 수식 (영점 50.84 반영)
            current_total = (raw_yaw * SENSITIVITY) + INSTALL_ANGLE
            final_yaw = current_total - INSTALL_OFFSET

            # [Step 5] 시각화 (축 길이를 0.07로 조절하여 경고 최소화)
            cv2.drawFrameAxes(frame, K, None, rvec, tvec, 0.07)
            
            bx, by = int(corners[i][0][3][0]), int(corners[i][0][3][1])
            cv2.putText(frame, f"WHEELCHAIR YAW: {final_yaw:.1f}", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, f"Dist: {np.linalg.norm(tvec):.2f}m", (bx, by + 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.imshow('Final Fixed Offset Tracker', frame)
    if cv2.waitKey(30) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()