import cv2
import numpy as np

# 캘리브레이션 데이터 (기존과 동일)
K = np.array([[601.71923257, 0.0, 630.47700714], [0.0, 601.34529853, 367.21223657], [0.0, 0.0, 1.0]], dtype=np.float32)
D = np.array([-0.18495647, 0.02541005, -0.01068433, 0.00321714], dtype=np.float32)

MARKER_SIZE = 0.25
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

obj_points = np.array([[-MARKER_SIZE/2, MARKER_SIZE/2, 0], [MARKER_SIZE/2, MARKER_SIZE/2, 0],
                       [MARKER_SIZE/2, -MARKER_SIZE/2, 0], [-MARKER_SIZE/2, -MARKER_SIZE/2, 0]], dtype=np.float32)

# 트랙바 콜백 함수 (아무것도 안 함)
def nothing(x): pass

cv2.namedWindow('Tuning Mode')
# 트랙바 생성 (값의 범위는 정수로만 가능하므로 실제 적용 시 나눗셈 사용)
cv2.createTrackbar('Sensitivity(x10)', 'Tuning Mode', 16, 30, nothing) # 1.6 의미
cv2.createTrackbar('Offset', 'Tuning Mode', 10, 20, nothing)          # 10 = 0도 의미 (0~20)
cv2.createTrackbar('Pause', 'Tuning Mode', 0, 1, nothing)             # 1이면 정지

video_path = "90_r.mp4"
cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cv2.createTrackbar('Frame', 'Tuning Mode', 0, total_frames - 1, nothing)

while True:
    # 정지 상태 확인
    pause = cv2.getTrackbarPos('Pause', 'Tuning Mode')
    
    if not pause:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # 무한 반복
            continue
        # 현재 프레임 위치를 슬라이더에 동기화
        current_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        cv2.setTrackbarPos('Frame', 'Tuning Mode', current_pos)
    else:
        # 정지 상태일 때는 선택된 프레임 번호로 고정해서 읽기
        target_frame = cv2.getTrackbarPos('Frame', 'Tuning Mode')
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()

    # 트랙바 값 읽기
    sens = cv2.getTrackbarPos('Sensitivity(x10)', 'Tuning Mode') / 10.0
    offset = cv2.getTrackbarPos('Offset', 'Tuning Mode') - 10.0 # 10을 0도로 기준

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is not None:
        for i in range(len(ids)):
            undistorted_corners = cv2.fisheye.undistortPoints(corners[i].reshape(-1, 1, 2), K, D, P=K)
            retval, rvec, tvec = cv2.solvePnP(obj_points, undistorted_corners, K, None, flags=cv2.SOLVEPNP_ITERATIVE)
            
            rmat, _ = cv2.Rodrigues(rvec)
            sy = np.sqrt(rmat[0,0]**2 + rmat[1,0]**2)
            raw_yaw = np.arctan2(-rmat[2,0], sy) * 180 / np.pi
            
            # 슬라이더 값이 적용된 최종 Yaw
            final_yaw = (raw_yaw * sens) + offset

            # 시각화 (D=None 적용하여 마커 위에 정확히 표시)
            cv2.drawFrameAxes(frame, K, None, rvec, tvec, 0.1)
            bx, by = int(corners[i][0][3][0]), int(corners[i][0][3][1])
            cv2.putText(frame, f"Sens:{sens} Offset:{offset}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"Yaw: {final_yaw:.1f}", (bx, by + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow('Tuning Mode', frame)
    if cv2.waitKey(30) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()