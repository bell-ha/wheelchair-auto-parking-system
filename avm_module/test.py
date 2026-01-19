import cv2
import numpy as np
import os

# --- [설정] 마커 정보 및 카메라 파라미터 ---
MARKER_SIZE = 25.0  # 실제 마커 크기 (cm)

# 광각 카메라를 위한 임의의 캘리브레이션 값
camera_matrix = np.array([[800, 0, 640],
                          [0, 800, 360],
                          [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros((4, 1)) 

def get_aruco_detector():
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
    parameters = cv2.aruco.DetectorParameters()
    parameters.polygonalApproxAccuracyRate = 0.05
    return cv2.aruco.ArucoDetector(aruco_dict, parameters)

def estimate_pose(frame, corners, ids):
    if ids is not None:
        for i in range(len(ids)):
            # 마커의 3D 좌표 정의
            obj_points = np.array([[-MARKER_SIZE/2,  MARKER_SIZE/2, 0],
                                   [ MARKER_SIZE/2,  MARKER_SIZE/2, 0],
                                   [ MARKER_SIZE/2, -MARKER_SIZE/2, 0],
                                   [-MARKER_SIZE/2, -MARKER_SIZE/2, 0]], dtype=np.float32)
            
            # PnP 알고리즘 수행
            _, rvec, tvec = cv2.solvePnP(obj_points, corners[i], camera_matrix, dist_coeffs)
            
            # --- [에러 수정 포인트] 거리 계산 방식 변경 ---
            # np.linalg.norm은 벡터의 크기(L2 norm)를 계산해줍니다. 인덱스 에러로부터 안전합니다.
            distance = np.linalg.norm(tvec)
            
            # 각도(Yaw) 계산
            rmat, _ = cv2.Rodrigues(rvec)
            yaw = np.arctan2(rmat[1, 0], rmat[0, 0]) * 180 / np.pi
            
            # 화면에 정보 표시
            cv2.aruco.drawDetectedMarkers(frame, [corners[i]], ids[i])
            # 마커의 좌측 상단 모서리 좌표 추출
            c = corners[i][0][0].astype(int) 
            
            # 가독성을 위해 텍스트 배경 처리 또는 선명한 색상 사용
            text = f"D: {distance:.1f}cm, Y: {yaw:.1f}deg"
            cv2.putText(frame, text, (c[0], c[1] - 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame

def analyze_dual_videos(left_path, rear_path):
    cap_l = cv2.VideoCapture(left_path)
    cap_r = cv2.VideoCapture(rear_path)
    
    if not cap_l.isOpened() or not cap_r.isOpened():
        print("⚠️ 파일을 열 수 없습니다. data/ 폴더에 left.mp4와 rear.mp4가 있는지 확인하세요.")
        return

    fps = cap_l.get(cv2.CAP_PROP_FPS)
    w_l, h_l = int(cap_l.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap_l.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w_r, h_r = int(cap_r.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap_r.get(cv2.CAP_PROP_FRAME_HEIGHT))

    h_min = min(h_l, h_r)
    total_w = int(w_l * h_min / h_l) + int(w_r * h_min / h_r)
    
    output_path = "data/detected_pose_combined.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (total_w, h_min))

    detector = get_aruco_detector()
    print(f"🔍 거리/각도 분석 시작: {output_path} 저장 중...")

    while True:
        ret_l, frame_l = cap_l.read()
        ret_r, frame_r = cap_r.read()

        if not ret_l or not ret_r:
            break

        # 포즈 추정 수행
        corners_l, ids_l, _ = detector.detectMarkers(frame_l)
        estimate_pose(frame_l, corners_l, ids_l)

        corners_r, ids_r, _ = detector.detectMarkers(frame_r)
        estimate_pose(frame_r, corners_r, ids_r)

        # 화면 합치기
        f_l_res = cv2.resize(frame_l, (int(w_l * h_min / h_l), h_min))
        f_r_res = cv2.resize(frame_r, (int(w_r * h_min / h_r), h_min))
        combined = cv2.hconcat([f_l_res, f_r_res])

        out.write(combined)
        
        # 확인용 출력
        display_scale = 1280 / combined.shape[1]
        display_frame = cv2.resize(combined, (0, 0), fx=display_scale, fy=display_scale)
        cv2.imshow("Dual Pose Analysis", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap_l.release()
    cap_r.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"✅ 분석 완료: {output_path}")

if __name__ == "__main__":
    analyze_dual_videos("data/left.mp4", "data/rear.mp4")