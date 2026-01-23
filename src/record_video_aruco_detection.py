import cv2
import numpy as np
import os

def get_aruco_detector():
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
    parameters = cv2.aruco.DetectorParameters()
    return cv2.aruco.ArucoDetector(aruco_dict, parameters)

def analyze_dual_videos(left_path, rear_path):
    # 1. 두 영상 파일 열기
    cap_l = cv2.VideoCapture(left_path)
    cap_r = cv2.VideoCapture(rear_path)
    
    if not cap_l.isOpened() or not cap_r.isOpened():
        print("⚠️ 파일을 열 수 없습니다. 경로를 확인하세요.")
        return

    # 2. 저장 설정을 위한 정보 획득 (왼쪽 영상 기준)
    fps = cap_l.get(cv2.CAP_PROP_FPS)
    w_l = int(cap_l.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_l = int(cap_l.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w_r = int(cap_r.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_r = int(cap_r.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 두 영상을 가로로 합칠 때의 최종 크기 계산
    # 높이를 맞추기 위해 h_min 기준 리사이즈를 고려한 출력 규격
    h_min = min(h_l, h_r)
    total_w = int(w_l * h_min / h_l) + int(w_r * h_min / h_r)
    
    output_path = "data/detected_combined.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (total_w, h_min))

    detector = get_aruco_detector()
    print(f"🔍 동시 분석 시작: {output_path} 저장 중...")

    while True:
        ret_l, frame_l = cap_l.read()
        ret_r, frame_r = cap_r.read()

        # 두 영상 중 하나라도 끝나면 종료
        if not ret_l or not ret_r:
            break

        # 3. 각 프레임에서 마커 감지
        for frame in [frame_l, frame_r]:
            corners, ids, _ = detector.detectMarkers(frame)
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        # 4. 화면 합치기 (이전 실시간 코드와 동일한 방식)
        f_l_res = cv2.resize(frame_l, (int(w_l * h_min / h_l), h_min))
        f_r_res = cv2.resize(frame_r, (int(w_r * h_min / h_r), h_min))
        combined = cv2.hconcat([f_l_res, f_r_res])

        # 결과 저장
        out.write(combined)
        
        # 확인용 출력 (1280 해상도로 조절)
        display_scale = 1280 / combined.shape[1]
        display_frame = cv2.resize(combined, (0, 0), fx=display_scale, fy=display_scale)
        cv2.imshow("Dual Video Analysis (Simultaneous)", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap_l.release()
    cap_r.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"✅ 분석 완료: {output_path}")

if __name__ == "__main__":
    analyze_dual_videos("data/left.mp4", "data/rear.mp4")