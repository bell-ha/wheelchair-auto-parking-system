import cv2
import numpy as np

def get_aruco_detector():
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
    parameters = cv2.aruco.DetectorParameters()
    return cv2.aruco.ArucoDetector(aruco_dict, parameters)

detector = get_aruco_detector()

def set_max_resolution(cap):
    # 카메라가 지원하는 최대 해상도를 얻기 위해 큰 값을 입력해봅니다.
    # 카메라는 자신이 지원하는 가장 큰 근접 해상도로 자동 설정됩니다.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 10000)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 10000)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return width, height

def run_dual_detector():
    cap0 = cv2.VideoCapture(0)
    cap1 = cv2.VideoCapture(1)

    # 각 카메라를 최대 해상도로 설정
    w0, h0 = set_max_resolution(cap0)
    w1, h1 = set_max_resolution(cap1)
    
    print(f"카메라 0 해상도: {w0}x{h0}")
    print(f"카메라 1 해상도: {w1}x{h1}")

    while True:
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()

        if not ret0 or not ret1:
            break

        # 마커 감지 및 그리기
        for frame in [frame0, frame1]:
            corners, ids, _ = detector.detectMarkers(frame)
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        # 듀얼 카메라 화면 합치기 
        # (최대 해상도 사용 시 화면이 너무 커서 모니터를 벗어날 수 있으므로 출력만 리사이즈)
        # 만약 원본 크기로 보고 싶다면 아래 resize 줄을 지우고 combined 사용
        h_min = min(frame0.shape[0], frame1.shape[0])
        f0_resized = cv2.resize(frame0, (int(frame0.shape[1] * h_min / frame0.shape[0]), h_min))
        f1_resized = cv2.resize(frame1, (int(frame1.shape[1] * h_min / frame1.shape[0]), h_min))
        
        combined = cv2.hconcat([f0_resized, f1_resized])
        
        # 화면 출력을 위해 적당한 크기로 조절 (예: 전체 가로 1280)
        display_scale = 1280 / combined.shape[1]
        display_frame = cv2.resize(combined, (0, 0), fx=display_scale, fy=display_scale)

        cv2.imshow("Max FOV Dual Mode", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap0.release()
    cap1.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_dual_detector()