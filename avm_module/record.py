import cv2
import numpy as np
import datetime

def set_max_resolution(cap):
    # 카메라가 지원하는 최대 해상도로 설정
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 10000)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 10000)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return width, height

def run_dual_recorder():
    cap0 = cv2.VideoCapture(0)
    cap1 = cv2.VideoCapture(1)

    # 각 카메라를 최대 해상도로 설정
    w0, h0 = set_max_resolution(cap0)
    w1, h1 = set_max_resolution(cap1)
    
    print(f"카메라 0 해상도: {w0}x{h0}")
    print(f"카메라 1 해상도: {w1}x{h1}")

    # --- [녹화 설정] ---
    # 코덱 설정 (mp4 사용)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # 현재 시간을 활용한 파일명 생성
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 영상 저장 객체 생성 (파일명, 코덱, 프레임수, 해상도)
    # 프레임수는 일반적으로 20.0~30.0으로 설정합니다.
    out0 = cv2.VideoWriter(f'video_cam0_{timestamp}.mp4', fourcc, 20.0, (w0, h0))
    out1 = cv2.VideoWriter(f'video_cam1_{timestamp}.mp4', fourcc, 20.0, (w1, h1))

    print("🔴 녹화 시작... 'q'를 누르면 종료하고 저장합니다.")

    while True:
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()

        if not ret0 or not ret1:
            break

        # 각각의 영상을 원본 해상도 그대로 저장
        out0.write(frame0)
        out1.write(frame1)

        # --- [화면 출력용 병합] ---
        # 원본 해상도가 크므로 모니터 확인용으로만 합치고 리사이즈합니다.
        h_min = min(frame0.shape[0], frame1.shape[0])
        f0_resized = cv2.resize(frame0, (int(frame0.shape[1] * h_min / frame0.shape[0]), h_min))
        f1_resized = cv2.resize(frame1, (int(frame1.shape[1] * h_min / frame1.shape[0]), h_min))
        
        combined = cv2.hconcat([f0_resized, f1_resized])
        
        display_scale = 1280 / combined.shape[1]
        display_frame = cv2.resize(combined, (0, 0), fx=display_scale, fy=display_scale)

        cv2.imshow("Dual Camera Recording (Press 'q' to stop)", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("💾 녹화 종료 및 저장 중...")
            break

    # 모든 자원 해제 (반드시 release 해야 영상이 정상 저장됩니다)
    cap0.release()
    cap1.release()
    out0.release()
    out1.release()
    cv2.destroyAllWindows()
    print("✅ 저장 완료.")

if __name__ == "__main__":
    run_dual_recorder()