import cv2

def show_dense_grid_camera(cam_idx=0):
    cap = cv2.VideoCapture(cam_idx)
    
    if not cap.isOpened():
        print(f"카메라 {cam_idx}번을 열 수 없습니다.")
        return

    # 해상도 확인
    ret, frame = cap.read()
    if not ret: return
    h, w, _ = frame.shape

    while True:
        ret, frame = cap.read()
        if not ret: break

        # 1. 매우 촘촘한 보조선 (50픽셀 단위, 아주 연하게)
        for x in range(0, w, 50):
            cv2.line(frame, (x, 0), (x, h), (60, 60, 60), 1)
        for y in range(0, h, 50):
            cv2.line(frame, (0, y), (w, y), (60, 60, 60), 1)

        # 2. 메인 격자 (100픽셀 단위, 조금 더 진하게)
        for x in range(0, w, 100):
            cv2.line(frame, (x, 0), (x, h), (100, 100, 100), 1)
        for y in range(0, h, 100):
            cv2.line(frame, (0, y), (w, y), (100, 100, 100), 1)

        # 3. 화면 8등분 가이드라인 (노란색)
        for i in range(1, 8):
            dx = int(w * i / 8)
            dy = int(h * i / 8)
            cv2.line(frame, (dx, 0), (dx, h), (0, 255, 255), 1)
            cv2.line(frame, (0, dy), (w, dy), (0, 255, 255), 1)

        # 4. 정중앙 십자선 (빨간색, 굵게)
        cv2.line(frame, (int(w/2), 0), (int(w/2), h), (0, 0, 255), 2)
        cv2.line(frame, (0, int(h/2)), (w, int(h/2)), (0, 0, 255), 2)

        # 현재 상태 표시
        cv2.putText(frame, f"REAR/SIDE CAM: {w}x{h}", (10, h - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow("High-Precision Alignment Grid", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    show_dense_grid_camera(0)