import cv2

def test_all_cameras():
    caps = []
    # 0번부터 4번까지 카메라 오픈 시도
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            caps.append((i, cap))
    
    if not caps:
        print("연결된 카메라가 없습니다.")
        return

    print(f"총 {len(caps)}개의 카메라를 발견했습니다.")

    while True:
        for idx, cap in caps:
            ret, frame = cap.read()
            if ret:
                # 화면에 인덱스 번호 표시
                cv2.putText(frame, f"Camera Index: {idx}", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow(f"Camera {idx}", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    for idx, cap in caps:
        cap.release()
    cv2.destroyAllWindows()

test_all_cameras()