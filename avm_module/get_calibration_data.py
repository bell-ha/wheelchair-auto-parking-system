import cv2
import os

# 1. 저장할 폴더 생성 (기존 폴더와 구분되게 새로 설정)
save_path = 'data/calib_images'
os.makedirs(save_path, exist_ok=True)

# 2. 카메라 연결 (카메라가 하나라면 0, 안 나오면 1이나 2로 변경)
cap = cv2.VideoCapture(0)

# 카메라 해상도를 사양서에 맞게 1080p로 설정 (지원하는 경우)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

count = 0
print(f"--- 캘리브레이션 촬영 시작 ---")
print(f"저장 경로: {save_path}")
print("가이드: 체커보드를 다양한 위치와 각도에서 보여주며 's'를 누르세요. 'q'는 종료.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("카메라를 찾을 수 없습니다.")
        break

    # 화면에 현재 찍은 장수 표시
    display_frame = frame.copy()
    cv2.putText(display_frame, f"Saved: {count}", (30, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('Calibration Capture', display_frame)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('s'):
        # 이미지 저장
        img_name = f'{save_path}/img_{count}.jpg'
        cv2.imwrite(img_name, frame)
        print(f"[{count}] 사진 저장 완료: {img_name}")
        count += 1
    elif key & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()