import cv2
import os

# 폴더 생성
os.makedirs('data/left', exist_ok=True)
os.makedirs('data/rear', exist_ok=True)

cap_rear = cv2.VideoCapture(1) # 후면으로 저장 매번 변동
cap_left = cv2.VideoCapture(0) # 좌측

count = 0
print("작동 가이드: 's'를 누르면 양쪽 사진이 동시 저장됩니다. 'q'는 종료.")

while True:
    ret_r, frame_r = cap_rear.read()
    ret_l, frame_l = cap_left.read()

    if not ret_r or not ret_l:
        break

    cv2.imshow('REAR (0) - Press S to Save', frame_r)
    cv2.imshow('LEFT (1) - Press S to Save', frame_l)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('s'):
        # 이미지 저장
        cv2.imwrite(f'data/rear/img_{count}.jpg', frame_r)
        cv2.imwrite(f'data/left/img_{count}.jpg', frame_l)
        print(f"[{count}] 사진 저장 완료!")
        count += 1
    elif key & 0xFF == ord('q'):
        break

cap_rear.release()
cap_left.release()
cv2.destroyAllWindows()