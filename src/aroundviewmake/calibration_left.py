import numpy as np
import cv2
import os

# --- 설정 및 경로 ---
SAVE_PATH = 'data/calib_left.npz'
if not os.path.exists('data'):
    os.makedirs('data')

# 카메라 초기화
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# 기본값 설정
ret, frame = cap.read()
if not ret:
    print("카메라를 찾을 수 없습니다.")
    exit()

h, w = frame.shape[:2]

# --- 수동 튜닝 윈도우 생성 ---
win_name = 'Manual Calibration Tuner'
cv2.namedWindow(win_name)

# 트랙바 초기값 설정 (중앙값 및 기본 배율)
cv2.createTrackbar('f_scale', win_name, 500, 2000, lambda x: None)  # 초점 거리
cv2.createTrackbar('cx', win_name, w // 2, w, lambda x: None)      # 중심점 X
cv2.createTrackbar('cy', win_name, h // 2, h, lambda x: None)      # 중심점 Y
cv2.createTrackbar('k1', win_name, 500, 1000, lambda x: None)     # 왜곡 계수 (500이 0)
cv2.createTrackbar('balance', win_name, 0, 100, lambda x: None)   # 화면 잘림 조절

print("트랙바를 조절하여 격자를 일직선으로 맞추세요.")
print("'s' 키: 설정 저장 후 종료 / 'q' 키: 저장 없이 종료")

while True:
    ret, frame = cap.read()
    if not ret: break

    # 1. 트랙바 값 읽기
    f = max(1, cv2.getTrackbarPos('f_scale', win_name))
    cx = cv2.getTrackbarPos('cx', win_name)
    cy = cv2.getTrackbarPos('cy', win_name)
    k1 = (cv2.getTrackbarPos('k1', win_name) - 500) / 100.0 # -5.0 ~ 5.0 범위
    bal = cv2.getTrackbarPos('balance', win_name) / 100.0

    # 2. 파라미터 적용 (fisheye 모델)
    K_tuned = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float32)
    D_tuned = np.array([[k1], [0.0], [0.0], [0.0]], dtype=np.float32)

    # 3. 언디스토션 (왜곡 펴기)
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K_tuned, D_tuned, (w, h), np.eye(3), balance=bal)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K_tuned, D_tuned, np.eye(3), new_K, (w, h), cv2.CV_16SC2)
    undistorted = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)

    # 4. 가이드 라인 그리기 (격자무늬)
    display_img = undistorted.copy()
    # 얇은 격자
    for x in range(0, w, 80): cv2.line(display_img, (x, 0), (x, h), (100, 100, 100), 1)
    for y in range(0, h, 80): cv2.line(display_img, (0, y), (w, y), (100, 100, 100), 1)
    # 중앙 십자선 (빨간색)
    cv2.line(display_img, (w//2, 0), (w//2, h), (0, 0, 255), 2)
    cv2.line(display_img, (0, h//2), (w, h//2), (0, 0, 255), 2)

    # 5. 화면 출력 (원본과 비교)
    res_orig = cv2.resize(frame, (640, 360))
    res_tuned = cv2.resize(display_img, (640, 360))
    cv2.imshow(win_name, np.hstack((res_orig, res_tuned)))

    # 키 입력 처리
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        np.savez(SAVE_PATH, mtx=K_tuned, dist=D_tuned, new_mtx=new_K)
        print(f"💾 {SAVE_PATH}에 파라미터가 저장되었습니다!")
        break
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()