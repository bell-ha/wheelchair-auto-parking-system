import cv2
import numpy as np

# --- 캔버스 및 기준점 설정 ---
W, H = 1200, 1200  # 전체 화면 크기
# 체스판이 고정될 왼쪽 아래 좌표 (여기가 두 영상의 합류 지점이 됩니다)
FIXED_X, FIXED_Y = 200, 800 
CELL_SIZE = 250    # 화면에서 체스판이 차지할 크기

pts_l, pts_r = [], []

def mouse_handler(event, x, y, flags, param):
    pts = param['pts']
    if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
        pts.append([x, y])

# 1. 좌표 수집 (순차적 실행)
for i in [0, 1]:
    cap = cv2.VideoCapture(i)
    name = f"CLICK 4 CORNERS - Cam {i}"
    curr_pts = pts_l if i == 0 else pts_r
    cv2.namedWindow(name)
    cv2.setMouseCallback(name, mouse_handler, {'pts': curr_pts})
    while len(curr_pts) < 4:
        ret, frame = cap.read()
        if not ret: break
        for p in curr_pts: cv2.circle(frame, tuple(p), 5, (0,0,255), -1)
        cv2.imshow(name, frame)
        cv2.waitKey(1)
    cap.release()
    cv2.destroyWindow(name)

# 2. 목적지 좌표 설정 (두 카메라 모두 동일한 FIXED 지점으로 매핑)
# [좌상, 우상, 우하, 좌하] 순서로 클릭한다고 가정
dst_points = np.float32([
    [FIXED_X, FIXED_Y],
    [FIXED_X + CELL_SIZE, FIXED_Y],
    [FIXED_X + CELL_SIZE, FIXED_Y + CELL_SIZE],
    [FIXED_X, FIXED_Y + CELL_SIZE]
])

M_l = cv2.getPerspectiveTransform(np.float32(pts_l), dst_points)
M_r = cv2.getPerspectiveTransform(np.float32(pts_r), dst_points)

# 3. 실시간 합성 실행
cap0, cap1 = cv2.VideoCapture(0), cv2.VideoCapture(1)

while True:
    ret0, frame0 = cap0.read()
    ret1, frame1 = cap1.read()
    if not ret0 or not ret1: break

    # 전체 화면 왜곡 (잘리는 부분 없이 캔버스 크기만큼 출력)
    warped_l = cv2.warpPerspective(frame0, M_l, (W, H))
    warped_r = cv2.warpPerspective(frame1, M_r, (W, H))

    # 두 영상을 합침 (체스판 위치가 동일하므로 정확히 정렬됨)
    # 0이 아닌 부분을 우선시하여 합성
    combined = np.where(warped_r == 0, warped_l, warped_r)
    # 혹은 반투명하게 겹침을 보려면 아래 코드 사용
    # combined = cv2.addWeighted(warped_l, 0.5, warped_r, 0.5, 0)

    # 화면에 기준점 가이드 표시 (확인용)
    cv2.rectangle(combined, (FIXED_X, FIXED_Y), (FIXED_X+CELL_SIZE, FIXED_Y+CELL_SIZE), (0, 255, 0), 2)
    cv2.putText(combined, "FIXED POINT (CHECKERBOARD)", (FIXED_X, FIXED_Y-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow("Unified AVM - Fixed Alignment", combined)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap0.release()
cap1.release()
cv2.destroyAllWindows()