import cv2
import numpy as np

# 1. 렌즈 보정 데이터 로드
data = np.load('data/calib_result_common.npz')
K, D, new_K = data['mtx'], data['dist'], data['new_mtx']

# 2. 캔버스 규격
W, H = 800, 1000
car_x1, car_y1, car_x2, car_y2 = 320, 320, 480, 680

# 목적지 좌표 (REAR 영역: 차량 뒷변에서 캔버스 하단 끝까지)
DST_PTS = np.float32([[320, 680], [480, 680], [800, 1000], [0, 1000]])

points = []
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append([x, y])

def run_integrated_avm():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # 어안 보정 맵
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, (1280, 720), cv2.CV_16SC2)

    print("📍 가이드: 보정된 화면에서 후방 바닥 4점을 클릭하세요 (좌상->우상->우하->좌하)")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # STEP 1: 왜곡 보정
        undistorted = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)
        display_undistorted = undistorted.copy()

        # 점 클릭 표시
        for p in points:
            cv2.circle(display_undistorted, tuple(p), 5, (0, 255, 0), -1)

        # STEP 2: 4점이 찍혔을 때만 합성 수행
        if len(points) == 4:
            src_pts = np.float32(points)
            M = cv2.getPerspectiveTransform(src_pts, DST_PTS)
            
            # 탑뷰 생성
            rear_topview = cv2.warpPerspective(undistorted, M, (W, H))

            # STEP 3: 캔버스 생성 및 색상 유지 합성
            # 기본 배경 (회색)
            canvas = np.full((H, W, 3), 200, dtype=np.uint8)
            
            # 나비넥타이 마스크 생성
            mask = np.zeros((H, W), dtype=np.uint8)
            roi_corners = np.array([[(0, 1000), (320, 680), (480, 680), (800, 1000)]], dtype=np.int32)
            cv2.fillPoly(mask, roi_corners, 255)

            # [핵심] 색상 변질 방지: 배경에서 영상이 들어갈 자리를 검게 파내고 영상을 얹음
            canvas_bg = cv2.bitwise_and(canvas, canvas, mask=cv2.bitwise_not(mask))
            rear_fg = cv2.bitwise_and(rear_topview, rear_topview, mask=mask)
            canvas = cv2.add(canvas_bg, rear_fg)

            # 차량 영역 덮기 (최종)
            cv2.rectangle(canvas, (car_x1, car_y1), (car_x2, car_y2), (0, 0, 0), -1)
            cv2.putText(canvas, "VEHICLE", (355, 510), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow("AVM Final Result", canvas)
        
        cv2.imshow("Step 1: Click 4 Points on Floor", display_undistorted)
        cv2.setMouseCallback("Step 1: Click 4 Points on Floor", mouse_callback)

        key = cv2.waitKey(1)
        if key == ord('q'): break
        if key == ord('r'): points.clear() # 'r' 누르면 좌표 초기화

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_integrated_avm()