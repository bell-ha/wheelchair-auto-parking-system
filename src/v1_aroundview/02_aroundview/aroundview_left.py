import cv2
import numpy as np
import os

# --- [아르코 마커 설정] ---
def get_aruco_detector():
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco.DetectorParameters()
    parameters.adaptiveThreshWinSizeMin = 3
    parameters.adaptiveThreshWinSizeMax = 23
    parameters.minMarkerPerimeterRate = 0.02
    return cv2.aruco.ArucoDetector(aruco_dict, parameters)

detector = get_aruco_detector()

# 1. 왜곡 보정 데이터 로드
try:
    data = np.load('data/calib_left.npz')
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        data['mtx'], data['dist'], np.eye(3), data['new_mtx'], (1280, 720), cv2.CV_16SC2
    )
except:
    print("⚠️ 왜곡 보정 데이터가 없습니다.")
    map1, map2 = None, None

# 2. 타겟 좌표 (8번 CAM ORIGIN 추가)
DST_PTS = np.float32([
    [0, 180], [0, 270], [0, 360], [0, 540],   # 왼쪽 외곽 라인 4점
    [200, 540],                               # 차량 왼쪽 뒤 모서리
    [0, 720], [200, 720],                     # 하단 구역 2점
    [200, 270]                                # [8번 추가] 카메라 시작점
])

GUIDE = [
    "1. SIDE-FRONT FAR (0, 180)",
    "2. SIDE-CENTER-UP FAR (0, 270)",
    "3. SIDE-CENTER EDGE (0, 360)",
    "4. SIDE-REAR FAR (0, 540)",
    "5. VEH-REAR LEFT (200, 540)", 
    "6. L-CORNER BTM (0, 720)",    
    "7. VEH-L BTM (200, 720)",
    "8. !!! CAM ORIGIN (200, 270) !!!"
]

points = []

def on_mouse(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 8:
        points.append([x, y])
        print(f"📍 Point {len(points)}: ({x}, {y})")

def run():
    global points
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    cv2.namedWindow("LEFT SETTING (8 PTS)")
    cv2.setMouseCallback("LEFT SETTING (8 PTS)", on_mouse)
    M = None

    while True:
        ret, frame = cap.read()
        if not ret: break

        undist = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR) if map1 is not None else frame.copy()
        display = undist.copy()
        
        # --- [격자 가이드 복구] ---
        # 세로선 (하늘색)
        for i in range(1, 20):
            cv2.line(display, (i*64, 0), (i*64, 720), (255, 255, 0), 1)
        # 가로선 (하늘색)
        for i in range(1, 20):
            cv2.line(display, (0, i*36), (1280, i*36), (255, 255, 0), 1)
        
        # 중앙 기준선 (분홍색 강조)
        cv2.line(display, (640, 0), (640, 720), (255, 0, 255), 1)
        cv2.line(display, (0, 360), (1280, 360), (255, 0, 255), 2)

        # UI 영역
        cv2.rectangle(display, (0, 0), (850, 60), (0, 0, 0), -1)
        if len(points) < 8:
            cv2.putText(display, f"NEXT: {GUIDE[len(points)]}", (20, 40), 1, 1.5, (0, 255, 255), 2)
        else:
            if M is None: 
                M, _ = cv2.findHomography(np.float32(points), DST_PTS)
            
            topview = cv2.warpPerspective(undist, M, (600, 720))
            cv2.imwrite("temp_left.jpg", topview)
            os.replace("temp_left.jpg", "left_result.jpg")
            cv2.putText(display, "LIVE SENDING (8 PTS)...", (20, 40), 1, 1.5, (0, 255, 0), 2)
        
        # 클릭 점 표시
        for i, p in enumerate(points):
            cv2.circle(display, tuple(p), 7, (0, 0, 255), -1)
            cv2.putText(display, str(i+1), (p[0]+10, p[1]), 1, 1.5, (0, 0, 255), 2)
            
        cv2.imshow("LEFT SETTING (8 PTS)", display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        if key == ord('r'): 
            points, M = [], None
            print("🔄 Points Reset")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run()