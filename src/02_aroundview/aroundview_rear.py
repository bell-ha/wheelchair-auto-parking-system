import cv2
import numpy as np
import os

# --- [아르코 마커 설정 최적화] ---
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
    data = np.load('data/calib_rear.npz')
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        data['mtx'], data['dist'], np.eye(3), data['new_mtx'], (1280, 720), cv2.CV_16SC2
    )
except:
    print("⚠️ 왜곡 보정 파일을 찾을 수 없습니다.")
    map1, map2 = None, None

# --- [2. 타겟 좌표 수정: 300라인 중심점 2개 추가] ---
DST_PTS = np.float32([
    [200, 540], [400, 540],                   # 1, 2. 범퍼 좌우 모서리
    [0, 720],   [200, 720],                   # 3, 4. 바닥 왼쪽 구간
    [400, 720], [600, 720],                   # 5, 6. 바닥 오른쪽 구간
    [300, 540],                               # 7. [추가] 후방 범퍼 정중앙
    [300, 720]                                # 8. [추가] 바닥 최하단 정중앙
])

GUIDE = [
    "1. L-BUMPER (200, 540)", 
    "2. R-BUMPER (400, 540)",
    "3. L-CORNER BTM (0, 720)", 
    "4. VEH-L BTM (200, 720)",
    "5. VEH-R BTM (400, 720)", 
    "6. R-CORNER BTM (600, 720)",
    "7. !!! REAR BUMPER CENTER (300, 540) !!!",
    "8. !!! GROUND BOTTOM CENTER (300, 720) !!!"
]

points = []

def on_mouse(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 8:
        points.append([x, y])
        print(f"📍 Point {len(points)}: ({x}, {y}) 클릭됨")

def run():
    global points
    cap = cv2.VideoCapture(1) # 카메라 번호 확인
    cv2.namedWindow("REAR SETTING (8 PTS)")
    cv2.setMouseCallback("REAR SETTING (8 PTS)", on_mouse)
    M = None

    while True:
        ret, frame = cap.read()
        if not ret: break

        undist = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR) if map1 is not None else frame.copy()
        display = undist.copy()
        
        # 아르코 마커 실시간 감지
        corners, ids, rejected = detector.detectMarkers(display)
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(display, corners, ids)
        
        # --- [격자 가이드 복구] ---
        for i in range(1, 20):
            cv2.line(display, (i*64, 0), (i*64, 720), (255, 255, 0), 1) # 세로 (하늘색)
            cv2.line(display, (0, i*36), (1280, i*36), (255, 255, 0), 1) # 가로 (하늘색)
        
        # 중앙 기준축 (분홍색)
        cv2.line(display, (640, 0), (640, 720), (255, 0, 255), 2)

        # 상단 안내 바
        cv2.rectangle(display, (0, 0), (850, 60), (0, 0, 0), -1)
        if len(points) < 8:
            cv2.putText(display, f"NEXT: {GUIDE[len(points)]}", (20, 40), 1, 1.5, (0, 255, 255), 2)
        else:
            if M is None:
                # 8개 점을 사용하여 호모그래피 행렬 계산
                M, _ = cv2.findHomography(np.float32(points), DST_PTS)
            
            topview = cv2.warpPerspective(undist, M, (600, 720))
            cv2.imwrite("temp_rear.jpg", topview)
            os.replace("temp_rear.jpg", "rear_result.jpg")
            cv2.putText(display, "LIVE SENDING (8 PTS)...", (20, 40), 1, 1.5, (0, 255, 0), 2)

        # 클릭한 점 표시
        for i, p in enumerate(points):
            cv2.circle(display, tuple(p), 7, (0, 0, 255), -1)
            cv2.putText(display, str(i+1), (p[0]+10, p[1]), 1, 1.5, (0, 0, 255), 2)

        cv2.imshow("REAR SETTING (8 PTS)", display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        if key == ord('r'):
            points, M = [], None
            print("🔄 Points Reset")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run()