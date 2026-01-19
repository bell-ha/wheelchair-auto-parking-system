import cv2
import numpy as np
import os

# --- [아르코 마커 설정 최적화] ---
def get_aruco_detector():
    # 이미지상의 마커 딕셔너리 (메인 코드와 동일하게 설정)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco.DetectorParameters()
    
    # 인식률 향상을 위한 파라미터 조절
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
    print("⚠️ 왜곡 보정 파일을 찾을 수 없습니다. 기본 맵핑을 시도합니다.")
    map1, map2 = None, None

# 2. 타겟 좌표 (600x720 규격)
DST_PTS = np.float32([
    [200, 540], [400, 540],
    [0, 720],   [200, 720],
    [400, 720], [600, 720]
])

GUIDE = [
    "1. L-BUMPER -> (200, 540)", 
    "2. R-BUMPER -> (400, 540)",
    "3. L-CORNER BTM -> (0, 720)", 
    "4. VEH-L BTM -> (200, 720)",
    "5. VEH-R BTM -> (400, 720)", 
    "6. R-CORNER BTM -> (600, 720)"
]

points = []

def on_mouse(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 6:
        points.append([x, y])
        print(f"📍 Point {len(points)}: ({x}, {y}) 클릭됨")

def run():
    global points
    cap = cv2.VideoCapture(1) # 카메라 번호 확인
    cv2.namedWindow("REAR SETTING")
    cv2.setMouseCallback("REAR SETTING", on_mouse)
    M = None

    while True:
        ret, frame = cap.read()
        if not ret: break

        # 왜곡 보정 적용
        undist = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR) if map1 is not None else frame.copy()
        display = undist.copy()
        
        # --- [아르코 마커 실시간 감지 추가] ---
        # 원본(undist) 이미지에서 마커를 찾아 display에 그립니다.
        corners, ids, rejected = detector.detectMarkers(display)
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(display, corners, ids)
        
        # 격자 가이드선
        for i in range(1, 20):
            cv2.line(display, (0, i*36), (1280, i*36), (50, 50, 50), 1)
            cv2.line(display, (i*64, 0), (i*64, 720), (50, 50, 50), 1)
        cv2.line(display, (640, 0), (640, 720), (255, 0, 255), 1)

        # 상단 안내 바
        cv2.rectangle(display, (0, 0), (700, 60), (0, 0, 0), -1)
        if len(points) < 6:
            cv2.putText(display, f"NEXT: {GUIDE[len(points)]}", (20, 40), 1, 1.8, (0, 255, 255), 2)
        else:
            if M is None:
                M, _ = cv2.findHomography(np.float32(points), DST_PTS)
            
            # 탑뷰 변환
            topview = cv2.warpPerspective(undist, M, (600, 720))
            
            # 마커가 포함된 결과물을 저장하고 싶다면 아래 주석을 해제하세요.
            # 하지만 보통 캘리브레이션 결과물은 깨끗한 영상을 선호하므로 
            # 여기서는 마커가 그려지지 않은 'topview'를 저장합니다.
            cv2.imwrite("temp_rear.jpg", topview)
            os.replace("temp_rear.jpg", "rear_result.jpg")
            
            cv2.putText(display, "LIVE SENDING...", (20, 40), 1, 1.8, (0, 255, 0), 2)

        # 클릭한 점 표시
        for i, p in enumerate(points):
            cv2.circle(display, tuple(p), 7, (0, 255, 0), -1)
            cv2.putText(display, str(i+1), (p[0]+10, p[1]), 1, 1.5, (0, 255, 0), 2)

        cv2.imshow("REAR SETTING", display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        if key == ord('r'):
            points, M = [], None
            print("🔄 Points Reset")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run()