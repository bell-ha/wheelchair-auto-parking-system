import cv2
import numpy as np
import os

# --- [아르코 마커 설정 최적화] ---
def get_aruco_detector():
    # 사용하시는 마커가 6x6인 경우 DICT_6X6_250을 사용합니다.
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco.DetectorParameters()
    
    # 사이드미러 특성상 멀리 있는 마커나 왜곡된 마커를 잡기 위해 파라미터 완화
    parameters.adaptiveThreshWinSizeMin = 3
    parameters.adaptiveThreshWinSizeMax = 23
    parameters.minMarkerPerimeterRate = 0.02
    
    return cv2.aruco.ArucoDetector(aruco_dict, parameters)

detector = get_aruco_detector()

# 1. 왜곡 보정 데이터 로드
try:
    data = np.load('data/calib_left.npz')
    # 캘리브레이션 시 사용한 해상도 (1280, 720)와 동일하게 맵 생성
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        data['mtx'], data['dist'], np.eye(3), data['new_mtx'], (1280, 720), cv2.CV_16SC2
    )
except:
    print("⚠️ 왜곡 보정 데이터가 없습니다. calib_result.npz 파일을 확인하세요.")
    map1, map2 = None, None

# 2. 정정된 6개 타겟 좌표 (600x720 규격)
DST_PTS = np.float32([
    [0, 180], [0, 270], [0, 540],   # 왼쪽 외곽 라인 3점
    [200, 540],                     # 차량 왼쪽 뒤 모서리
    [0, 720], [200, 720]            # 후방과 겹치는 바닥 구역 2점
])

GUIDE = [
    "1. SIDE-FRONT FAR (0, 180)",
    "2. SIDE-CENTER FAR (0, 270)",
    "3. SIDE-REAR FAR (0, 540)",
    "4. VEH-REAR LEFT (200, 540)", 
    "5. L-CORNER BTM (0, 720)",    
    "6. VEH-L BTM (200, 720)"      
]

points = []

def on_mouse(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 6:
        points.append([x, y])
        print(f"📍 Point {len(points)}: ({x}, {y})")

def run():
    global points
    cap = cv2.VideoCapture(0) # 좌측 카메라 인덱스 확인 필요
    
    # --- [중요: 해상도 강제 설정] ---
    # 캘리브레이션 데이터와 일치하도록 1280x720으로 고정합니다.
    # 이 설정이 없으면 640x480으로 열려 이미지가 찌그러질 수 있습니다.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    cv2.namedWindow("LEFT SETTING")
    cv2.setMouseCallback("LEFT SETTING", on_mouse)
    M = None

    while True:
        ret, frame = cap.read()
        if not ret: break

        # 왜곡 보정 적용 (1280x720 프레임에 최적화된 맵핑)
        undist = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR) if map1 is not None else frame.copy()
        display = undist.copy()
        
        # --- [아르코 마커 실시간 감지] ---
        # 보정된 이미지 위에서 마커를 찾아 시각화합니다.
        corners, ids, rejected = detector.detectMarkers(display)
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(display, corners, ids)
        
        # --- [시각 가이드 (격자)] ---
        for i in range(1, 20):
            cv2.line(display, (0, i*36), (1280, i*36), (255, 255, 0), 1)
            cv2.line(display, (i*64, 0), (i*64, 720), (255, 255, 0), 1)
        
        cv2.line(display, (640, 0), (640, 720), (255, 0, 255), 1)
        cv2.line(display, (0, 360), (1280, 360), (255, 0, 255), 2)

        # 상단 UI 바
        cv2.rectangle(display, (0, 0), (700, 60), (0, 0, 0), -1)
        if len(points) < 6:
            cv2.putText(display, f"NEXT: {GUIDE[len(points)]}", (20, 40), 1, 1.5, (0, 255, 255), 2)
        else:
            if M is None: 
                M, _ = cv2.findHomography(np.float32(points), DST_PTS)
            
            # 탑뷰 변환 (600x720 규격)
            topview = cv2.warpPerspective(undist, M, (600, 720))
            
            # 메인 시스템으로 전달할 이미지 저장
            cv2.imwrite("temp_left.jpg", topview)
            os.replace("temp_left.jpg", "left_result.jpg")
            cv2.putText(display, "LIVE SENDING...", (20, 40), 1, 1.5, (0, 255, 0), 2)
        
        # 클릭한 점 표시
        for i, p in enumerate(points):
            cv2.circle(display, tuple(p), 7, (0, 0, 255), -1)
            cv2.putText(display, str(i+1), (p[0]+10, p[1]), 1, 1.5, (0, 0, 255), 2)
            
        cv2.imshow("LEFT SETTING", display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        if key == ord('r'): # 'r' 키를 눌러 점 초기화
            points, M = [], None
            print("🔄 Points Reset")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run()