import cv2
import numpy as np
import os

# --- [1. 설정 및 상수] ---
LEFT_CAM_ORIGIN = (200, 270)
REAR_CAM_ORIGIN = (300, 540)
CANVAS_SIZE = (600, 720)
CAR_RECT = {"top": 180, "bottom": 540, "left": 200, "right": 400}

# 표시할 주요 좌표 리스트
GUIDE_POINTS = [
    (0, 0), (600, 0), (0, 720), (600, 720),           # 전체 외곽
    (200, 180), (400, 180), (200, 540), (400, 540),   # 차량 모서리
    (200, 0), (400, 0), (0, 540), (600, 540),          # 가이드라인 교차점
    (0, 360)                                          # 좌측 중앙 포인트
]

def get_universal_detector():
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco.DetectorParameters()
    parameters.minMarkerPerimeterRate = 0.02
    return cv2.aruco.ArucoDetector(aruco_dict, parameters)

detector = get_universal_detector()

def create_diagonal_blend_mask():
    """
    세 조각 현상을 방지하기 위해 200x180 영역을 
    대각선 하나로 부드럽게 가르는 마스크를 생성합니다.
    """
    h, w = 180, 200 # (720-540, 200-0)
    mask = np.zeros((h, w), dtype=np.float32)
    
    for y in range(h):
        for x in range(w):
            # 대각선 기준 가중치 계산: 좌측상단(1.0) -> 우측하단(0.0)
            # 수치 조절을 통해 경계선의 각도를 바꿀 수 있습니다.
            val = ( (w - x) + (h - y) ) / (w + h)
            mask[y, x] = np.clip(val, 0, 1)
            
    return cv2.merge([mask, mask, mask]) # BGR 3채널 대응

def process_camera_frame(file_name):
    if not os.path.exists(file_name): return None
    img = cv2.imread(file_name)
    if img is None: return None
    
    # 마커 검출
    corners, ids, _ = detector.detectMarkers(img)
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(img, corners, ids)
    
    # 사이즈 확인 (이미 600x720이어야 하지만 안전하게 한번 더 체크)
    if img.shape[0:2] != (720, 600):
        img = cv2.resize(img, (600, 720))
        
    return img.astype(np.float32) / 255.0

def main():
    blend_mask = create_diagonal_blend_mask()
    
    while True:
        # 최종 화면 캔버스 (실수형으로 계산 후 나중에 uint8 변환)
        canvas_f = np.zeros((CANVAS_SIZE[1], CANVAS_SIZE[0], 3), dtype=np.float32)
        
        l_img_f = process_camera_frame("left_result.jpg")
        r_img_f = process_camera_frame("rear_result.jpg")

        if l_img_f is not None and r_img_f is not None:
            # 1. 겹치지 않는 확실한 구역 배치
            # 상단 절반 (좌측 카메라 담당)
            canvas_f[0:540, 0:600] = l_img_f[0:540, 0:600]
            
            # 하단 우측 (후방 카메라 담당)
            canvas_f[540:720, 200:600] = r_img_f[540:720, 200:600]
            
            # 2. 문제의 '세 조각' 발생 지점 (0~200, 540~720) 처리
            # 이 영역을 대각선 마스크로 한 번에 합쳐서 경계선을 하나로 만듭니다.
            corner_l = l_img_f[540:720, 0:200]
            corner_r = r_img_f[540:720, 0:200]
            
            # 마스크 적용 합성
            stitched = (corner_l * blend_mask) + (corner_r * (1.0 - blend_mask))
            canvas_f[540:720, 0:200] = stitched

        # 8비트 정수형으로 변환
        final_view = (canvas_f * 255).astype(np.uint8)

        # --- [3. UI 레이아웃 표시] ---
        # 가이드라인 (어두운 회색)
        cv2.line(final_view, (200, 0), (200, 720), (80, 80, 80), 1)
        cv2.line(final_view, (400, 0), (400, 720), (80, 80, 80), 1)
        cv2.line(final_view, (0, 540), (600, 540), (80, 80, 80), 1)

        # 중앙 차량 본체 (검정 사각형)
        cv2.rectangle(final_view, (CAR_RECT["left"], CAR_RECT["top"]), 
                      (CAR_RECT["right"], CAR_RECT["bottom"]), (20, 20, 20), -1)
        cv2.putText(final_view, "FRONT", (275, 170), 1, 1.2, (255, 255, 255), 1)

        # 주요 지점 표시 (초록 점)
        for pt in GUIDE_POINTS:
            cv2.circle(final_view, pt, 4, (0, 255, 0), -1)
            cv2.putText(final_view, f"{pt}", (pt[0] + 5, pt[1] - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

        # 카메라 위치 표시 (빨강: 좌측, 주황: 후방)
        cv2.circle(final_view, LEFT_CAM_ORIGIN, 7, (0, 0, 255), -1)
        cv2.circle(final_view, REAR_CAM_ORIGIN, 7, (255, 100, 0), -1)

        cv2.imshow("AVM Final - Diagonal Seamless", final_view)
        
        # 'q' 누르면 종료
        if cv2.waitKey(10) & 0xFF == ord('q'): 
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()