import cv2
import numpy as np
import os

# --- [1. 모든 아르코 마커 대응 검출기 설정] ---
def get_universal_detector():
    # 이미지의 마커는 6x6으로 보입니다. 범용성을 위해 6x6_250을 기본으로 설정합니다.
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco.DetectorParameters()
    
    # 왜곡되거나 작은 마커를 더 잘 잡기 위한 파라미터 튜닝
    parameters.adaptiveThreshWinSizeMin = 3
    parameters.adaptiveThreshWinSizeMax = 23
    parameters.minMarkerPerimeterRate = 0.02
    
    return cv2.aruco.ArucoDetector(aruco_dict, parameters)

detector = get_universal_detector()

def draw_base_layout():
    canvas = np.zeros((720, 600, 3), dtype=np.uint8)
    cv2.rectangle(canvas, (200, 180), (400, 540), (40, 40, 40), -1)
    return canvas

def create_panorama_mask():
    mask = np.zeros((720, 600), dtype=np.float32)
    for y in range(540, 720):
        for x in range(0, 200):
            dist = (200 - x) + (y - 540)
            val = np.clip(dist / 380, 0, 1) 
            mask[y, x] = val
    return np.expand_dims(mask, axis=2)

# --- [2. 개별 이미지에서 마커를 찾고 처리하는 함수] ---
def process_camera_frame(file_name):
    if not os.path.exists(file_name):
        return None
    
    img = cv2.imread(file_name)
    if img is None: return None

    # 중요: uint8(BGR) 상태에서 마커 검출 수행
    corners, ids, rejected = detector.detectMarkers(img)
    if ids is not None:
        # 마커 테두리 및 ID 화면에 그리기
        cv2.aruco.drawDetectedMarkers(img, corners, ids)
    
    # 합성 규격에 맞게 리사이즈 후 float 변환
    img_res = cv2.resize(img, (600, 720))
    return img_res.astype(np.float32) / 255.0

def main():
    print("🌟 AVM SYSTEM: Marker Detection & Synthesis START")
    blend_mask = create_panorama_mask()
    
    while True:
        # 기본 캔버스 준비
        base_canvas = draw_base_layout()
        
        # 각 카메라 이미지 처리 (마커 감지 포함)
        l_img_f = process_camera_frame("left_result.jpg")
        r_img_f = process_camera_frame("rear_result.jpg")

        canvas_f = base_canvas.astype(np.float32) / 255.0
        
        # 합성 로직
        if l_img_f is not None and r_img_f is not None:
            result = np.zeros_like(canvas_f)
            # 좌측 영역 배치
            result[0:540, 0:200] = l_img_f[0:540, 0:200]
            # 후방 영역 배치
            result[540:720, 200:600] = r_img_f[540:720, 200:600]
            
            # 코너 블렌딩 (스티칭)
            corner_l = l_img_f[540:720, 0:200]
            corner_r = r_img_f[540:720, 0:200]
            stitched = corner_l * blend_mask[540:720, 0:200] + \
                       corner_r * (1.0 - blend_mask[540:720, 0:200])
            result[540:720, 0:200] = stitched
            canvas_f = result
            
        elif l_img_f is not None:
            canvas_f[0:720, 0:200] = l_img_f[0:720, 0:200]
        elif r_img_f is not None:
            canvas_f[540:720, 0:600] = r_img_f[540:720, 0:600]

        # 최종 출력을 위해 다시 uint8로 변환
        final_view = (canvas_f * 255).astype(np.uint8)

        # --- [3. UI 오버레이 및 좌표 표시] ---
        # 차량 내부 사각형
        cv2.rectangle(final_view, (200, 180), (400, 540), (25, 25, 25), -1)
        cv2.putText(final_view, "FRONT", (275, 170), 1, 1.2, (255, 255, 255), 1)

        # 기존 초록색 좌표 점들 표시
        pts = [
            (0, 0), (600, 0), (0, 720), (600, 720),
            (200, 180), (400, 180), (200, 540), (400, 540),
            (200, 0), (400, 0), (0, 540), (600, 540)
        ]
        for pt in pts:
            cv2.circle(final_view, pt, 4, (0, 255, 0), -1)
            cv2.putText(final_view, f"{pt}", (pt[0] + 5, pt[1] - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

        # 가이드라인 표시
        cv2.line(final_view, (200, 0), (200, 720), (100, 100, 100), 1)
        cv2.line(final_view, (400, 0), (400, 720), (100, 100, 100), 1)
        cv2.line(final_view, (0, 540), (600, 540), (100, 100, 100), 1)

        cv2.imshow("AVM Universal Monitor", final_view)
        if cv2.waitKey(10) & 0xFF == ord('q'): break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()