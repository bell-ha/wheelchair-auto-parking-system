import cv2
import numpy as np
import os

def draw_base_layout():
    # 800x1000 캔버스 (검정색 배경)
    canvas = np.zeros((1000, 800, 3), dtype=np.uint8)
    # 차량 본체 (중앙 하단 배치)
    cv2.rectangle(canvas, (320, 320), (480, 680), (40, 40, 40), -1)
    return canvas

def create_panorama_mask():
    """좌측과 후방이 만나는 코너 구역에 부드러운 그라데이션 마스크 생성"""
    mask = np.zeros((1000, 800), dtype=np.float32)
    # 겹치는 핵심 구역: 좌측 하단 (0, 680) ~ (320, 1000)
    # 이 구역을 대각선으로 나누어 투명도 그라데이션을 만듭니다.
    for y in range(680, 1000):
        for x in range(0, 320):
            # 대각선 거리 기반 가중치 계산 (파노라마 스티칭 원리)
            dist = (320 - x) + (y - 680)
            val = np.clip(dist / 640, 0, 1) # 0~1 사이로 정규화
            mask[y, x] = val
    return np.expand_dims(mask, axis=2)

def main():
    print("🌟 REAL-TIME PANORAMA STITCHING START...")
    blend_mask = create_panorama_mask()
    
    while True:
        canvas = draw_base_layout().astype(np.float32) / 255.0
        
        # 1. 파일 로드
        left_exists = os.path.exists("left_result.jpg")
        rear_exists = os.path.exists("rear_result.jpg")
        
        l_img = cv2.imread("left_result.jpg").astype(np.float32)/255.0 if left_exists else None
        r_img = cv2.imread("rear_result.jpg").astype(np.float32)/255.0 if rear_exists else None

        # 2. 파노라마 합성 로직
        if l_img is not None and r_img is not None:
            # 기본 베이스 합성
            result = np.zeros_like(canvas)
            
            # 후방과 겹치지 않는 좌측 상단 영역
            result[0:680, 0:320] = l_img[0:680, 0:320]
            # 좌측과 겹치지 않는 후방 우측 영역
            result[680:1000, 320:800] = r_img[680:1000, 320:800]
            
            # [핵심] 겹치는 코너 구역 (0:320, 680:1000) 스티칭
            # blend_mask를 이용해 두 영상을 부드럽게 섞음
            corner_l = l_img[680:1000, 0:320]
            corner_r = r_img[680:1000, 0:320]
            stitched_corner = corner_l * blend_mask[680:1000, 0:320] + \
                              corner_r * (1 - blend_mask[680:1000, 0:320])
            
            result[680:1000, 0:320] = stitched_corner
            canvas = result
            
        elif l_img is not None: # 좌측만 있을 때
            canvas[0:1000, 0:320] = l_img[0:1000, 0:320]
        elif r_img is not None: # 후방만 있을 때
            canvas[680:1000, 0:800] = r_img[680:1000, 0:800]

        # 3. 차량 이미지 및 가이드라인 마감
        cv2.rectangle(canvas, (320, 320), (480, 680), (0.1, 0.1, 0.1), -1)
        cv2.putText(canvas, "FRONT", (370, 310), 1, 1, (1,1,1), 1)
        
        cv2.imshow("AVM PANORAMA VIEW", (canvas * 255).astype(np.uint8))
        if cv2.waitKey(10) & 0xFF == ord('q'): break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()