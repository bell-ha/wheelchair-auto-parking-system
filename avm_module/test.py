import cv2
import numpy as np

def create_avm_layout_with_dimensions():
    # 캔버스 및 차량 규격 정의
    canvas_w, canvas_h = 800, 1000
    car_w, car_h = 160, 360

    # 캔버스 생성 (배경은 밝은 회색으로)
    canvas = np.full((canvas_h, canvas_w, 3), 230, dtype=np.uint8)

    # 차량 영역 좌표 계산
    car_x1 = (canvas_w - car_w) // 2
    car_y1 = (canvas_h - car_h) // 2
    car_x2 = car_x1 + car_w
    car_y2 = car_y1 + car_h

    # 중앙 차량 영역을 검은색으로 칠하기
    cv2.rectangle(canvas, (car_x1, car_y1), (car_x2, car_y2), (0, 0, 0), -1)

    # 텍스트 색상 및 폰트
    text_color_dim = (100, 100, 100) # 치수 표시용 회색
    text_color_label = (0, 0, 0)     # 라벨 표시용 검정
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1

    # ==========================================================
    # 1. 전체 캔버스 치수 표시
    # 가로
    cv2.line(canvas, (0, 10), (canvas_w, 10), text_color_dim, 2)
    cv2.arrowedLine(canvas, (canvas_w - 50, 10), (canvas_w - 10, 10), text_color_dim, 2, tipLength=0.2)
    cv2.arrowedLine(canvas, (50, 10), (10, 10), text_color_dim, 2, tipLength=0.2)
    cv2.putText(canvas, f"{canvas_w} px", (canvas_w // 2 - 30, 30), font, font_scale, text_color_dim, thickness)

    # 세로
    cv2.line(canvas, (10, 0), (10, canvas_h), text_color_dim, 2)
    cv2.arrowedLine(canvas, (10, canvas_h - 50), (10, canvas_h - 10), text_color_dim, 2, tipLength=0.2)
    cv2.arrowedLine(canvas, (10, 50), (10, 10), text_color_dim, 2, tipLength=0.2)
    cv2.putText(canvas, f"{canvas_h} px", (20, canvas_h // 2), font, font_scale, text_color_dim, thickness)

    # ==========================================================
    # 2. 중앙 차량 영역 치수 표시
    # 차량 가로
    cv2.line(canvas, (car_x1, car_y1 - 10), (car_x2, car_y1 - 10), text_color_dim, 1)
    cv2.arrowedLine(canvas, (car_x2 - 30, car_y1 - 10), (car_x2 - 10, car_y1 - 10), text_color_dim, 1, tipLength=0.2)
    cv2.arrowedLine(canvas, (car_x1 + 30, car_y1 - 10), (car_x1 + 10, car_y1 - 10), text_color_dim, 1, tipLength=0.2)
    cv2.putText(canvas, f"{car_w} px", (car_x1 + car_w // 2 - 20, car_y1 - 20), font, font_scale, text_color_dim, thickness)

    # 차량 세로
    cv2.line(canvas, (car_x1 - 10, car_y1), (car_x1 - 10, car_y2), text_color_dim, 1)
    cv2.arrowedLine(canvas, (car_x1 - 10, car_y2 - 30), (car_x1 - 10, car_y2 - 10), text_color_dim, 1, tipLength=0.2)
    cv2.arrowedLine(canvas, (car_x1 - 10, car_y1 + 30), (car_x1 - 10, car_y1 + 10), text_color_dim, 1, tipLength=0.2)
    cv2.putText(canvas, f"{car_h} px", (car_x1 - 40, car_y1 + car_h // 2 + 10), font, font_scale, text_color_dim, thickness)

    # ==========================================================
    # 3. 각 카메라 가용 영역 치수 표시 (여백)
    # 상단 여백 (전방)
    cv2.line(canvas, (canvas_w // 2 + 10, 0), (canvas_w // 2 + 10, car_y1), text_color_dim, 1)
    cv2.arrowedLine(canvas, (canvas_w // 2 + 10, car_y1 - 30), (canvas_w // 2 + 10, car_y1 - 10), text_color_dim, 1, tipLength=0.2)
    cv2.arrowedLine(canvas, (canvas_w // 2 + 10, 30), (canvas_w // 2 + 10, 10), text_color_dim, 1, tipLength=0.2)
    cv2.putText(canvas, f"{car_y1} px", (canvas_w // 2 + 20, car_y1 // 2 + 10), font, font_scale, text_color_dim, thickness)

    # 하단 여백 (후방)
    cv2.line(canvas, (canvas_w // 2 + 10, car_y2), (canvas_w // 2 + 10, canvas_h), text_color_dim, 1)
    cv2.arrowedLine(canvas, (canvas_w // 2 + 10, canvas_h - 30), (canvas_w // 2 + 10, canvas_h - 10), text_color_dim, 1, tipLength=0.2)
    cv2.arrowedLine(canvas, (canvas_w // 2 + 10, car_y2 + 30), (canvas_w // 2 + 10, car_y2 + 10), text_color_dim, 1, tipLength=0.2)
    cv2.putText(canvas, f"{canvas_h - car_y2} px", (canvas_w // 2 + 20, car_y2 + (canvas_h - car_y2) // 2 + 10), font, font_scale, text_color_dim, thickness)
    
    # 좌측 여백
    cv2.line(canvas, (0, canvas_h // 2 + 10), (car_x1, canvas_h // 2 + 10), text_color_dim, 1)
    cv2.arrowedLine(canvas, (car_x1 - 30, canvas_h // 2 + 10), (car_x1 - 10, canvas_h // 2 + 10), text_color_dim, 1, tipLength=0.2)
    cv2.arrowedLine(canvas, (30, canvas_h // 2 + 10), (10, canvas_h // 2 + 10), text_color_dim, 1, tipLength=0.2)
    cv2.putText(canvas, f"{car_x1} px", (car_x1 // 2 - 20, canvas_h // 2 + 30), font, font_scale, text_color_dim, thickness)

    # 우측 여백
    cv2.line(canvas, (car_x2, canvas_h // 2 + 10), (canvas_w, canvas_h // 2 + 10), text_color_dim, 1)
    cv2.arrowedLine(canvas, (canvas_w - 30, canvas_h // 2 + 10), (canvas_w - 10, canvas_h // 2 + 10), text_color_dim, 1, tipLength=0.2)
    cv2.arrowedLine(canvas, (car_x2 + 30, canvas_h // 2 + 10), (car_x2 + 10, canvas_h // 2 + 10), text_color_dim, 1, tipLength=0.2)
    cv2.putText(canvas, f"{canvas_w - car_x2} px", (car_x2 + (canvas_w - car_x2) // 2 - 20, canvas_h // 2 + 30), font, font_scale, text_color_dim, thickness)

    # ==========================================================
    # 4. 주요 라벨 표시
    cv2.putText(canvas, "EGO VEHICLE", (car_x1 + 10, car_y1 + car_h // 2 + 5), font, 0.6, (255, 255, 255), 2)
    cv2.putText(canvas, "FRONT VIEW", (canvas_w // 2 - 50, car_y1 - 50), font, 0.7, text_color_label, 2)
    cv2.putText(canvas, "REAR VIEW", (canvas_w // 2 - 40, car_y2 + 70), font, 0.7, text_color_label, 2)
    cv2.putText(canvas, "LEFT SIDE", (car_x1 - 100, canvas_h // 2 + 10), font, 0.7, text_color_label, 2)
    cv2.putText(canvas, "RIGHT SIDE", (car_x2 + 20, canvas_h // 2 + 10), font, 0.7, text_color_label, 2)

    return canvas

if __name__ == "__main__":
    avm_layout = create_avm_layout_with_dimensions()
    cv2.imshow("AVM Layout with Dimensions", avm_layout)
    cv2.waitKey(0)
    cv2.destroyAllWindows()