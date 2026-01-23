import cv2
import numpy as np

# ===============================
# 설정
# ===============================
IMAGE_PATH = "capture_.png"   # calib_left.jpg 로 바꿔도 됨
WINDOW_NAME = "POINT PICKER (GRID)"

GRID_X = 64    # 세로 격자 간격
GRID_Y = 36    # 가로 격자 간격

MAX_POINTS = 8  # rear면 8, left면 원하는 수로 조절

points = []

# ===============================
def draw_grid(img):
    h, w = img.shape[:2]

    for x in range(0, w, GRID_X):
        cv2.line(img, (x, 0), (x, h), (80, 80, 80), 1)

    for y in range(0, h, GRID_Y):
        cv2.line(img, (0, y), (w, y), (80, 80, 80), 1)

    # 중심선 강조
    cv2.line(img, (w//2, 0), (w//2, h), (255, 0, 255), 2)
    cv2.line(img, (0, h//2), (w, h//2), (255, 0, 255), 2)


def mouse_cb(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < MAX_POINTS:
        points.append((x, y))
        print(f"📍 Point {len(points)}: ({x}, {y})")


# ===============================
def main():
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        raise RuntimeError("이미지를 불러올 수 없습니다.")

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, mouse_cb)

    while True:
        disp = img.copy()

        # 격자
        draw_grid(disp)

        # 찍은 점 표시
        for i, (x, y) in enumerate(points):
            cv2.circle(disp, (x, y), 6, (0, 0, 255), -1)
            cv2.putText(
                disp, str(i+1),
                (x + 8, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

        # 안내 문구
        cv2.rectangle(disp, (0, 0), (500, 40), (0, 0, 0), -1)
        cv2.putText(
            disp,
            f"Click points in order ({len(points)}/{MAX_POINTS}) | q: quit | r: reset",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.imshow(WINDOW_NAME, disp)

        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            break
        if key == ord('r'):
            points.clear()
            print("🔄 points reset")

    cv2.destroyAllWindows()

    print("\n===== RESULT =====")
    for i, p in enumerate(points):
        print(f"({p[0]}, {p[1]}),")

# ===============================
if __name__ == "__main__":
    main()
