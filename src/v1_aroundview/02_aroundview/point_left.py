import cv2
import numpy as np
import os

# ===============================
# 파일 설정
# ===============================
SRC_IMAGE = "data/calib_left.jpg"
OUT_IMAGE = "data/left_result.jpg"

OUT_W, OUT_H = 600, 720

# ===============================
# 🔒 SOURCE POINTS (직접 찍은 좌표, 순서 그대로)
# ===============================
LEFT_SRC_PTS = np.float32([
    (642, 716),  # -> (200,270)
    (822, 375),  # -> (0,270)
    (632, 318),  # -> (0,360)
    (339, 265),  # -> (0,540)
    (272, 249),  # -> (0,630)
    (40,  439),  # -> (200,540)
])

# ===============================
# 🎯 DESTINATION POINTS (의미 좌표)
# ===============================
LEFT_DST_PTS = np.float32([
    (200, 270),
    (0,   270),
    (0,   360),
    (0,   540),
    (0,   630),
    (200, 540),
])

# ===============================
def main():
    if not os.path.exists(SRC_IMAGE):
        raise FileNotFoundError("❌ calib_left.jpg not found")

    src = cv2.imread(SRC_IMAGE)
    if src is None:
        raise RuntimeError("❌ image load failed")

    # ❗ resize 안 함 (좌표 정확도 유지)
    h_src, w_src = src.shape[:2]

    # ===============================
    # 1️⃣ Homography (위치 보정)
    # ===============================
    H, _ = cv2.findHomography(LEFT_SRC_PTS, LEFT_DST_PTS, 0)
    if H is None:
        raise RuntimeError("❌ homography failed")

    warped = cv2.warpPerspective(src, H, (OUT_W, OUT_H))

    # ===============================
    # 2️⃣ LEFT 다각형 마스크
    # ===============================
    mask = np.zeros((OUT_H, OUT_W), dtype=np.uint8)

    left_polygon = np.array([
        (200, 270),
        (0,   270),
        (0,   360),
        (0,   540),
        (0,   630),
        (200, 540),
    ], dtype=np.int32)

    cv2.fillPoly(mask, [left_polygon], 255)

    # ===============================
    # 3️⃣ 마스크 적용
    # ===============================
    left_result = cv2.bitwise_and(warped, warped, mask=mask)

    # ===============================
    # 디버그 시각화 (선택)
    # ===============================
    debug = left_result.copy()
    cv2.polylines(debug, [left_polygon], True, (0, 0, 255), 2)

    cv2.imwrite(OUT_IMAGE, left_result)
    cv2.imshow("LEFT RESULT (POLYGON)", debug)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("✅ left_result.jpg generated successfully")
    print("✅ All points mapped exactly as clicked")

# ===============================
if __name__ == "__main__":
    main()
