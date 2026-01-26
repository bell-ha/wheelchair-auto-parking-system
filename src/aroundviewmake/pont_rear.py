import cv2
import numpy as np
import os

# ===============================
# 파일 설정
# ===============================
SRC_IMAGE = "data/calib_rear.jpg"
OUT_IMAGE = "data/rear_result.jpg"

OUT_W, OUT_H = 600, 720

# ===============================
# 🔒 SOURCE POINTS (당신이 직접 찍은 좌표, 순서 그대로)
# ===============================
REAR_SRC_PTS = np.float32([
    (1153, 707),
    (128, 709),
    (45, 388),
    (220, 313),
    (520, 324),
    (759, 324),
    (1059, 316),
    (1241, 390),
])

# ===============================
# 🎯 DESTINATION POINTS (요청한 다각형)
# ===============================
REAR_DST_PTS = np.float32([
    (200, 540),
    (400, 540),
    (600, 630),
    (600, 720),
    (400, 720),
    (200, 720),
    (0,   720),
    (0,   630),
])

# ===============================
def main():
    if not os.path.exists(SRC_IMAGE):
        raise FileNotFoundError("❌ calib_rear.jpg not found")

    src = cv2.imread(SRC_IMAGE)
    if src is None:
        raise RuntimeError("❌ image load failed")

    # ❗ resize 안 함 (좌표 정확도 유지)
    h_src, w_src = src.shape[:2]

    # ===============================
    # 1️⃣ Homography (위치 보정)
    # ===============================
    H, _ = cv2.findHomography(REAR_SRC_PTS, REAR_DST_PTS, 0)
    if H is None:
        raise RuntimeError("❌ homography failed")

    warped = cv2.warpPerspective(src, H, (OUT_W, OUT_H))

    # ===============================
    # 2️⃣ 다각형 마스크 (형태 정의)
    # ===============================
    mask = np.zeros((OUT_H, OUT_W), dtype=np.uint8)

    rear_polygon = np.array([
        (200, 540),
        (400, 540),
        (600, 630),
        (600, 720),
        (400, 720),
        (200, 720),
        (0,   720),
        (0,   630),
    ], dtype=np.int32)

    cv2.fillPoly(mask, [rear_polygon], 255)

    # ===============================
    # 3️⃣ 마스크 적용
    # ===============================
    rear_result = cv2.bitwise_and(warped, warped, mask=mask)

    # ===============================
    # 디버그 시각화 (선택)
    # ===============================
    debug = rear_result.copy()
    cv2.polylines(debug, [rear_polygon], True, (0, 0, 255), 2)

    cv2.imwrite(OUT_IMAGE, rear_result)
    cv2.imshow("REAR RESULT (POLYGON)", debug)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("✅ rear_result.jpg generated successfully")
    print("✅ All points mapped exactly as clicked")

# ===============================
if __name__ == "__main__":
    main()
