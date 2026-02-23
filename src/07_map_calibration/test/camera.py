#!/usr/bin/env python3
import cv2
import os
import time
import numpy as np

def main():
    # 필요하면 인덱스만 바꿔줘 (0,1)
    rear_idx, left_idx = 0, 1

    cap_rear = cv2.VideoCapture(rear_idx)
    cap_left = cv2.VideoCapture(left_idx)

    if not cap_rear.isOpened() or not cap_left.isOpened():
        print("[ERR] failed to open cameras. check indices (rear,left) =", rear_idx, left_idx)
        return

    out_dir = "snapshots_rear_left"
    os.makedirs(out_dir, exist_ok=True)
    snap_idx = 0

    # ---- ArUco setup (OpenCV 4.7+ friendly) ----
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    print("[Keys] SPACE: save rear+left | q/ESC: quit")
    print("[Preview] Draw ArUco boxes when detected (rear/left)")

    while True:
        ok_r, fr_r = cap_rear.read()
        ok_l, fr_l = cap_left.read()
        if not ok_r or not ok_l:
            print("[ERR] read failed")
            break

        # ---- detect & draw on preview frames only ----
        prev_r = fr_r.copy()
        prev_l = fr_l.copy()

        # Rear
        gray_r = cv2.cvtColor(prev_r, cv2.COLOR_BGR2GRAY)
        corners_r, ids_r, _ = detector.detectMarkers(gray_r)
        if ids_r is not None and len(ids_r) > 0:
            cv2.aruco.drawDetectedMarkers(prev_r, corners_r, ids_r)

        # Left
        gray_l = cv2.cvtColor(prev_l, cv2.COLOR_BGR2GRAY)
        corners_l, ids_l, _ = detector.detectMarkers(gray_l)
        if ids_l is not None and len(ids_l) > 0:
            cv2.aruco.drawDetectedMarkers(prev_l, corners_l, ids_l)

        # 보기용(저장은 원본 fr_r, fr_l)
        mon_r = cv2.resize(prev_r, (640, 360))
        mon_l = cv2.resize(prev_l, (640, 360))
        cv2.imshow("rear | left", cv2.hconcat([mon_r, mon_l]))

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break

        if key == 32:  # SPACE
            ts = time.strftime("%Y%m%d_%H%M%S")
            ms = int((time.time() % 1.0) * 1000)
            base = f"{ts}_{ms:03d}_{snap_idx:04d}"

            rear_path = os.path.join(out_dir, f"rear_{base}.jpg")
            left_path = os.path.join(out_dir, f"left_{base}.jpg")

            ok1 = cv2.imwrite(rear_path, fr_r)  # 원본 저장
            ok2 = cv2.imwrite(left_path, fr_l)  # 원본 저장

            if ok1 and ok2:
                print(f"[SNAP] {rear_path} | {left_path}")
                snap_idx += 1
            else:
                print("[WARN] cv2.imwrite failed")

    cap_rear.release()
    cap_left.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()