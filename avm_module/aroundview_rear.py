import cv2
import numpy as np
import os

data = np.load('data/calib_result_common.npz')
map1, map2 = cv2.fisheye.initUndistortRectifyMap(data['mtx'], data['dist'], np.eye(3), data['new_mtx'], (1280, 720), cv2.CV_16SC2)

DST_PTS = np.float32([[320, 680], [480, 680], [480, 1000], [320, 1000], [0, 680], [800, 680], [0, 1000], [800, 1000]])
GUIDE = [
    "1. L-BUMPER -> (320, 680)", "2. R-BUMPER -> (480, 680)",
    "3. VEH-R BTM -> (480, 1000)", "4. VEH-L BTM -> (320, 1000)",
    "5. L-SIDE EDGE -> (0, 680)", "6. R-SIDE EDGE -> (800, 680)",
    "7. L-CORNER BTM -> (0, 1000)", "8. R-CORNER BTM -> (800, 1000)"
]
points = []

def on_mouse(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 8:
        points.append([x, y])

def run():
    global points
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("REAR SETTING")
    cv2.setMouseCallback("REAR SETTING", on_mouse)
    M = None
    while True:
        ret, frame = cap.read()
        if not ret: break
        undist = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
        display = undist.copy()
        
        # 격자 & 가이드라인
        for i in range(1, 20):
            cv2.line(display, (0, i*36), (1280, i*36), (0, 255, 255), 1)
            cv2.line(display, (i*64, 0), (i*64, 720), (0, 255, 255), 1)
        cv2.line(display, (640, 0), (640, 720), (255, 0, 255), 2)

        # 상단 좌표 자막 바 (검정 배경)
        cv2.rectangle(display, (0, 0), (650, 60), (0, 0, 0), -1)
        if len(points) < 8:
            cv2.putText(display, f"NEXT: {GUIDE[len(points)]}", (20, 40), 1, 1.8, (0, 255, 255), 2)
        else:
            if M is None: M, _ = cv2.findHomography(np.float32(points), DST_PTS)
            topview = cv2.warpPerspective(undist, M, (800, 1000))
            cv2.imwrite("temp_rear.jpg", topview); os.replace("temp_rear.jpg", "rear_result.jpg")
            cv2.putText(display, "LIVE SENDING...", (20, 40), 1, 1.8, (0, 255, 0), 2)

        for i, p in enumerate(points):
            cv2.circle(display, tuple(p), 7, (0, 255, 0), -1)
            cv2.putText(display, str(i+1), (p[0]+10, p[1]), 1, 1.5, (0, 255, 0), 2)

        cv2.imshow("REAR SETTING", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        if key == ord('r'): points, M = [], None
    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__": run()