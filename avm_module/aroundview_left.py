import cv2
import numpy as np
import os

data = np.load('data/calib_result.npz')
map1, map2 = cv2.fisheye.initUndistortRectifyMap(data['mtx'], data['dist'], np.eye(3), data['new_mtx'], (1280, 720), cv2.CV_16SC2)

DST_PTS = np.float32([[0, 0], [320, 320], [0, 680], [320, 680], [0, 1000], [320, 1000]])
GUIDE = [
    "1. FRONT FAR -> (0, 0)", "2. V-FRONT -> (320, 320)",
    "3. REAR SIDE -> (0, 680)", "4. V-REAR -> (320, 680)",
    "5. REAR FAR -> (0, 1000)", "6. V-BTM FAR -> (320, 1000)"
]
points = []

def on_mouse(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 6:
        points.append([x, y])

def run():
    global points
    cap = cv2.VideoCapture(1)
    cv2.namedWindow("LEFT SETTING")
    cv2.setMouseCallback("LEFT SETTING", on_mouse)
    M = None
    while True:
        ret, frame = cap.read()
        if not ret: break
        undist = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
        display = undist.copy()
        
        # 격자 (하늘색)
        for i in range(1, 20):
            cv2.line(display, (0, i*36), (1280, i*36), (255, 255, 0), 1)
            cv2.line(display, (i*64, 0), (i*64, 720), (255, 255, 0), 1)
        
        # ★ 강조 가이드라인 (중앙 가로선 분홍색 두껍게)
        cv2.line(display, (0, 360), (1280, 360), (255, 0, 255), 3) 
        cv2.line(display, (640, 0), (640, 720), (255, 0, 255), 1)

        cv2.rectangle(display, (0, 0), (650, 60), (0, 0, 0), -1)
        if len(points) < 6:
            cv2.putText(display, f"NEXT: {GUIDE[len(points)]}", (20, 40), 1, 1.8, (255, 255, 0), 2)
        else:
            if M is None: M, _ = cv2.findHomography(np.float32(points), DST_PTS)
            topview = cv2.warpPerspective(undist, M, (800, 1000))
            cv2.imwrite("temp_left.jpg", topview); os.replace("temp_left.jpg", "left_result.jpg")
            cv2.putText(display, "LIVE SENDING...", (20, 40), 1, 1.8, (0, 255, 0), 2)
        
        for i, p in enumerate(points):
            cv2.circle(display, tuple(p), 7, (255, 0, 0), -1)
            cv2.putText(display, str(i+1), (p[0]+10, p[1]), 1, 1.5, (255, 0, 0), 2)
            
        cv2.imshow("LEFT SETTING", display)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        if cv2.waitKey(1) & 0xFF == ord('r'): points, M = [], None
    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__": run()