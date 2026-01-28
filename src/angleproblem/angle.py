import cv2
import numpy as np
import math

def wrap_angle(deg):
    while deg > 180:
        deg -= 360
    while deg < -180:
        deg += 360
    return deg


class FinalOptimizedTracker:
    def __init__(self):
        self.marker_size = 25.0

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_5X5_250
        )
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

        # === 핵심 상태 변수 ===
        self.prev_raw_yaw = None
        self.virtual_heading = 0.0   # ★ 우리가 쓸 heading
        self.marker_pos = None

    def get_dummy_camera_matrix(self, w, h):
        f = w * 1.0
        return (
            np.array([[f, 0, w/2],
                      [0, f, h/2],
                      [0, 0, 1]], dtype=np.float32),
            np.zeros((4,1))
        )

    def yaw_from_rvec(self, rvec):
        R, _ = cv2.Rodrigues(rvec)
        yaw = math.atan2(R[1,0], R[0,0])
        return math.degrees(yaw)

    def process(self, frame):
        if frame is None:
            return frame

        h, w = frame.shape[:2]
        cam_mtx, dist = self.get_dummy_camera_matrix(w, h)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is not None:
            c = corners[0]
            obj = np.array([
                [-12.5,  12.5, 0],
                [ 12.5,  12.5, 0],
                [ 12.5, -12.5, 0],
                [-12.5, -12.5, 0]
            ], dtype=np.float32)

            ret, rvec, tvec = cv2.solvePnP(
                obj, c, cam_mtx, dist,
                flags=cv2.SOLVEPNP_IPPE_SQUARE
            )

            if ret:
                raw_yaw = self.yaw_from_rvec(rvec)

                if self.prev_raw_yaw is not None:
                    delta = wrap_angle(raw_yaw - self.prev_raw_yaw)
                    self.virtual_heading += delta
                    self.virtual_heading = wrap_angle(self.virtual_heading)

                self.prev_raw_yaw = raw_yaw
                self.marker_pos = tvec.flatten()

                cv2.drawFrameAxes(frame, cam_mtx, dist, rvec, tvec, 20)

                cv2.putText(frame,
                    f"Virtual heading: {self.virtual_heading:+.1f} deg",
                    (10,40), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0,255,0), 2)

        return frame

    def reset_heading(self):
        self.virtual_heading = 0.0
        print("🔁 Heading reset to 0°")


# ==========================
# 실행부
# ==========================
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    tracker = FinalOptimizedTracker()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = tracker.process(frame)
        cv2.imshow("Tracking", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            tracker.reset_heading()

    cap.release()
    cv2.destroyAllWindows()
