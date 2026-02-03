# main.py
import cv2
import numpy as np
import math

from planner import PathPlanner
from tracker import PathTracker
from localization import PoseEstimator
from visualizer import Visualizer


K = np.array([[601.71923257, 0.0, 630.47700714],
              [0.0, 601.34529853, 367.21223657],
              [0.0, 0.0, 1.0]], dtype=np.float32)

D = np.array([-0.18495647, 0.02541005, -0.01068433, 0.00321714], dtype=np.float32)


class CompactTracker:
    def __init__(self):
        # =========================
        # MAP / ROI
        # =========================
        self.map_w, self.map_h = 1200, 1200

        # 중앙 ROI 600x720
        self.grid_w, self.grid_h = 600, 720
        self.off_x = (self.map_w - self.grid_w) // 2   # 300
        self.off_y = (self.map_h - self.grid_h) // 2   # 240

        self.map_scale = 0.5
        self.wc_w, self.wc_l = 57.0, 100.0

        # logging
        self.logging_active = False
        self.angle_logs = []

        # marker
        self.marker_size = 25.0  # cm

        # =========================
        # CAR ZONE (ROI 기준: (200,180)~(400,540))
        # =========================
        self.car_dim = [200, 360]
        self.car_x = self.off_x + 200
        self.car_y = self.off_y + 180

        # ROI center
        car_cx = self.off_x + self.grid_w / 2
        car_cy = self.off_y + self.grid_h / 2

        # =========================
        # CAM POS (ROI -> 전체)
        # =========================
        left_cam_px = np.array([self.off_x + 200, self.off_y + 270], dtype=np.float32)
        rear_cam_px = np.array([self.off_x + 300, self.off_y + 540], dtype=np.float32)

        # =========================
        # PARAMS
        # =========================
        self.alpha = 0.25
        self.dist_gain = 1.88  # (188 => 1.88)

        # 이전처럼 기본 트림 넣어둠(원하면 0으로 시작해도 됨)
        rear_yaw_trim = 3.0
        left_yaw_trim = 8.0

        self.cams = {
            'cam0': {  # Rear
                'name': 'REAR',
                'pos_px': rear_cam_px,
                'h_cm': 105.0,
                'map_angle_deg': 90.0,
                'yaw_trim_deg': float(rear_yaw_trim),
                'sens': 1.6,
                'install_angle': 0.0,
                'install_offset': 0.0,
                'map_scale': self.map_scale,
                'base_weight': 1.00,
                'color': (255, 0, 0)
            },
            'cam1': {  # Left
                'name': 'LEFT',
                'pos_px': left_cam_px,
                'h_cm': 110.0,
                'map_angle_deg': 157.0,
                'yaw_trim_deg': float(left_yaw_trim),
                'sens': 1.6,
                'install_angle': 113.0,
                'install_offset': 50.84,
                'map_scale': self.map_scale,
                'base_weight': 0.85,
                'color': (0, 0, 255)
            }
        }

        # =========================
        # GOALS (ROI -> 전체)
        # =========================
        # 주차 목표: (300,600)에서 북쪽(=0) 바라보기
        # visualizer 표현에서는 heading=-90deg가 "북쪽"처럼 보이므로 그대로 -90 유지
        park_goal = (self.off_x + 300, self.off_y + 600, -90)

        # 출차 목표(최근 네 설명 기준): (150,260)
        exit_goal_left = (self.off_x + 150, self.off_y + 260, None)
        exit_goal_right = (self.off_x + 450, self.off_y + 260, None)

        self.parking_mode = True
        self.goals = [[park_goal]]

        mid_y = self.off_y + 400
        self.exit_goals = [
            [park_goal],
            [(car_cx - 230, mid_y, None), (car_cx + 230, mid_y, None)],
            [exit_goal_left, exit_goal_right],
        ]

        self.stage, self.goal_idx = 0, 0
        self.exit_choice = 0
        self.goal_selected = False

        # obstacles
        self.dynamic_obstacles = []

        # =========================
        # MODULES
        # =========================
        self.planner = PathPlanner(self.map_w, self.map_h, self.wc_w, self.map_scale)
        self.planner.set_obstacle_checker(self.is_obstacle)

        self.tracker = PathTracker(self.wc_l, self.map_scale)
        self.tracker.set_planner(self.planner)
        self.tracker.set_obstacle_checker(self.is_obstacle)

        self.estimator = PoseEstimator(
            K, D, self.cams, self.marker_size, self.alpha,
            wc_l_cm=self.wc_l,
            marker_to_center_cm=50.0
        )

        self.visualizer = Visualizer(
            self.map_w, self.map_h,
            self.car_dim, (self.car_x, self.car_y),
            self.wc_w, self.wc_l, self.map_scale
        )

        # =========================
        # VIDEO
        # =========================
        self.cap0 = cv2.VideoCapture('test_rear.mp4')
        self.cap1 = cv2.VideoCapture('test_left.mp4')
        self.total_frames = int(min(self.cap0.get(cv2.CAP_PROP_FRAME_COUNT),
                                    self.cap1.get(cv2.CAP_PROP_FRAME_COUNT)))

        self.win_name = "Compact Tracker (cam0/cam1 + FUSED)"
        cv2.namedWindow(self.win_name)
        cv2.setMouseCallback(self.win_name, self.mouse_callback)

        # =========================
        # TRACKBARS
        # =========================
        INIT_FRAME = 278
        INIT_FRAME = int(max(0, min(INIT_FRAME, max(0, self.total_frames - 1))))

        cv2.createTrackbar("Frame", self.win_name, INIT_FRAME, max(0, self.total_frames - 1), self.on_frame)
        cv2.createTrackbar("Mode", self.win_name, 1, 1, self.on_mode)
        cv2.createTrackbar("ExitDir", self.win_name, 0, 1, self.on_exit)

        cv2.createTrackbar("Smooth(%)", self.win_name, int(self.alpha * 100), 100, self.on_alpha)
        cv2.createTrackbar("DistGain(%)", self.win_name, int(self.dist_gain * 100), 300, self.on_dist_gain)

        cv2.createTrackbar("Rear_MapYaw", self.win_name, 90 + int(self.cams["cam0"]["yaw_trim_deg"]), 180,
                           lambda v: self.set_cam("cam0", "yaw_trim_deg", v - 90))
        cv2.createTrackbar("Left_MapYaw", self.win_name, 90 + int(self.cams["cam1"]["yaw_trim_deg"]), 180,
                           lambda v: self.set_cam("cam1", "yaw_trim_deg", v - 90))

        cv2.createTrackbar("Rear_Sens(x10)", self.win_name, int(self.cams["cam0"]["sens"] * 10), 30,
                           lambda v: self.set_cam("cam0", "sens", v / 10.0))
        cv2.createTrackbar("Left_Sens(x10)", self.win_name, int(self.cams["cam1"]["sens"] * 10), 30,
                           lambda v: self.set_cam("cam1", "sens", v / 10.0))

        cv2.createTrackbar("Rear_InstAngle", self.win_name, int(self.cams["cam0"]["install_angle"]), 180,
                           lambda v: self.set_cam("cam0", "install_angle", float(v)))
        cv2.createTrackbar("Left_InstAngle", self.win_name, int(self.cams["cam1"]["install_angle"]), 180,
                           lambda v: self.set_cam("cam1", "install_angle", float(v)))

        cv2.createTrackbar("Rear_InstOffset(x10)", self.win_name, int(self.cams["cam0"]["install_offset"] * 10), 1800,
                           lambda v: self.set_cam("cam0", "install_offset", v / 10.0))
        cv2.createTrackbar("Left_InstOffset(x10)", self.win_name, int(self.cams["cam1"]["install_offset"] * 10), 1800,
                           lambda v: self.set_cam("cam1", "install_offset", v / 10.0))

        # --- 안정화 핵심 ---
        cv2.createTrackbar("MinArea", self.win_name, 450, 3000,
                           lambda v: self.estimator.set_quality_gates(min_area_px2=max(50.0, float(v))))
        cv2.createTrackbar("MaxErr(x0.1)", self.win_name, 38, 150,
                           lambda v: self.estimator.set_quality_gates(reproj_err_th=max(0.5, float(v) / 10.0)))
        cv2.createTrackbar("EdgeGate(px)", self.win_name, 25, 120,
                           lambda v: self.estimator.set_quality_gates(min_edge_px=max(0.0, float(v))))
        cv2.createTrackbar("M2C(cm x10)", self.win_name, 500, 1500,
                           lambda v: self.estimator.set_marker_to_center(float(v) / 10.0))
        cv2.createTrackbar("HeadFilter", self.win_name, 1, 1,
                           lambda v: self.estimator.set_heading_filter(use_kalman=(v == 1)))

        cv2.createTrackbar("LeftBase(%)", self.win_name, int(self.cams["cam1"]["base_weight"] * 100), 100,
                           lambda v: self.set_cam("cam1", "base_weight", max(0.05, v / 100.0)))

        self.on_frame(INIT_FRAME)

    # ---------------- UI callbacks ----------------
    def set_cam(self, cam_key, key, val):
        self.cams[cam_key][key] = float(val)

    def on_frame(self, v):
        if self.total_frames <= 0:
            self.f0 = None
            self.f1 = None
            return
        self.cap0.set(cv2.CAP_PROP_POS_FRAMES, int(v))
        self.cap1.set(cv2.CAP_PROP_POS_FRAMES, int(v))
        ret0, self.f0 = self.cap0.read()
        ret1, self.f1 = self.cap1.read()
        if not ret0:
            self.f0 = None
        if not ret1:
            self.f1 = None

    def on_alpha(self, v):
        self.alpha = max(0.01, v / 100.0)
        self.estimator.set_alpha(self.alpha)

    def on_dist_gain(self, v):
        self.dist_gain = max(0.01, v / 100.0)

    def on_mode(self, v):
        self.parking_mode = (v == 1)
        self.stage, self.goal_idx, self.goal_selected = 0, 0, False
        self.tracker.clear_path()

    def on_exit(self, v):
        self.exit_choice = v
        if not self.parking_mode and self.stage == 1:
            final = self.exit_goals[2][v][0:2]
            dists = [math.dist(final, g[0:2]) for g in self.exit_goals[1]]
            self.goal_idx = dists.index(min(dists))

    # ---------------- mouse / obstacles ----------------
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dynamic_obstacles.append((x, y, 30))
            print(f"➕ 장애물 추가: ({x}, {y})")
            if self.estimator.is_initialized:
                self.update_path()
        elif event == cv2.EVENT_RBUTTONDOWN:
            for i, (ox, oy, r) in enumerate(self.dynamic_obstacles):
                if math.sqrt((ox - x) ** 2 + (oy - y) ** 2) < r:
                    self.dynamic_obstacles.pop(i)
                    print(f"➖ 장애물 제거: ({ox}, {oy})")
                    if self.estimator.is_initialized:
                        self.update_path()
                    break

    def is_obstacle(self, px, py):
        safe_margin = (self.wc_w * self.map_scale / 2) + 15

        if (self.car_x - safe_margin) <= px <= (self.car_x + self.car_dim[0] + safe_margin) and \
           (self.car_y - safe_margin) <= py <= (self.car_y + self.car_dim[1] + safe_margin):
            return True

        for ox, oy, r in self.dynamic_obstacles:
            dist = math.sqrt((px - ox) ** 2 + (py - oy) ** 2)
            if dist < (r + safe_margin):
                return True
        return False

    # ---------------- goals / stages ----------------
    def get_goal(self):
        goals = self.goals if self.parking_mode else self.exit_goals
        g = goals[self.stage][self.goal_idx]
        return (g[0], g[1]), g[2]

    def check_reached(self, pos):
        gpos, gang = self.get_goal()
        dist = math.dist(pos, gpos)
        if dist < 15:
            if gang is not None:
                angle_diff = abs(math.atan2(
                    math.sin(math.radians(gang) - self.estimator.heading_angle),
                    math.cos(math.radians(gang) - self.estimator.heading_angle)
                ))
                return angle_diff < math.radians(20)
            return True
        return False

    def advance(self):
        goals = self.goals if self.parking_mode else self.exit_goals

        if self.goal_idx < len(goals[self.stage]) - 1:
            self.goal_idx += 1
        elif self.stage < len(goals) - 1:
            self.stage += 1
            self.goal_idx = 0
            if not self.parking_mode and self.stage == 1:
                final = self.exit_goals[2][self.exit_choice][0:2]
                dists = [math.dist(final, g[0:2]) for g in self.exit_goals[1]]
                self.goal_idx = dists.index(min(dists))

        self.tracker.clear_path()
        self.goal_selected = False
        print(f"🏁 Stage {self.stage} 전환 - 기존 경로 초기화 및 재계획 예약")

    def select_nearest(self, pos):
        goals = self.goals if self.parking_mode else self.exit_goals
        if self.goal_selected or self.stage != 0:
            return
        dists = [math.dist(pos, g[0:2]) for g in goals[0]]
        self.goal_idx = dists.index(min(dists))
        self.goal_selected = True

    def update_path(self):
        if not self.estimator.is_initialized:
            return
        gpos, _ = self.get_goal()
        self.tracker.update_path(self.estimator.marker_pos, self.estimator.heading_angle, gpos)

    # ---------------- main loop ----------------
    def run(self):
        play = False

        while True:
            if play:
                ret0, self.f0 = self.cap0.read()
                ret1, self.f1 = self.cap1.read()
                if not ret0 or not ret1:
                    self.on_frame(0)
                    continue
                cv2.setTrackbarPos("Frame", self.win_name, int(self.cap0.get(cv2.CAP_PROP_POS_FRAMES)))
            else:
                target = cv2.getTrackbarPos("Frame", self.win_name)
                self.on_frame(target)

            frames = {'cam0': self.f0, 'cam1': self.f1}
            center_pos, heading_angle, is_init, per_cam_est, det_all = \
                self.estimator.detect_and_estimate(frames, self.dist_gain, apply_smoothing=play)

            mon0 = self.f0.copy() if self.f0 is not None else np.zeros((360, 640, 3), np.uint8)
            mon1 = self.f1.copy() if self.f1 is not None else np.zeros((360, 640, 3), np.uint8)

            if self.logging_active and is_init:
                current_deg = (math.degrees(heading_angle) + 90.0) % 360.0
                self.angle_logs.append(current_deg)

            img = self.visualizer.create_map()
            self.visualizer.draw_car(img)
            self.visualizer.draw_obstacles(img, self.dynamic_obstacles)

            goals = self.goals if self.parking_mode else self.exit_goals
            self.visualizer.draw_goals(img, goals, self.stage, self.goal_idx, self.parking_mode)
            if not self.parking_mode:
                self.visualizer.draw_exit_goals(img, self.exit_goals, self.exit_choice)

            self.visualizer.draw_rays_and_markers(img, det_all, self.cams)

            if is_init and center_pos is not None:
                if self.parking_mode and self.stage == 0:
                    self.select_nearest(center_pos)

                if self.check_reached(center_pos):
                    self.advance()

                self.update_path()

                path = self.tracker.get_path()
                if len(path) >= 2:
                    self.visualizer.draw_path(img, path, center_pos, heading_angle)
                    self.visualizer.draw_stage_info(img, center_pos, self.stage)

                if per_cam_est.get("cam0") is not None:
                    c, h, w = per_cam_est["cam0"]
                    self.visualizer.draw_wheelchair(img, c, h,
                                                    body_color=(255, 0, 0), front_color=(255, 255, 255),
                                                    thickness=2, label=f"REAR (w:{w:.2f})")
                if per_cam_est.get("cam1") is not None:
                    c, h, w = per_cam_est["cam1"]
                    self.visualizer.draw_wheelchair(img, c, h,
                                                    body_color=(0, 0, 255), front_color=(255, 255, 255),
                                                    thickness=2, label=f"LEFT (w:{w:.2f})")

                self.visualizer.draw_wheelchair(img, center_pos, heading_angle,
                                                body_color=(0, 255, 0), front_color=(0, 255, 255),
                                                thickness=3, label="FUSED")
                self.visualizer.draw_angle_info(img, center_pos, heading_angle)

            self.visualizer.draw_help_text(img)

            cv2.imshow(self.win_name, img)
            cv2.imshow("Monitor", np.hstack([cv2.resize(mon1, (640, 360)),
                                             cv2.resize(mon0, (640, 360))]))

            key = cv2.waitKey(30) & 0xFF
            if key == ord(' '):
                play = not play
            elif key == ord('q'):
                break
            elif key == ord('c'):
                self.dynamic_obstacles.clear()
                self.update_path()
            elif key == ord('d'):
                if not self.logging_active:
                    self.angle_logs = []
                    self.logging_active = True
                    print("📊 각도 기록 시작... (다시 'd'를 누르면 종료 및 결과 출력)")
                else:
                    self.logging_active = False
                    if len(self.angle_logs) > 0:
                        arr = np.array(self.angle_logs)
                        print("\n" + "=" * 30)
                        print(f"📈 각도 분석 결과 ({len(arr)} 샘플)")
                        print(f"  - 평균(Avg): {np.mean(arr):.2f}°")
                        print(f"  - 최대(Max): {np.max(arr):.2f}°")
                        print(f"  - 최소(Min): {np.min(arr):.2f}°")
                        print(f"  - 편차(Range): {np.max(arr) - np.min(arr):.2f}°")
                        print(f"  - 표준편차: {np.std(arr):.2f}°")
                        print("=" * 30 + "\n")
                    else:
                        print("⚠️ 기록된 데이터가 없습니다.")
                    self.dynamic_obstacles.clear()
                    self.update_path()

        self.cap0.release()
        self.cap1.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    CompactTracker().run()
