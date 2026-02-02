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
        # 맵 설정
        self.map_w, self.map_h = 1200, 1200
        self.grid_w, self.grid_h = 800, 900
        self.off_x, self.off_y = 200, 150
        self.map_scale = 0.5
        self.wc_w, self.wc_l = 57.0, 100.0

        # 데이터 기록용 변수
        self.logging_active = False
        self.angle_logs = []

        # 마커 (cm)
        self.marker_size = 25.0

        # 카메라 위치/설치(기본값 유지)
        car_cx, car_cy = self.off_x + self.grid_w/2, self.off_y + self.grid_h/2
        self.cams = {
            'cam0': {  # Rear
                'name': 'REAR',
                'pos_px': np.array([car_cx + 1.4, car_cy + 170], dtype=np.float32),
                'h_cm': 105.0,
                'map_angle_deg': 90.0,
                'yaw_trim_deg': 0.0,
                'sens': 1.6,
                'install_angle': 0.0,
                'install_offset': 0.0,
                'map_scale': self.map_scale,
                'color': (255, 0, 0)
            },
            'cam1': {  # Left
                'name': 'LEFT',
                'pos_px': np.array([car_cx - 140, car_cy - 135], dtype=np.float32),
                'h_cm': 110.0,
                'map_angle_deg': 157.0,
                'yaw_trim_deg': 0.0,
                'sens': 1.6,
                'install_angle': 113.0,
                'install_offset': 50.84,
                'map_scale': self.map_scale,
                'color': (0, 0, 255)
            }
        }

        # ✅ 내부 파라미터 기본값(= 시작 시 실제 적용되는 값)
        # (주의: 아래 INIT_* 트랙바 값은 "표시만", 시작 시 강제 적용하지 않습니다)
        self.alpha = 0.25
        self.dist_gain = 2.0

        # 차량
        self.car_dim = [200, 360]
        self.car_x, self.car_y = car_cx - self.car_dim[0]/2, car_cy - self.car_dim[1]/1.6
        car_rear_y = self.car_y + self.car_dim[1] + 150

        # 시나리오
        self.parking_mode = True
        self.goals = [
            [(car_cx, car_rear_y+100, -90)],
            [(car_cx, car_rear_y+100, -90)],
            [(car_cx, car_rear_y-30, -90)]
        ]
        self.exit_goals = [
            [(car_cx, car_rear_y+70, -90)],
            [(car_cx-230, self.off_y+400, None), (car_cx+230, self.off_y+400, None)],
            [(car_cx-230, self.off_y+400, None), (car_cx+230, self.off_y+400, None)]
        ]

        self.stage, self.goal_idx = 0, 0
        self.exit_choice = 0
        self.goal_selected = False

        # 동적 장애물
        self.dynamic_obstacles = []

        # 모듈 초기화
        self.planner = PathPlanner(self.map_w, self.map_h, self.wc_w, self.map_scale)
        self.planner.set_obstacle_checker(self.is_obstacle)

        self.tracker = PathTracker(self.wc_l, self.map_scale)
        self.tracker.set_planner(self.planner)
        self.tracker.set_obstacle_checker(self.is_obstacle)

        # ✅ PoseEstimator는 self.alpha(실제값)를 사용 (INIT_SMOOTH 표시값과는 무관)
        self.estimator = PoseEstimator(K, D, self.cams, self.marker_size, self.alpha, wc_l_cm=self.wc_l)

        self.visualizer = Visualizer(self.map_w, self.map_h, self.car_dim,
                                     (self.car_x, self.car_y), self.wc_w, self.wc_l, self.map_scale)

        # 영상
        self.cap0 = cv2.VideoCapture('test_rear.mp4')
        self.cap1 = cv2.VideoCapture('test_left.mp4')
        self.total_frames = int(min(self.cap0.get(cv2.CAP_PROP_FRAME_COUNT),
                                    self.cap1.get(cv2.CAP_PROP_FRAME_COUNT)))

        self.win_name = "Compact Tracker (cam0/cam1 + FUSED)"
        cv2.namedWindow(self.win_name)
        cv2.setMouseCallback(self.win_name, self.mouse_callback)

        # ====== Trackbars (표시용 INIT 값) ======
        INIT_FRAME = 278

        INIT_SMOOTH = 100          # 표시상 1.00
        INIT_DISTGAIN = 188        # 표시상 1.88

        INIT_REAR_MAPYAW = 93      # 표시상 yaw_trim +3
        INIT_LEFT_MAPYAW = 98      # 표시상 yaw_trim +8
        INIT_REAR_SENS_X10 = 16    # 표시상 1.6
        INIT_LEFT_SENS_X10 = 16
        INIT_REAR_INSTANGLE = 0
        INIT_LEFT_INSTANGLE = 113
        INIT_REAR_INSTOFFSET_X10 = 0
        INIT_LEFT_INSTOFFSET_X10 = 508

        INIT_FRAME = int(max(0, min(INIT_FRAME, self.total_frames - 1)))

        # ✅ 여기서부터가 포인트:
        # - 트랙바는 INIT_*로 만들지만
        # - self.alpha/self.dist_gain/self.cams 값을 INIT_*로 "강제 대입"하지 않습니다.
        # - 그래서 시작 시 실제 적용되는 값은 위에서 정한 기본값(0.25, 2.0, cams 기본값)입니다.

        cv2.createTrackbar("Frame", self.win_name, INIT_FRAME, max(0, self.total_frames - 1), self.on_frame)
        cv2.createTrackbar("Mode", self.win_name, 1, 1, self.on_mode)
        cv2.createTrackbar("ExitDir", self.win_name, 0, 1, self.on_exit)

        cv2.createTrackbar("Smooth(%)", self.win_name, INIT_SMOOTH, 100, self.on_alpha)
        cv2.createTrackbar("DistGain(%)", self.win_name, INIT_DISTGAIN, 300, self.on_dist_gain)

        cv2.createTrackbar("Rear_MapYaw", self.win_name, INIT_REAR_MAPYAW, 180,
                           lambda v: self.set_cam("cam0", "yaw_trim_deg", v - 90))
        cv2.createTrackbar("Left_MapYaw", self.win_name, INIT_LEFT_MAPYAW, 180,
                           lambda v: self.set_cam("cam1", "yaw_trim_deg", v - 90))

        cv2.createTrackbar("Rear_Sens(x10)", self.win_name, INIT_REAR_SENS_X10, 30,
                           lambda v: self.set_cam("cam0", "sens", v / 10.0))
        cv2.createTrackbar("Left_Sens(x10)", self.win_name, INIT_LEFT_SENS_X10, 30,
                           lambda v: self.set_cam("cam1", "sens", v / 10.0))

        cv2.createTrackbar("Rear_InstAngle", self.win_name, INIT_REAR_INSTANGLE, 180,
                           lambda v: self.set_cam("cam0", "install_angle", float(v)))
        cv2.createTrackbar("Left_InstAngle", self.win_name, INIT_LEFT_INSTANGLE, 180,
                           lambda v: self.set_cam("cam1", "install_angle", float(v)))

        cv2.createTrackbar("Rear_InstOffset(x10)", self.win_name, INIT_REAR_INSTOFFSET_X10, 1800,
                           lambda v: self.set_cam("cam0", "install_offset", v / 10.0))
        cv2.createTrackbar("Left_InstOffset(x10)", self.win_name, INIT_LEFT_INSTOFFSET_X10, 1800,
                           lambda v: self.set_cam("cam1", "install_offset", v / 10.0))

        # 시작 프레임 로드
        self.on_frame(INIT_FRAME)

    # ---------------- UI callbacks ----------------
    def set_cam(self, cam_key, key, val):
        self.cams[cam_key][key] = float(val)

    def on_frame(self, v):
        self.cap0.set(cv2.CAP_PROP_POS_FRAMES, v)
        self.cap1.set(cv2.CAP_PROP_POS_FRAMES, v)
        _, self.f0 = self.cap0.read()
        _, self.f1 = self.cap1.read()

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
                if math.sqrt((ox-x)**2 + (oy-y)**2) < r:
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
            dist = math.sqrt((px - ox)**2 + (py - oy)**2)
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
