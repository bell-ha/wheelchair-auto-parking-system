import cv2
import numpy as np
import math
import heapq


class FinalOptimizedTracker:
    def __init__(self):
        # --------------------------
        # 1) 물리 및 지도 설정
        # --------------------------
        self.marker_size, self.marker_h = 25.0, 72.0
        self.map_w, self.map_h = 1000, 1000
        self.grid_w, self.grid_h = 600, 720
        self.map_scale = 0.5
        self.off_x, self.off_y = 200, 150
        self.wc_w, self.wc_l = 57.0, 100.0

        # --------------------------
        # 2) 목표/AREA
        # --------------------------
        self.exit_goal = (150 + self.off_x, 360 + self.off_y)     # 출차 최종
        self.wp_600    = (300 + self.off_x, 600 + self.off_y)     # (300,600)
        self.park_goal = (300 + self.off_x, 720 + self.off_y)     # (300,720)

        self.area_rect = (0 + self.off_x, 630 + self.off_y, 100 + self.off_x, 720 + self.off_y)
        self.area_goal = (int((self.area_rect[0] + self.area_rect[2]) / 2),
                          int((self.area_rect[1] + self.area_rect[3]) / 2))

        self.col_exit, self.col_wp, self.col_park, self.col_area = (255, 120, 0), (180, 180, 255), (0, 150, 255), (0, 255, 255)

        # --------------------------
        # 3) 상태 (여기서 marker_pos는 "휠체어 중심(center)"으로 사용)
        # --------------------------
        self.marker_pos = None          # = wheelchair center (map coord)
        self.heading_angle = 0.0
        self.is_initialized = False

        # --------------------------
        # 4) 라이브 카메라
        # --------------------------
        self.cap0 = cv2.VideoCapture(0)  # rear
        self.cap1 = cv2.VideoCapture(1)  # left
        for c in [self.cap0, self.cap1]:
            c.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            c.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.curr_f0 = None
        self.curr_f1 = None

        # 카메라 설정 (지도 상 위치는 "카메라 위치(고정)" / 실측값 유지)
        self.cams = {
            'cam1': {  # Left camera
                'pos': np.array([200.0 + self.off_x, 270.0 + self.off_y]),
                'h': 110.0, 'focal': 841.0, 'map_angle': 157, 'yaw': 1.0, 'fov': 45,
                'color': (255, 120, 100), 'name': 'Left'
            },
            'cam0': {  # Rear camera
                'pos': np.array([300.0 + self.off_x, 540.0 + self.off_y]),
                'h': 105.0, 'focal': 836.0, 'map_angle': 90, 'yaw': 1.0, 'fov': 45,
                'color': (100, 120, 255), 'name': 'Rear'
            }
        }

        # --------------------------
        # 5) 추정 파라미터
        # --------------------------
        self.dist_gain, self.angle_gain, self.alpha = 1.03, 1.56, 0.75

        self.detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250),
            cv2.aruco.DetectorParameters()
        )

        # --------------------------
        # 6) 모드/스텝
        # --------------------------
        self.MODE_EXIT = 0
        self.MODE_PARK = 1
        self.mode = self.MODE_PARK

        # PARK steps
        self.PARK_ROT_L_FIND_ID1 = 0
        self.PARK_GO_AREA        = 1
        self.PARK_WAIT_REAR_LOCK = 2
        self.PARK_GO_600         = 3
        self.PARK_BACK_720       = 4

        # EXIT steps
        self.EXIT_GO_600         = 0
        self.EXIT_ROT_R_REAR_ID1 = 1
        self.EXIT_GO_AREA        = 2
        self.EXIT_ROT_R_LEFT_ID0 = 3
        self.EXIT_GO_EXITGOAL    = 4

        self.step = self.PARK_ROT_L_FIND_ID1

        self.step_name = {
            self.PARK_ROT_L_FIND_ID1: "PARK: rotate(Left) until ID1",
            self.PARK_GO_AREA:        "PARK: go AREA",
            self.PARK_WAIT_REAR_LOCK: "PARK: wait Rear LOCK",
            self.PARK_GO_600:         "PARK: go (300,600)",
            self.PARK_BACK_720:       "PARK: back to (300,720)",

            self.EXIT_GO_600:         "EXIT: go (300,600)",
            self.EXIT_ROT_R_REAR_ID1: "EXIT: rotate(Rear) until ID1",
            self.EXIT_GO_AREA:        "EXIT: go AREA",
            self.EXIT_ROT_R_LEFT_ID0: "EXIT: rotate(Left) until ID0",
            self.EXIT_GO_EXITGOAL:    "EXIT: go (150,360)",
        }

        # --------------------------
        # 7) 독점(권한) 전환: 10프레임 연속 감지 시 잠김
        # --------------------------
        self.authority = 'cam1'           # PARK 시작은 left, EXIT 시작은 reset에서 변경
        self.authority_locked = False
        self.switch_need_frames = 10
        self.detect_streak = {'cam0': 0, 'cam1': 0}   # "해당 카메라에서 마커가 보인 프레임 연속"
        self.ids_seen = {'cam0': set(), 'cam1': set()}  # 프레임별 ids

        # --------------------------
        # 8) 제어 파라미터
        # --------------------------
        self.rotate_thresh_deg = 20.0
        self.goal_dist_px = 25.0
        self.wp_dist_px = 30.0
        self.align_thresh_deg = 20.0

        # --------------------------
        # 9) 장애물 (마우스 클릭)
        # --------------------------
        self.user_obstacles = []   # [(x,y,r), ...]
        self.user_obs_r = 28

        # --------------------------
        # 10) UI
        # --------------------------
        self.win_name = "Integrated UI Wheelchair Tracker"
        cv2.namedWindow(self.win_name)

        # 튜닝 트랙바(원하면 빼도 됨)
        cv2.createTrackbar("L_Focal", self.win_name, int(self.cams['cam1']['focal']), 1500, lambda v: self.upd('cam1','focal',v))
        cv2.createTrackbar("L_Yaw",   self.win_name, int(self.cams['cam1']['yaw'] + 90), 180, lambda v: self.upd('cam1','yaw',v-90))
        cv2.createTrackbar("R_Focal", self.win_name, int(self.cams['cam0']['focal']), 1500, lambda v: self.upd('cam0','focal',v))
        cv2.createTrackbar("R_Yaw",   self.win_name, int(self.cams['cam0']['yaw'] + 90), 180, lambda v: self.upd('cam0','yaw',v-90))

        cv2.createTrackbar("Smooth", self.win_name, int(self.alpha * 100), 100, self.on_alpha)
        cv2.createTrackbar("Dist_Gain", self.win_name, int(self.dist_gain * 100), 200, self.on_dist_gain)
        cv2.createTrackbar("Mode(0:EXIT 1:PARK)", self.win_name, int(self.mode), 1, self.on_mode_change)

        cv2.setMouseCallback(self.win_name, self.on_mouse)

        self.reset_scenario()

    # ----------------- 트랙바 콜백 -----------------
    def on_alpha(self, v):
        self.alpha = max(0.01, v / 100.0)

    def on_dist_gain(self, v):
        self.dist_gain = v / 100.0

    def upd(self, side, key, val):
        self.cams[side][key] = float(val)

    # ----------------- 마우스 -----------------
    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.user_obstacles.append((int(x), int(y), int(self.user_obs_r)))
        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.user_obstacles:
                self.user_obstacles.pop()

    # ----------------- 유틸 -----------------
    def wrap(self, rad):
        return math.atan2(math.sin(rad), math.cos(rad))

    def dist(self, a, b):
        return float(np.linalg.norm(np.array(a, float) - np.array(b, float)))

    def in_area(self, pt):
        x, y = float(pt[0]), float(pt[1])
        return (self.area_rect[0] <= x <= self.area_rect[2]) and (self.area_rect[1] <= y <= self.area_rect[3])

    # ----------------- 모드/리셋 -----------------
    def on_mode_change(self, v):
        self.mode = int(v)
        self.reset_scenario()

    def reset_scenario(self):
        self.marker_pos, self.heading_angle, self.is_initialized = None, 0.0, False
        self.authority_locked = False
        self.detect_streak = {'cam0': 0, 'cam1': 0}
        self.ids_seen = {'cam0': set(), 'cam1': set()}

        if self.mode == self.MODE_PARK:
            self.authority = 'cam1'
            self.step = self.PARK_ROT_L_FIND_ID1
        else:
            self.authority = 'cam0'
            self.step = self.EXIT_GO_600

    # ----------------- 장애물 -----------------
    def is_car_obstacle(self, px, py):
        wx, wy = px - self.off_x, py - self.off_y
        m = 45.0
        return (200 - m) <= wx <= (400 + m) and (180 - m) <= wy <= (540 + m)

    def is_user_obstacle(self, px, py):
        for (ox, oy, r) in self.user_obstacles:
            if (px - ox) ** 2 + (py - oy) ** 2 <= (r ** 2):
                return True
        return False

    def is_obstacle(self, px, py):
        return self.is_car_obstacle(px, py) or self.is_user_obstacle(px, py)

    def can_see_goal(self, start, goal):
        sx, sy = float(start[0]), float(start[1])
        gx, gy = float(goal[0]), float(goal[1])
        dist = math.hypot(gx - sx, gy - sy)
        steps = max(20, int(dist / 6))

        for i in range(steps + 1):
            t = i / steps
            x = sx * (1 - t) + gx * t
            y = sy * (1 - t) + gy * t
            if self.is_obstacle(x, y):
                return False
        return True

    # ----------------- 경로 단순화 -----------------
    def simplify_path(self, path, epsilon=12.0):
        if len(path) < 3:
            return path

        def get_dist(p, a, b):
            ap, ab = p - a, b - a
            if np.array_equal(a, b):
                return np.linalg.norm(ap)
            return np.abs(np.cross(ab, ap)) / np.linalg.norm(ab)

        dmax, idx = 0, 0
        for i in range(1, len(path) - 1):
            d = get_dist(np.array(path[i]), np.array(path[0]), np.array(path[-1]))
            if d > dmax:
                idx, dmax = i, d
        if dmax > epsilon:
            return self.simplify_path(path[:idx + 1], epsilon)[:-1] + self.simplify_path(path[idx:], epsilon)
        return [path[0], path[-1]]

    # ----------------- A* (장애물 없으면 직진) -----------------
    def astar_plan(self, start, goal):
        sn = (int(start[0]), int(start[1]))
        gn = (int(goal[0]), int(goal[1]))

        if self.is_obstacle(*sn) or self.is_obstacle(*gn):
            return []

        if self.can_see_goal(sn, gn):
            return [sn, gn]

        open_l = []
        heapq.heappush(open_l, (0, sn))
        came, g_s = {}, {sn: 0}

        moves = [(0, 7), (0, -7), (7, 0), (-7, 0),
                 (5, 5), (5, -5), (-5, 5), (-5, -5)]

        while open_l:
            _, curr = heapq.heappop(open_l)

            if math.dist(curr, gn) < 10:
                res = []
                while curr in came:
                    res.append(curr)
                    curr = came[curr]
                return self.simplify_path(res[::-1], epsilon=15.0)

            for dx, dy in moves:
                nb = (curr[0] + dx, curr[1] + dy)
                if not (0 <= nb[0] < self.map_w and 0 <= nb[1] < self.map_h):
                    continue
                if self.is_obstacle(nb[0], nb[1]):
                    continue

                tg = g_s[curr] + math.dist((0, 0), (dx, dy))
                if nb not in g_s or tg < g_s[nb]:
                    came[nb], g_s[nb] = curr, tg
                    heapq.heappush(open_l, (tg + math.dist(nb, gn), nb))
        return []

    # ----------------- 그리기 -----------------
    def draw_static_map(self, img):
        for i in range(0, 1001, 50):
            cv2.line(img, (i, 0), (i, 1000), (20, 20, 20), 1)

        step = int(20 * self.map_scale * 2)
        for x in range(0, self.grid_w + 1, step):
            c = (40, 40, 40) if x % 100 != 0 else (70, 70, 70)
            cv2.line(img, (self.off_x + x, self.off_y), (self.off_x + x, self.off_y + self.grid_h), c, 1)
        for y in range(0, self.grid_h + 1, step):
            c = (40, 40, 40) if y % 100 != 0 else (70, 70, 70)
            cv2.line(img, (self.off_x, self.off_y + y), (self.off_x + self.grid_w, self.off_y + y), c, 1)

        # 차량 고정 장애물
        cv2.rectangle(img,
                      (200 + self.off_x, 180 + self.off_y),
                      (400 + self.off_x, 540 + self.off_y),
                      (30, 30, 40), -1)

        # 작업 구역
        cv2.rectangle(img, (self.off_x, self.off_y),
                      (self.off_x + self.grid_w, self.off_y + self.grid_h),
                      (150, 150, 150), 2)

        # 카메라 점
        for side, cfg in self.cams.items():
            cp = tuple(cfg['pos'].astype(int))
            cv2.circle(img, cp, 7, cfg['color'], -1)
            cv2.putText(img, cfg['name'], (cp[0] - 20, cp[1] + 20), 0, 0.4, (200, 200, 200), 1)

        # AREA
        cv2.rectangle(img,
                      (self.area_rect[0], self.area_rect[1]),
                      (self.area_rect[2], self.area_rect[3]),
                      (0, 40, 0), -1)
        cv2.circle(img, self.area_goal, 7, self.col_area, -1)

        # 목표점
        cv2.circle(img, self.exit_goal, 8, self.col_exit, -1)
        cv2.circle(img, self.wp_600, 7, self.col_wp, -1)
        cv2.circle(img, self.park_goal, 8, self.col_park, -1)
        cv2.putText(img, "600", (self.wp_600[0] + 8, self.wp_600[1] - 8), 0, 0.45, self.col_wp, 1)
        cv2.putText(img, "720", (self.park_goal[0] + 8, self.park_goal[1] - 8), 0, 0.45, self.col_park, 1)

        # 사용자 장애물
        for (ox, oy, r) in self.user_obstacles:
            cv2.circle(img, (ox, oy), r, (60, 60, 180), -1)
            cv2.circle(img, (ox, oy), r, (120, 120, 255), 2)

    def draw_path(self, img, path, color):
        if not path or len(path) < 2:
            return
        cv2.polylines(img, [np.array(path, np.int32)], False, color, 2, cv2.LINE_AA)
        cv2.circle(img, tuple(np.array(path[-1], int)), 6, color, -1)

    def draw_cmd_and_step(self, img, cmd):
        txt = cmd if cmd else "-"
        (tw, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 2.8, 7)
        x = self.map_w - tw - 20
        y = 80
        cv2.putText(img, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2.8, (255, 255, 255), 8, cv2.LINE_AA)

        mode_str = "PARK" if self.mode == self.MODE_PARK else "EXIT"
        auth_str = self.cams[self.authority]['name']
        lock_str = "LOCK" if self.authority_locked else "FREE"
        step_str = self.step_name.get(self.step, f"step:{self.step}")

        cv2.putText(img, f"{mode_str} / {auth_str} / {lock_str}", (20, 30), 0, 0.58, (220, 220, 220), 2, cv2.LINE_AA)
        cv2.putText(img, f"{step_str}", (20, 55), 0, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(img, "LClick:add obstacle | RClick:pop | C:clear | SPACE:pause", (20, 80), 0, 0.5, (160, 160, 160), 1, cv2.LINE_AA)

    def draw_camera_lines(self, img, center):
        center_i = tuple(np.array(center, int))
        for side in ['cam1', 'cam0']:
            cp = tuple(self.cams[side]['pos'].astype(int))
            if side == self.authority:
                cv2.line(img, cp, center_i, (0, 255, 255), 3, cv2.LINE_AA)
            else:
                cv2.line(img, cp, center_i, (90, 90, 90), 1, cv2.LINE_AA)

    # ----------------- 권한 잠금 -----------------
    def update_authority_lock(self):
        if self.authority_locked:
            return

        # 주차: left 시작 -> rear 10연속 감지되면 rear 독점
        if self.mode == self.MODE_PARK and self.authority == 'cam1':
            if self.detect_streak['cam0'] >= self.switch_need_frames:
                self.authority = 'cam0'
                self.authority_locked = True

        # 출차: rear 시작 -> left 10연속 감지되면 left 독점
        if self.mode == self.MODE_EXIT and self.authority == 'cam0':
            if self.detect_streak['cam1'] >= self.switch_need_frames:
                self.authority = 'cam1'
                self.authority_locked = True

    # ----------------- (핵심) 마커 -> 센터/헤딩 추정 (1번째 코드 방식 활용) -----------------
    def estimate_from_one_marker(self, frame, corner_4x2, marker_id, cfg):
        """
        return: (center_xy, heading_rad, weight, marker_xy, d_cm, m_yaw_deg, t_rad)
        heading은 '앞(id0) 기준'으로 통일 (id1이면 +pi)
        center는 '휠체어 중심'으로 변환 (id0/id1에 따라 +-offset)
        """
        c = corner_4x2.reshape(4, 2)

        px_h = (np.linalg.norm(c[0] - c[3]) + np.linalg.norm(c[1] - c[2])) / 2.0
        if px_h < 2.0:
            return None

        raw_dist = (self.marker_size * cfg['focal']) / px_h
        corr_dist = raw_dist * (1 + (self.dist_gain - 1) * (raw_dist / 500.0))
        d = math.sqrt(max(0.0, corr_dist**2 - abs(cfg['h'] - self.marker_h)**2))

        # 화면 중심에서 좌우 얼마나 치우쳤나 [-1,1]
        rel_x = (np.mean(c[:, 0]) - frame.shape[1] / 2) / (frame.shape[1] / 2)
        weight = max(0.1, 1.0 - abs(rel_x))  # 중심에 가까울수록 weight↑

        m_yaw_deg = (rel_x * cfg['fov']) * self.angle_gain
        t_rad = math.radians(cfg['map_angle'] + cfg['yaw'] + m_yaw_deg)

        marker_xy = cfg['pos'] + np.array([
            d * self.map_scale * math.cos(t_rad),
            d * self.map_scale * math.sin(t_rad)
        ])

        # 마커 자세로 heading 계산
        marker_vec = c[0] - c[3]
        h = t_rad + math.atan2(marker_vec[1], marker_vec[0]) - (math.pi / 2.0)

        # id1이면(반대 마커) heading을 '앞(id0) 기준'으로 맞추기 위해 +pi
        if int(marker_id) == 1:
            h += math.pi
        h = self.wrap(h)

        # marker->wheelchair center 변환
        offset = (self.wc_l / 2.0) * self.map_scale
        if int(marker_id) == 0:
            center_xy = marker_xy - np.array([offset * math.cos(h), offset * math.sin(h)])
        else:
            center_xy = marker_xy + np.array([offset * math.cos(h), offset * math.sin(h)])

        return center_xy, h, weight, marker_xy, d, m_yaw_deg, t_rad

    # ----------------- 네비게이션 명령(F/L/R) -----------------
    def nav_cmd_to_target(self, center, target_xy):
        target = np.array(target_xy, float)
        if float(np.linalg.norm(target - center)) <= self.goal_dist_px:
            return "-", []

        path = self.astar_plan(center, target)
        if not path or len(path) < 2:
            return "-", []

        nxt = np.array(path[1], float)
        target_yaw = math.atan2(nxt[1] - center[1], nxt[0] - center[0])
        yaw_err = self.wrap(target_yaw - self.heading_angle)
        yaw_deg = math.degrees(yaw_err)

        # (중요) y-down 좌표계에서 +각도는 시계방향(=Right)이라 yaw_err>0 -> R
        if abs(yaw_deg) > self.rotate_thresh_deg:
            return ("R" if yaw_deg > 0 else "L"), path
        return "F", path

    # ----------------- 정렬 유지 후진(B/L/R) -----------------
    def back_cmd_keep_aligned(self, center, target_xy, cam_pos_xy, require_id0_in_rear=True):
        if self.dist(center, target_xy) <= self.goal_dist_px:
            return "-"

        if require_id0_in_rear and (0 not in self.ids_seen['cam0']):
            return "R"  # rear에서 id0이 안 보이면 회전 우선

        desired = math.atan2(cam_pos_xy[1] - center[1], cam_pos_xy[0] - center[0])
        err = self.wrap(desired - self.heading_angle)
        err_deg = abs(math.degrees(err))

        if err_deg > self.align_thresh_deg:
            return ("R" if err > 0 else "L")
        return "B"

    # ----------------- 시나리오 스텝 로직 -----------------
    def scenario_cmd(self, center):
        left_ids = self.ids_seen['cam1']
        rear_ids = self.ids_seen['cam0']

        if self.mode == self.MODE_PARK:
            # 0) left에서 id1 보일 때까지 회전
            if self.step == self.PARK_ROT_L_FIND_ID1:
                if 1 in left_ids:
                    self.step = self.PARK_GO_AREA
                    return "-", []
                return "R", []

            # 1) AREA로 이동
            if self.step == self.PARK_GO_AREA:
                cmd, path = self.nav_cmd_to_target(center, self.area_goal)
                if self.in_area(center):
                    self.step = self.PARK_WAIT_REAR_LOCK
                    return "-", path
                return cmd, path

            # 2) AREA에서 Rear LOCK 될 때까지 대기
            if self.step == self.PARK_WAIT_REAR_LOCK:
                if self.authority_locked and self.authority == 'cam0':
                    self.step = self.PARK_GO_600
                    return "-", []
                return "-", []

            # 3) (300,600)으로 이동
            if self.step == self.PARK_GO_600:
                cmd, path = self.nav_cmd_to_target(center, self.wp_600)
                if self.dist(center, self.wp_600) <= self.wp_dist_px:
                    self.step = self.PARK_BACK_720
                    return "-", path
                return cmd, path

            # 4) 정렬 유지 후진으로 (300,720)
            if self.step == self.PARK_BACK_720:
                rear_cam = self.cams['cam0']['pos']
                cmd = self.back_cmd_keep_aligned(center, self.park_goal, rear_cam, require_id0_in_rear=True)
                path = self.astar_plan(center, self.park_goal)  # 표시용
                return cmd, path

        else:
            # EXIT 0) (300,720)->(300,600)
            if self.step == self.EXIT_GO_600:
                cmd, path = self.nav_cmd_to_target(center, self.wp_600)
                if self.dist(center, self.wp_600) <= self.wp_dist_px:
                    self.step = self.EXIT_ROT_R_REAR_ID1
                    return "-", path
                return cmd, path

            # 1) (300,600)에서 rear에 id1 보일 때까지 회전
            if self.step == self.EXIT_ROT_R_REAR_ID1:
                if 1 in rear_ids:
                    self.step = self.EXIT_GO_AREA
                    return "-", []
                return "R", []

            # 2) AREA로 이동
            if self.step == self.EXIT_GO_AREA:
                cmd, path = self.nav_cmd_to_target(center, self.area_goal)
                if self.in_area(center):
                    self.step = self.EXIT_ROT_R_LEFT_ID0
                    return "-", path
                return cmd, path

            # 3) Left LOCK 상태에서 id0 보일 때까지 회전
            if self.step == self.EXIT_ROT_R_LEFT_ID0:
                if not (self.authority_locked and self.authority == 'cam1'):
                    return "-", []
                if 0 in left_ids:
                    self.step = self.EXIT_GO_EXITGOAL
                    return "-", []
                return "R", []

            # 4) (150,360)으로 이동
            if self.step == self.EXIT_GO_EXITGOAL:
                cmd, path = self.nav_cmd_to_target(center, self.exit_goal)
                return cmd, path

        return "-", []

    # ----------------- 실행 -----------------
    def run(self):
        play = True

        while True:
            if play:
                ret0, self.curr_f0 = self.cap0.read()
                ret1, self.curr_f1 = self.cap1.read()
                if not ret0 or not ret1:
                    # 라이브에서는 frame seek 같은 거 하지 말고 그냥 skip
                    continue

            m_map = np.ones((self.map_h, self.map_w, 3), dtype=np.uint8) * 12
            self.draw_static_map(m_map)

            mon0 = self.curr_f0.copy() if self.curr_f0 is not None else np.zeros((720, 1280, 3), np.uint8)
            mon1 = self.curr_f1.copy() if self.curr_f1 is not None else np.zeros((720, 1280, 3), np.uint8)

            # 프레임별 ids 초기화
            self.ids_seen = {'cam0': set(), 'cam1': set()}

            # (핵심) 이번 프레임에서 얻은 포즈 측정치들(센터/헤딩/가중치)
            measurements = []

            # 감지 루프
            for frame, mon, side in [(self.curr_f0, mon0, 'cam0'), (self.curr_f1, mon1, 'cam1')]:
                if frame is None:
                    self.detect_streak[side] = 0
                    continue

                corners, ids, _ = self.detector.detectMarkers(frame)

                if ids is None or len(ids) == 0:
                    self.detect_streak[side] = 0
                    continue

                # streak / ids 기록
                self.detect_streak[side] += 1
                ids_list = ids.flatten().tolist()
                self.ids_seen[side] = set(ids_list)

                # 모니터에 마커 표시
                cv2.aruco.drawDetectedMarkers(mon, corners, ids)

                cfg = self.cams[side]
                cp = tuple(cfg['pos'].astype(int))

                # 여러 마커가 잡히면 각각을 측정치로 넣음 (id0/id1만 사용)
                for i, mid in enumerate(ids_list):
                    if mid not in (0, 1):
                        continue

                    est = self.estimate_from_one_marker(frame, corners[i], mid, cfg)
                    if est is None:
                        continue

                    center_xy, h, w, marker_xy, d_cm, m_yaw_deg, t_rad = est

                    # authority 카메라의 측정치를 살짝 더 믿고 싶으면 가중치 보정(선택)
                    if side == self.authority:
                        w *= 1.15

                    measurements.append((center_xy, h, w))

                    # 디버그: 카메라->마커 선/호/텍스트 (원하면 주석처리 가능)
                    rp = tuple(marker_xy.astype(int))
                    dist_px = int(d_cm * self.map_scale)
                    cv2.ellipse(m_map, cp, (dist_px, dist_px), 0,
                                math.degrees(t_rad) - 5, math.degrees(t_rad) + 5,
                                cfg['color'], 2, cv2.LINE_AA)
                    cv2.line(m_map, cp, rp, cfg['color'], 1, cv2.LINE_AA)
                    txt_pos = ((cp[0] + rp[0]) // 2, (cp[1] + rp[1]) // 2)
                    cv2.putText(m_map, f"id{mid} {d_cm:.0f}cm / {m_yaw_deg:+.1f}deg (w:{w:.1f})",
                                (txt_pos[0] + 5, txt_pos[1] - 5), 0, 0.4,
                                (180, 180, 180), 1, cv2.LINE_AA)

            # 독점 업데이트(10프레임 연속 조건)
            self.update_authority_lock()

            # 포즈 융합 업데이트 (1번째 코드 방식: 가중 평균 + 각도 벡터 평균 + smoothing)
            if len(measurements) > 0:
                total_w = sum(m[2] for m in measurements)
                avg_center = sum(m[0] * m[2] for m in measurements) / total_w
                avg_sin = sum(math.sin(m[1]) * m[2] for m in measurements) / total_w
                avg_cos = sum(math.cos(m[1]) * m[2] for m in measurements) / total_w
                avg_h = math.atan2(avg_sin, avg_cos)

                if not self.is_initialized:
                    self.marker_pos = avg_center
                    self.heading_angle = avg_h
                    self.is_initialized = True
                else:
                    self.marker_pos = self.marker_pos * (1 - self.alpha) + avg_center * self.alpha
                    # 각도는 wrap된 벡터 평균 방식으로 부드럽게
                    self.heading_angle = math.atan2(
                        math.sin(self.heading_angle) * (1 - self.alpha) + math.sin(avg_h) * self.alpha,
                        math.cos(self.heading_angle) * (1 - self.alpha) + math.cos(avg_h) * self.alpha
                    )

            cmd = "-"
            path_to_draw = []

            if self.is_initialized and self.marker_pos is not None:
                center = self.marker_pos.copy()

                # 시나리오 명령 + 다음 스텝 목표까지 경로
                cmd, path_to_draw = self.scenario_cmd(center)

                if path_to_draw:
                    color = (0, 180, 255) if self.mode == self.MODE_PARK else (255, 140, 0)
                    self.draw_path(m_map, path_to_draw, color)

                # 휠체어 시각화(중심 + heading)
                w, l = (self.wc_w * self.map_scale) / 2, (self.wc_l * self.map_scale) / 2
                rot = np.array([[math.cos(self.heading_angle), -math.sin(self.heading_angle)],
                                [math.sin(self.heading_angle),  math.cos(self.heading_angle)]])
                pts = np.dot(np.array([[-l, -w], [l, -w], [l, w], [-l, w]]), rot.T) + center
                cv2.polylines(m_map, [pts.astype(np.int32)], True, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.line(m_map, tuple(pts[0].astype(int)), tuple(pts[3].astype(int)), (0, 0, 255), 3)

                # 활성/비활성 카메라 선
                self.draw_camera_lines(m_map, center)

            # UI
            self.draw_cmd_and_step(m_map, cmd)

            cv2.imshow(self.win_name, m_map)

            # Monitor 창(Left, Rear)
            m1_res = cv2.resize(mon1, (640, 360))
            m0_res = cv2.resize(mon0, (640, 360))
            cv2.imshow("Monitor", np.hstack([m1_res, m0_res]))

            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                play = not play
            elif key == ord('c'):
                self.user_obstacles.clear()
            elif key == ord('q'):
                break

        self.cap0.release()
        self.cap1.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    FinalOptimizedTracker().run()
