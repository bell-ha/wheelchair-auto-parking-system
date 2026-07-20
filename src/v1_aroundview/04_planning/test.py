import cv2
import numpy as np
import math
import heapq
from collections import deque  # ✅ 추가

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
        
        # 마커 및 카메라
        self.marker_size, self.marker_h = 25.0, 72.0
        car_cx, car_cy = self.off_x + self.grid_w/2, self.off_y + self.grid_h/2
        self.cams = {
            'cam1': {'pos': np.array([car_cx-140, car_cy-135]), 'h': 110, 'focal': 950, 'map_angle': 157, 'yaw': 1, 'fov': 45, 'color': (255,120,100)},
            'cam0': {'pos': np.array([car_cx+1.4, car_cy+170]), 'h': 105, 'focal': 950, 'map_angle': 90, 'yaw': 1, 'fov': 45, 'color': (100,120,255)}
        }
        self.dist_gain, self.angle_gain, self.alpha = 2.0, 1.56, 0.75
        self.detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250),
            cv2.aruco.DetectorParameters()
        )
        
        # 차량
        self.car_dim = [200, 360]
        self.car_x, self.car_y = car_cx - self.car_dim[0]/2, car_cy - self.car_dim[1]/1.6
        car_rear_y = self.car_y + self.car_dim[1] + 150
        
        # 시나리오 (간소화)
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
        self.path = []
        
        # 상태
        self.marker_pos, self.heading_angle, self.is_initialized = None, 0.0, False
        
        # ✅ 각도 평균 버퍼 (최근 15프레임)
        self.HEADING_WIN = 15
        self.heading_hist = deque(maxlen=self.HEADING_WIN)  # [(angle_rad, weight), ...]

        # 동적 장애물
        self.dynamic_obstacles = []
        
        # 라이브 카메라
        self.cap0 = cv2.VideoCapture(0)
        self.cap1 = cv2.VideoCapture(1)

        # ⚠️ 라이브면 프레임 카운트가 0/NaN일 수 있음 → 트랙바 Frame 제거 권장
        self.total_frames = int(min(self.cap0.get(cv2.CAP_PROP_FRAME_COUNT), self.cap1.get(cv2.CAP_PROP_FRAME_COUNT)))

        self.win_name = "Compact Tracker"
        cv2.namedWindow(self.win_name)
        cv2.setMouseCallback(self.win_name, self.mouse_callback)

        # ✅ 라이브에서는 Frame 트랙바가 의미 없고 에러도 나기 쉬움
        # 아래 2줄은 영상 파일일 때만 사용하세요.
        # cv2.createTrackbar("Frame", self.win_name, 278, self.total_frames-1, self.on_frame)
        # self.on_frame(278)

        cv2.createTrackbar("Mode", self.win_name, 1, 1, self.on_mode)
        cv2.createTrackbar("ExitDir", self.win_name, 0, 1, self.on_exit)

        # 초기 프레임
        self.f0 = None
        self.f1 = None
    
    # ✅ 원형 평균(가중치 포함)
    @staticmethod
    def circular_mean_weighted(angle_weight_list):
        """
        angle_weight_list: [(rad, w), ...]
        return: mean rad in [-pi, pi]
        """
        if not angle_weight_list:
            return None
        s = 0.0
        c = 0.0
        for ang, w in angle_weight_list:
            s += math.sin(ang) * w
            c += math.cos(ang) * w
        if abs(s) < 1e-9 and abs(c) < 1e-9:
            return None
        return math.atan2(s, c)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dynamic_obstacles.append((x, y, 30))
            print(f"➕ 장애물 추가: ({x}, {y})")
            if self.is_initialized:
                self.update_path()
        elif event == cv2.EVENT_RBUTTONDOWN:
            for i, (ox, oy, r) in enumerate(self.dynamic_obstacles):
                if math.sqrt((ox-x)**2 + (oy-y)**2) < r:
                    self.dynamic_obstacles.pop(i)
                    print(f"➖ 장애물 제거: ({ox}, {oy})")
                    if self.is_initialized:
                        self.update_path()
                    break
    
    def on_frame(self, v):
        self.cap0.set(cv2.CAP_PROP_POS_FRAMES, v)
        self.cap1.set(cv2.CAP_PROP_POS_FRAMES, v)
        _, self.f0 = self.cap0.read()
        _, self.f1 = self.cap1.read()
    
    def on_mode(self, v):
        self.parking_mode = (v == 1)
        self.stage, self.goal_idx, self.goal_selected = 0, 0, False
        self.path = []
    
    def on_exit(self, v):
        self.exit_choice = v
        if not self.parking_mode and self.stage == 1:
            final = self.exit_goals[2][v][0:2]
            dists = [math.dist(final, g[0:2]) for g in self.exit_goals[1]]
            self.goal_idx = dists.index(min(dists))
    
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

    def interpolate_path(self, path, interval=30.0):
        if len(path) < 2:
            return path
        new_path = []
        for i in range(len(path) - 1):
            p1 = np.array(path[i])
            p2 = np.array(path[i+1])
            dist = math.dist(p1, p2)
            new_path.append(path[i])
            if dist > interval:
                num_points = int(dist // interval)
                for j in range(1, num_points + 1):
                    t = j / (num_points + 1)
                    inter_pt = p1 * (1 - t) + p2 * t
                    new_path.append(inter_pt.tolist())
        new_path.append(path[-1])
        return new_path

    def astar(self, start, goal):
        sn, gn = (int(start[0]), int(start[1])), (int(goal[0]), int(goal[1]))
        if self.is_obstacle(*sn):
            return [start, goal]

        ALLOWED_SLOPE = math.radians(20)
        SLOPE_PENALTY_WEIGHT = 50.0

        open_l = []
        heapq.heappush(open_l, (0, sn, (0, 0)))
        came, g_s = {}, {sn: 0}

        ROTATION_PENALTY = 100.0

        while open_l:
            _, curr, prev_dir = heapq.heappop(open_l)
            if math.dist(curr, gn) < 25:
                res = [list(curr)]
                while curr in came:
                    curr = came[curr]
                    res.append(list(curr))
                simplified = self.simplify_path(res[::-1], epsilon=20.0)
                return self.interpolate_path(simplified, interval=30.0)

            for dx, dy in [(0,12),(0,-12),(12,0),(-12,0),(9,9),(9,-9),(-9,9),(-9,-9)]:
                nb = (curr[0] + dx, curr[1] + dy)
                if not (0 <= nb[0] < self.map_w and 0 <= nb[1] < self.map_h) or self.is_obstacle(*nb):
                    continue

                move_cost = math.dist(curr, nb)

                slope_penalty = 0.0
                if dx != 0:
                    current_slope = math.atan2(abs(dx), abs(dy))
                    if current_slope > ALLOWED_SLOPE:
                        slope_penalty = SLOPE_PENALTY_WEIGHT * (current_slope / (math.pi/2))

                rot_penalty = ROTATION_PENALTY if (prev_dir != (0, 0) and prev_dir != (dx, dy)) else 0
                tg = g_s[curr] + move_cost + slope_penalty + rot_penalty

                if nb not in g_s or tg < g_s[nb]:
                    came[nb], g_s[nb] = curr, tg
                    f_score = tg + math.dist(nb, gn) * 1.5
                    heapq.heappush(open_l, (f_score, nb, (dx, dy)))

        return [start, goal]
    
    def simplify_path(self, path, epsilon=5.0):
        if len(path) < 3:
            return path
        pts = np.array(path)

        def get_dist(p, a, b):
            if np.array_equal(a, b):
                return np.linalg.norm(p - a)
            return np.abs(np.cross(b - a, a - p)) / np.linalg.norm(b - a)

        dmax, idx = 0, 0
        for i in range(1, len(pts) - 1):
            d = get_dist(pts[i], pts[0], pts[-1])
            if d > dmax:
                idx, dmax = i, d

        if dmax > epsilon:
            left = self.simplify_path(path[:idx+1], epsilon)
            right = self.simplify_path(path[idx:], epsilon)
            return left[:-1] + right

        return [path[0], path[-1]]
    
    def get_goal(self):
        goals = self.goals if self.parking_mode else self.exit_goals
        g = goals[self.stage][self.goal_idx]
        return (g[0], g[1]), g[2]
    
    def check_reached(self, pos):
        gpos, gang = self.get_goal()
        dist = math.dist(pos, gpos)
        if dist < 15:
            if gang is not None:
                angle_diff = abs(math.atan2(math.sin(math.radians(gang)-self.heading_angle),
                                            math.cos(math.radians(gang)-self.heading_angle)))
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

        self.path = []
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
        if not self.is_initialized:
            return
        
        center = self.marker_pos + np.array([(self.wc_l/2)*self.map_scale*math.cos(self.heading_angle),
                                             (self.wc_l/2)*self.map_scale*math.sin(self.heading_angle)])
        gpos, _ = self.get_goal()

        need_replan = False

        if not self.path or len(self.path) < 2:
            need_replan = True
        else:
            for i in range(len(self.path)-1):
                p1, p2 = np.array(self.path[i]), np.array(self.path[i+1])
                for t in [0.3, 0.6, 0.9]:
                    check_pt = p1 * (1-t) + p2 * t
                    if self.is_obstacle(check_pt[0], check_pt[1]):
                        need_replan = True
                        break
                if need_replan:
                    break

            min_d = float('inf')
            for i in range(len(self.path)-1):
                p1, p2 = np.array(self.path[i]), np.array(self.path[i+1])
                line_vec = p2 - p1
                p_vec = center - p1
                line_len = np.sum(line_vec**2)
                if line_len == 0:
                    d = math.dist(center, p1)
                else:
                    t = max(0, min(1, np.dot(p_vec, line_vec) / line_len))
                    projection = p1 + t * line_vec
                    d = math.dist(center, projection)
                min_d = min(min_d, d)

            if min_d > 70:
                need_replan = True

        if need_replan:
            self.path = self.astar(center, gpos)
            print("🔄 경로 재계획 실행")
        else:
            if len(self.path) > 1:
                p1 = np.array(self.path[0])
                p2 = np.array(self.path[1])

                v_path = p2 - p1
                v_wc = center - p1

                dist_to_p1 = math.dist(center, p1)
                dot_product = np.dot(v_path, v_wc)

                if dist_to_p1 < 25 or dot_product > 0:
                    if len(self.path) > 2:
                        self.path.pop(0)
    
    def draw_map(self, img):
        for i in range(0, self.map_w, 50):
            cv2.line(img, (i,0), (i,self.map_h), (25,25,25), 1)
        for i in range(0, self.map_h, 50):
            cv2.line(img, (0,i), (self.map_w,i), (25,25,25), 1)

        cv2.rectangle(img, (int(self.car_x), int(self.car_y)),
                      (int(self.car_x+self.car_dim[0]), int(self.car_y+self.car_dim[1])), (35,35,45), -1)

        for ox, oy, r in self.dynamic_obstacles:
            cv2.circle(img, (ox, oy), r, (0, 0, 150), -1)
            cv2.circle(img, (ox, oy), r, (0, 0, 255), 2)

        goals = self.goals if self.parking_mode else self.exit_goals
        for si, stage_goals in enumerate(goals):
            for gi, g in enumerate(stage_goals):
                gp = (int(g[0]), int(g[1]))
                is_curr = (si == self.stage and gi == self.goal_idx)
                col = (0,255,0) if is_curr else (100,100,100)
                cv2.circle(img, gp, 10, col, -1 if is_curr else 2)
                cv2.putText(img, f"S{si}", (gp[0]-8, gp[1]-15), 0, 0.4, col, 1)
                if g[2] is not None:
                    ax = int(gp[0] + 25*math.cos(math.radians(g[2])))
                    ay = int(gp[1] + 25*math.sin(math.radians(g[2])))
                    cv2.arrowedLine(img, gp, (ax,ay), (150,150,255), 2, tipLength=0.4)

        if not self.parking_mode:
            for i, g in enumerate(self.exit_goals[2]):
                gp = (int(g[0]), int(g[1]))
                col = (255,100,0) if i == self.exit_choice else (80,80,80)
                cv2.circle(img, gp, 8, col, -1 if i == self.exit_choice else 2)
    
    def draw_path(self, img):
        if len(self.path) < 2:
            return
        cv2.polylines(img, [np.array(self.path, np.int32)], False, (0,255,255), 2)

        pivot = self.marker_pos
        target = self.path[-1]
        dx, dy = target[0]-pivot[0], target[1]-pivot[1]
        target_yaw = math.atan2(dy, dx)
        yaw_err = math.degrees(math.atan2(math.sin(target_yaw-self.heading_angle),
                                          math.cos(target_yaw-self.heading_angle)))

        cv2.ellipse(img, (int(pivot[0]), int(pivot[1])), (45,45), 0,
                    -math.degrees(self.heading_angle), -math.degrees(target_yaw),
                    (0,200,255) if yaw_err>0 else (255,150,0), 2)

        cv2.putText(img, f"Rot: {yaw_err:+.1f}deg", (int(pivot[0])+50, int(pivot[1])-70),
                    0, 0.5, (0,255,255), 1)
        cv2.putText(img, f"Stage: {self.stage}", (int(pivot[0])+50, int(pivot[1])-55),
                    0, 0.4, (255,200,100), 1)
    
    def run(self):
        play = True
        while True:
            # ✅ 라이브 프레임 읽기
            ret0, self.f0 = self.cap0.read()
            ret1, self.f1 = self.cap1.read()
            if not ret0 or not ret1:
                continue
            
            img = np.ones((self.map_h, self.map_w, 3), dtype=np.uint8) * 15
            self.draw_map(img)
            
            detected_data = []
            mon0 = self.f0.copy() if self.f0 is not None else np.zeros((360,640,3), np.uint8)
            mon1 = self.f1.copy() if self.f1 is not None else np.zeros((360,640,3), np.uint8)

            for frame, side in [(self.f0, 'cam0'), (self.f1, 'cam1')]:
                if frame is None:
                    continue

                corners, ids, _ = self.detector.detectMarkers(frame)
                if ids is None:
                    continue

                cfg = self.cams[side]

                pts_2d = corners[0].reshape(-1, 1, 2)
                undistorted_pts = cv2.fisheye.undistortPoints(pts_2d, K, D, P=K)

                ms = self.marker_size
                obj_points = np.array([
                    [-ms/2,  ms/2, 0],
                    [ ms/2,  ms/2, 0],
                    [ ms/2, -ms/2, 0],
                    [-ms/2, -ms/2, 0]
                ], dtype=np.float32)

                ret, rvec, tvec = cv2.solvePnP(
                    obj_points, undistorted_pts, K, None,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE
                )

                if not ret:
                    continue

                tvec = tvec.flatten()
                x_offset, y_offset, z_dist = tvec

                d_raw = np.linalg.norm(tvec)
                d = d_raw * (1 + (self.dist_gain - 1) * (d_raw / 500))

                dh = abs(cfg['h'] - self.marker_h)
                ground_d = math.sqrt(max(0, d**2 - dh**2))

                ray_angle = math.atan2(x_offset, z_dist)
                cam_global_angle = math.radians(cfg['map_angle'] + cfg['yaw'])
                t_rad = cam_global_angle + ray_angle

                pos = cfg['pos'] + np.array([
                    ground_d * self.map_scale * math.cos(t_rad),
                    ground_d * self.map_scale * math.sin(t_rad)
                ])

                R, _ = cv2.Rodrigues(rvec)
                sy = math.sqrt(R[0, 0]**2 + R[1, 0]**2)
                if sy < 1e-6:
                    local_yaw = math.atan2(-R[1, 2], R[1, 1])
                else:
                    local_yaw = math.atan2(R[1, 0], R[0, 0])

                h = cam_global_angle + local_yaw + math.pi
                if ids[0][0] == 1:
                    h += math.pi

                rel_x = (np.mean(corners[0][:, 0, 0]) - frame.shape[1]/2) / (frame.shape[1]/2)
                weight = max(0.1, 1.0 - abs(rel_x))

                detected_data.append((pos, h, weight))

                # 모니터에도 표시(원하면)
                cv2.aruco.drawDetectedMarkers(mon0 if side == 'cam0' else mon1, corners, ids)

            # ✅ 데이터 통합 + 각도 15프레임 평균 적용
            if len(detected_data) > 0:
                total_w = sum(p[2] for p in detected_data)
                avg_pos = sum(p[0] * p[2] for p in detected_data) / total_w

                # 한 프레임 instantaneous heading (원형평균)
                avg_sin = sum(math.sin(p[1]) * p[2] for p in detected_data) / total_w
                avg_cos = sum(math.cos(p[1]) * p[2] for p in detected_data) / total_w
                avg_h_inst = math.atan2(avg_sin, avg_cos)

                # ✅ 최근 15프레임 버퍼에 저장(가중치는 total_w로)
                self.heading_hist.append((avg_h_inst, total_w))
                avg_h_win = self.circular_mean_weighted(self.heading_hist) or avg_h_inst

                if not self.is_initialized:
                    self.marker_pos, self.heading_angle, self.is_initialized = avg_pos, avg_h_win, True
                else:
                    self.marker_pos = self.marker_pos * (1 - self.alpha) + avg_pos * self.alpha
                    diff = (avg_h_win - self.heading_angle + math.pi) % (2 * math.pi) - math.pi
                    self.heading_angle += diff * self.alpha

                center = self.marker_pos + np.array([(self.wc_l/2)*self.map_scale*math.cos(self.heading_angle),
                                                     (self.wc_l/2)*self.map_scale*math.sin(self.heading_angle)])

                if self.parking_mode and self.stage == 0:
                    self.select_nearest(center)
                if self.check_reached(center):
                    self.advance()
                self.update_path()
            
            if self.is_initialized:
                self.draw_path(img)

                center = self.marker_pos + np.array([(self.wc_l/2)*self.map_scale*math.cos(self.heading_angle),
                                                     (self.wc_l/2)*self.map_scale*math.sin(self.heading_angle)])
                w, l = (self.wc_w*self.map_scale)/2, (self.wc_l*self.map_scale)/2
                rot = np.array([[math.cos(self.heading_angle), -math.sin(self.heading_angle)],
                                [math.sin(self.heading_angle),  math.cos(self.heading_angle)]])
                pts = np.dot([[-l,-w],[l,-w],[l,w],[-l,w]], rot.T) + center
                cv2.polylines(img, [pts.astype(np.int32)], True, (0,255,0), 2)
                cv2.line(img, tuple(pts[0].astype(int)), tuple(pts[3].astype(int)), (0,0,255), 3)
                cv2.arrowedLine(img, tuple(self.marker_pos.astype(int)),
                                (int(self.marker_pos[0]+45*math.cos(self.heading_angle)),
                                 int(self.marker_pos[1]+45*math.sin(self.heading_angle))),
                                (255,255,255), 2)

            cv2.putText(img, "L-Click: Add Obstacle | R-Click: Remove | c: Clear | q: Quit",
                        (10, 30), 0, 0.5, (200,200,200), 1)

            cv2.imshow(self.win_name, img)
            cv2.imshow("Monitor", np.hstack([cv2.resize(mon1,(640,360)), cv2.resize(mon0,(640,360))]))
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.dynamic_obstacles.clear()
                self.update_path()
        
        self.cap0.release()
        self.cap1.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    CompactTracker().run()
