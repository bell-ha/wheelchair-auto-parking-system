import cv2
import numpy as np
import math
import heapq

class FinalOptimizedTracker:
    def __init__(self):
        # 1. 물리 및 지도 설정
        self.marker_size = 25.0
        self.marker_h = 72.0        
        self.map_w, self.map_h = 1000, 1000 
        self.grid_w, self.grid_h = 600, 720 
        self.map_scale = 0.5 
        self.off_x, self.off_y = 200, 150

        self.wc_w, self.wc_l = 57.0, 100.0           
        
        self.marker_pos = None     
        self.heading_angle = 0.0   
        self.is_initialized = False 

        self.cap0 = cv2.VideoCapture('../wheelchairdetect/rear.mp4')
        self.cap1 = cv2.VideoCapture('../wheelchairdetect/left.mp4')
        self.total_frames = int(min(self.cap0.get(cv2.CAP_PROP_FRAME_COUNT), 
                                    self.cap1.get(cv2.CAP_PROP_FRAME_COUNT)))
        
        self.curr_f0 = None
        self.curr_f1 = None

        self.cams = {
            'cam1': { 
                'pos': np.array([200.0 + self.off_x, 270.0 + self.off_y]), 
                'h': 110.0, 'focal': 841.0, 'map_angle': 157, 
                'yaw': 1.0, 'fov': 45, 'color': (255, 120, 100), 'name': 'Left'
            },
            'cam0': { 
                'pos': np.array([301.4 + self.off_x, 540.0 + self.off_y]), 
                'h': 105.0, 'focal': 836.0, 'map_angle': 90, 
                'yaw': 1.0, 'fov': 45, 'color': (100, 120, 255), 'name': 'Rear'
            }
        }
        
        self.dist_gain = 1.03      
        self.angle_gain = 1.56     
        self.alpha = 0.75          

        self.detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250),
            cv2.aruco.DetectorParameters()
        )

        # A* 경로 계획 관련 변수 (원본 코드와 동일하게 설정)
        self.car_dim = [115.0, 200.0]  # 57cm * 2 (map_scale 0.5)
        self.ramp_dim = [115.0, 80.0]  # 40cm * 2
        self.wheelchair_size_cm = 40.0  # 80cm / 2
        
        # 차량 후방 목표점 설정
        car_center_y = self.off_y + self.grid_h / 2
        car_rear_y = car_center_y + self.car_dim[1] / 2 + self.ramp_dim[1] + 75  # 램프 뒤 1.5m
        self.goal_positions = [
            (self.off_x + self.grid_w / 2 + 100, int(car_rear_y)),  # 목표 1 (우측)
            (self.off_x + self.grid_w / 2 - 100, int(car_rear_y))   # 목표 2 (좌측)
        ]
        self.current_goal_idx = 0
        self.path = []
        self.planning_enabled = True  # 기본값을 True로 변경

        self.win_name = "Integrated Wheelchair Tracker"
        cv2.namedWindow(self.win_name)
        
        cv2.createTrackbar("Frame", self.win_name, 278, self.total_frames - 1, self.on_frame_change)
        cv2.createTrackbar("L_Focal", self.win_name, 841, 1500, lambda v: self.upd('cam1','focal',v))
        cv2.createTrackbar("L_Yaw", self.win_name, 91, 180, lambda v: self.upd('cam1','yaw',v-90))
        cv2.createTrackbar("R_Focal", self.win_name, 836, 1500, lambda v: self.upd('cam0','focal',v))
        cv2.createTrackbar("R_Yaw", self.win_name, 91, 180, lambda v: self.upd('cam0','yaw',v-90))
        cv2.createTrackbar("Dist_Gain", self.win_name, 103, 200, self.on_dist_gain)
        cv2.createTrackbar("Smooth", self.win_name, 75, 100, self.on_alpha)
        cv2.createTrackbar("Plan", self.win_name, 1, 1, self.on_plan_toggle)  # 기본값 1로

        self.on_frame_change(278)

    def on_frame_change(self, v):
        self.cap0.set(cv2.CAP_PROP_POS_FRAMES, v)
        self.cap1.set(cv2.CAP_PROP_POS_FRAMES, v)
        _, self.curr_f0 = self.cap0.read()
        _, self.curr_f1 = self.cap1.read()

    def on_alpha(self, v): self.alpha = max(0.01, v / 100.0)
    def on_dist_gain(self, v): self.dist_gain = v / 100.0
    def upd(self, side, key, val): self.cams[side][key] = float(val)
    
    def on_plan_toggle(self, v):
        self.planning_enabled = (v == 1)
        if self.planning_enabled and self.marker_pos is not None:
            self.update_path()
        else:
            self.path = []

    def is_obstacle(self, px, py):
        """장애물 체크 (원본 코드와 동일한 좌표계)"""
        # 픽셀 좌표를 그리드 좌표로 변환
        wx = px - self.off_x
        wy = py - self.off_y
        
        margin = self.wheelchair_size_cm + 10.0
        
        # 차량 충돌 체크 (200~400, 180~540)
        if (200 - margin) <= wx <= (400 + margin) and \
           (180 - margin) <= wy <= (540 + margin):
            return True
        
        return False

    def simplify_path(self, path, epsilon=3.0):
        """Douglas-Peucker 알고리즘으로 경로 단순화"""
        if len(path) < 3: 
            return path
        
        def get_distance(p, a, b):
            if np.array_equal(a, b): 
                return np.linalg.norm(p - a)
            return np.abs(np.cross(b - a, a - p)) / np.linalg.norm(b - a)
        
        dmax, index = 0, 0
        for i in range(1, len(path) - 1):
            d = get_distance(np.array(path[i]), np.array(path[0]), np.array(path[-1]))
            if d > dmax: 
                index, dmax = i, d
        
        if dmax > epsilon:
            left = self.simplify_path(path[:index+1], epsilon)
            right = self.simplify_path(path[index:], epsilon)
            return left[:-1] + right
        
        return [path[0], path[-1]]

    def astar_plan(self, start, goal):
        """A* 경로 계획 알고리즘"""
        def heuristic(a, b):
            return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2) * 1.8

        start_node = (int(start[0]), int(start[1]))
        goal_node = (int(goal[0]), int(goal[1]))

        if self.is_obstacle(start_node[0], start_node[1]): 
            return []

        open_list = []
        heapq.heappush(open_list, (0, start_node, (0, 0)))
        came_from = {}
        g_score = {start_node: 0}
        
        ROTATION_PENALTY = 50.0
        BACKWARD_PENALTY = 500.0

        while open_list:
            _, current, prev_dir = heapq.heappop(open_list)

            if math.sqrt((current[0]-goal_node[0])**2 + (current[1]-goal_node[1])**2) < 15:
                raw_path = []
                while current in came_from:
                    raw_path.append([current[0], current[1]])
                    current = came_from[current]
                raw_path.reverse()
                return self.simplify_path(raw_path, epsilon=5.0)

            for dx, dy in [(0,3),(0,-3),(3,0),(-3,0),(2,2),(2,-2),(-2,2),(-2,-2)]:
                neighbor = (current[0] + dx, current[1] + dy)
                
                if not (self.off_x <= neighbor[0] < self.off_x + self.grid_w and 
                        self.off_y <= neighbor[1] < self.off_y + self.grid_h): 
                    continue
                if self.is_obstacle(neighbor[0], neighbor[1]): 
                    continue

                move_dist = math.sqrt(dx**2 + dy**2)
                
                penalty = 0.0
                if prev_dir != (0, 0):
                    if prev_dir != (dx, dy):
                        penalty = ROTATION_PENALTY
                        dot_product = prev_dir[0]*dx + prev_dir[1]*dy
                        if dot_product < 0:
                            penalty += BACKWARD_PENALTY
                        elif dot_product == 0:
                            penalty += ROTATION_PENALTY
                
                tentative_g_score = g_score[current] + move_dist + penalty

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score = tentative_g_score + heuristic(neighbor, goal_node)
                    heapq.heappush(open_list, (f_score, neighbor, (dx, dy)))
        
        return []

    def select_nearest_goal(self):
        """현재 위치에서 가장 가까운 목표 자동 선택"""
        if self.marker_pos is None or not self.is_initialized:
            return
        
        # 휠체어 중심 위치 계산
        offset_dist = (self.wc_l / 2) * self.map_scale
        center_pos = self.marker_pos + np.array([
            offset_dist * math.cos(self.heading_angle), 
            offset_dist * math.sin(self.heading_angle)
        ])
        
        min_dist = float('inf')
        nearest_idx = 0
        
        for i, goal in enumerate(self.goal_positions):
            dist = math.sqrt((goal[0] - center_pos[0])**2 + (goal[1] - center_pos[1])**2)
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i
        
        if nearest_idx != self.current_goal_idx:
            self.current_goal_idx = nearest_idx
            cv2.setTrackbarPos("Goal", self.win_name, self.current_goal_idx)
            print(f"🎯 목표 자동 변경: G{self.current_goal_idx + 1} (거리: {min_dist:.1f}px)")

    def update_path(self):
        """경로 업데이트"""
        if self.marker_pos is None or not self.is_initialized:
            return
        
        # 휠체어 중심 위치 계산
        offset_dist = (self.wc_l / 2) * self.map_scale
        center_pos = self.marker_pos + np.array([
            offset_dist * math.cos(self.heading_angle), 
            offset_dist * math.sin(self.heading_angle)
        ])
        
        start = (int(center_pos[0]), int(center_pos[1]))
        goal = self.goal_positions[self.current_goal_idx]
        
        new_path = self.astar_plan(start, goal)
        if new_path:
            self.path = new_path
            print(f"✅ 경로 계획 완료: {len(self.path)} waypoints")
        else:
            print("⚠️ 경로를 찾을 수 없습니다")

    def draw_static_map(self, img):
        step = int(20 * self.map_scale * 2) 
        for x in range(0, self.grid_w + 1, step):
            c = (45, 45, 45) if x % 100 != 0 else (80, 80, 80)
            cv2.line(img, (self.off_x + x, self.off_y), (self.off_x + x, self.off_y + self.grid_h), c, 1)
        for y in range(0, self.grid_h + 1, step):
            c = (45, 45, 45) if y % 100 != 0 else (80, 80, 80)
            cv2.line(img, (self.off_x, self.off_y + y), (self.off_x + self.grid_w, self.off_y + y), c, 1)
        
        # 차량 영역 (원본 코드와 동일)
        cv2.rectangle(img, (200 + self.off_x, 180 + self.off_y), 
                     (400 + self.off_x, 540 + self.off_y), 
                     (35, 35, 45), -1)
        
        # 램프 영역은 그리지 않음 (원본 코드와 동일)
        
        cv2.rectangle(img, (self.off_x, self.off_y), 
                     (self.off_x+self.grid_w, self.off_y+self.grid_h), 
                     (180, 180, 180), 2)
        
        # 카메라 위치
        for side in self.cams:
            cfg = self.cams[side]
            cp = tuple(cfg['pos'].astype(int))
            cv2.circle(img, cp, 7, cfg['color'], -1)
            cv2.putText(img, cfg['name'], (cp[0]-25, cp[1]+25), 
                       0, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        
        # 목표 지점들
        for i, goal in enumerate(self.goal_positions):
            color = (0, 255, 0) if i == self.current_goal_idx else (100, 100, 100)
            goal_pt = (int(goal[0]), int(goal[1]))
            cv2.circle(img, goal_pt, 10, color, -1)
            cv2.circle(img, goal_pt, 12, (255, 255, 255), 2)
            cv2.putText(img, f"G{i+1}", (goal_pt[0]-10, goal_pt[1]-20), 
                       0, 0.5, color, 2, cv2.LINE_AA)

    def draw_path(self, img):
        """계획된 경로 시각화"""
        if len(self.path) < 2:
            return
        
        # 경로 선 그리기
        for i in range(len(self.path) - 1):
            p1 = tuple(self.path[i])
            p2 = tuple(self.path[i+1])
            cv2.line(img, p1, p2, (0, 255, 255), 2, cv2.LINE_AA)
        
        # 웨이포인트 표시
        for i, point in enumerate(self.path):
            if i == 0:
                cv2.circle(img, tuple(point), 5, (255, 0, 255), -1)  # 시작점
            elif i == len(self.path) - 1:
                cv2.circle(img, tuple(point), 5, (0, 255, 0), -1)  # 끝점
            else:
                cv2.circle(img, tuple(point), 3, (0, 255, 255), -1)  # 중간점
    
    def draw_path_following_info(self, img, center_pos):
        """경로 추종 정보 시각화 (회전 각도 포함)"""
        if not self.path or not self.is_initialized:
            return
        
        # 다음 목표 지점 (첫 번째 웨이포인트)
        target_point = self.path[0]
        
        # 현재 위치에서 목표까지의 방향
        dx = target_point[0] - center_pos[0]
        dy = target_point[1] - center_pos[1]
        target_yaw = math.atan2(dy, dx)
        
        # 현재 헤딩과의 각도 차이 계산
        yaw_error = math.atan2(
            math.sin(target_yaw - self.heading_angle), 
            math.cos(target_yaw - self.heading_angle)
        )
        yaw_error_deg = math.degrees(yaw_error)
        
        # 회전 각도 호(arc) 그리기
        radius = 45
        start_angle = -math.degrees(self.heading_angle)
        end_angle = -math.degrees(target_yaw)
        
        # 회전 방향에 따라 색상 결정
        arc_color = (0, 200, 255) if yaw_error > 0 else (255, 150, 0)
        
        # 각도 호 그리기
        cv2.ellipse(img, (int(center_pos[0]), int(center_pos[1])), 
                   (radius, radius), 0, start_angle, end_angle, 
                   arc_color, 2, cv2.LINE_AA)
        
        # 회전 각도 텍스트 표시
        wx, wy = int(center_pos[0]), int(center_pos[1])
        rotation_text = f"Rotate: {yaw_error_deg:+.1f}deg"
        direction_text = "CCW" if yaw_error > 0 else "CW"
        
        # 배경 박스
        text_size = cv2.getTextSize(rotation_text, 0, 0.5, 1)[0]
        cv2.rectangle(img, (wx + 30, wy - 75), 
                     (wx + 35 + text_size[0], wy - 45), 
                     (0, 0, 0), -1)
        
        # 텍스트
        cv2.putText(img, rotation_text, (wx + 32, wy - 58), 
                   0, 0.5, arc_color, 1, cv2.LINE_AA)
        cv2.putText(img, f"({direction_text})", (wx + 32, wy - 48), 
                   0, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
        
        # 다음 웨이포인트까지 거리
        dist_to_target = math.sqrt(dx**2 + dy**2)
        cv2.putText(img, f"Dist: {dist_to_target:.1f}px", (wx + 32, wy - 85), 
                   0, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

    def run(self):
        play = False
        while True:
            if play:
                ret0, self.curr_f0 = self.cap0.read()
                ret1, self.curr_f1 = self.cap1.read()
                if not ret0 or not ret1:
                    self.on_frame_change(0); continue
                cv2.setTrackbarPos("Frame", self.win_name, int(self.cap0.get(cv2.CAP_PROP_POS_FRAMES)))

            m_map = np.ones((self.map_h, self.map_w, 3), dtype=np.uint8) * 15
            self.draw_static_map(m_map)
            
            detected_data = []
            mon0 = self.curr_f0.copy() if self.curr_f0 is not None else None
            mon1 = self.curr_f1.copy() if self.curr_f1 is not None else None

            for frame, mon_frame, side in [(self.curr_f0, mon0, 'cam0'), (self.curr_f1, mon1, 'cam1')]:
                if frame is None: continue
                corners, ids, _ = self.detector.detectMarkers(frame)
                if ids is not None:
                    cv2.aruco.drawDetectedMarkers(mon_frame, corners, ids)
                    cfg = self.cams[side]
                    c = corners[0].reshape(4, 2)
                    px_h = (np.linalg.norm(c[0]-c[3]) + np.linalg.norm(c[1]-c[2])) / 2.0
                    raw_dist = (self.marker_size * cfg['focal']) / px_h
                    corr_dist = raw_dist * (1 + (self.dist_gain - 1) * (raw_dist / 500)) 
                    d = math.sqrt(max(0, corr_dist**2 - abs(cfg['h'] - self.marker_h)**2))
                    
                    rel_x = (np.mean(c[:, 0]) - frame.shape[1]/2) / (frame.shape[1]/2)
                    m_yaw_deg = (rel_x * cfg['fov']) * self.angle_gain
                    
                    t_rad = math.radians(cfg['map_angle'] + cfg['yaw'] + m_yaw_deg)
                    raw_pos = cfg['pos'] + np.array([d * self.map_scale * math.cos(t_rad), d * self.map_scale * math.sin(t_rad)])
                    
                    marker_vec = c[0] - c[3]
                    h = t_rad + math.atan2(marker_vec[1], marker_vec[0]) - (math.pi/2)
                    if ids[0][0] == 1: h += math.pi 
                    
                    detected_data.append((raw_pos, h, d, t_rad))

                    cp, rp = tuple(cfg['pos'].astype(int)), tuple(raw_pos.astype(int))
                    dist_px = int(d * self.map_scale)
                    cv2.ellipse(m_map, cp, (dist_px, dist_px), 0, math.degrees(t_rad)-5, math.degrees(t_rad)+5, cfg['color'], 2, cv2.LINE_AA)
                    cv2.line(m_map, cp, rp, cfg['color'], 1, cv2.LINE_AA)
                    
                    txt_pos = ((cp[0]+rp[0])//2, (cp[1]+rp[1])//2)
                    cv2.putText(m_map, f"{d:.0f}cm / {m_yaw_deg:+.1f}deg", (txt_pos[0]+5, txt_pos[1]-5), 0, 0.4, (180, 180, 180), 1, cv2.LINE_AA)

            if len(detected_data) > 0:
                avg_pos = np.mean([p[0] for p in detected_data], axis=0)
                avg_h = math.atan2(np.mean([math.sin(p[1]) for p in detected_data]), np.mean([math.cos(p[1]) for p in detected_data]))
                
                if play and self.is_initialized:
                    self.marker_pos = self.marker_pos * (1 - self.alpha) + avg_pos * self.alpha
                    self.heading_angle = math.atan2(math.sin(self.heading_angle)*(1-self.alpha) + math.sin(avg_h)*self.alpha, 
                                                    math.cos(self.heading_angle)*(1-self.alpha) + math.cos(avg_h)*self.alpha)
                else:
                    self.marker_pos, self.heading_angle = avg_pos, avg_h
                    self.is_initialized = True
                
                # 경로 업데이트
                if self.planning_enabled:
                    self.select_nearest_goal()  # 가장 가까운 목표 자동 선택
                    self.update_path()

            # 경로 그리기
            if self.planning_enabled:
                self.draw_path(m_map)

            if self.is_initialized:
                offset_dist = (self.wc_l / 2) * self.map_scale
                center_pos = self.marker_pos + np.array([offset_dist * math.cos(self.heading_angle), offset_dist * math.sin(self.heading_angle)])
                w_px, l_px = (self.wc_w * self.map_scale) / 2, (self.wc_l * self.map_scale) / 2
                base_pts = np.array([[-l_px, -w_px], [l_px, -w_px], [l_px, w_px], [-l_px, w_px]])
                rot_m = np.array([[math.cos(self.heading_angle), -math.sin(self.heading_angle)], [math.sin(self.heading_angle), math.cos(self.heading_angle)]])
                rotated_pts = np.dot(base_pts, rot_m.T) + center_pos
                cv2.polylines(m_map, [rotated_pts.astype(np.int32)], True, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.line(m_map, tuple(rotated_pts[0].astype(int)), tuple(rotated_pts[3].astype(int)), (0, 0, 255), 3)
                
                wx, wy = int(center_pos[0]), int(center_pos[1])
                cv2.putText(m_map, f"GRID:({wx-self.off_x},{wy-self.off_y})", (wx+25, wy-25), 0, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.arrowedLine(m_map, tuple(center_pos.astype(int)), (int(center_pos[0] + 45 * math.cos(self.heading_angle)), int(center_pos[1] + 45 * math.sin(self.heading_angle))), (255, 255, 255), 2)
                
                # 경로 추종 정보 표시 (회전 각도)
                if self.planning_enabled:
                    self.draw_path_following_info(m_map, center_pos)
                
                # 경로 정보 표시
                if self.planning_enabled and self.path:
                    cv2.putText(m_map, f"Path: {len(self.path)} waypoints", (wx+25, wy-45), 
                               0, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow(self.win_name, m_map)
            cv2.imshow("Monitor", np.hstack([cv2.resize(mon0, (640, 360)), cv2.resize(mon1, (640, 360))]))
            
            key = cv2.waitKey(80) & 0xFF
            if key == ord(' '): play = not play
            elif key == ord('q'): break
            elif key == ord('p'):  # 'p' 키로 경로 계획 토글
                self.planning_enabled = not self.planning_enabled
                cv2.setTrackbarPos("Plan", self.win_name, 1 if self.planning_enabled else 0)

        self.cap0.release(); self.cap1.release(); cv2.destroyAllWindows()

if __name__ == '__main__':
    FinalOptimizedTracker().run()
