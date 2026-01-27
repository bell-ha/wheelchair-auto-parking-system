import cv2
import numpy as np
import math
import heapq

class FinalOptimizedTracker:
    def __init__(self):
        # 1. 물리 및 지도 설정
        self.marker_size, self.marker_h = 25.0, 72.0        
        self.map_w, self.map_h = 1000, 1000 
        self.grid_w, self.grid_h = 600, 720  
        self.map_scale = 0.5 
        self.off_x, self.off_y = 200, 150
        self.wc_w, self.wc_l = 57.0, 100.0           
        
        # 목적지 및 AREA 설정
        self.p1 = (150 + self.off_x, 360 + self.off_y) 
        self.p2 = (300 + self.off_x, 600 + self.off_y) 
        self.area_rect = (0 + self.off_x, 630 + self.off_y, 100 + self.off_x, 720 + self.off_y)
        self.area_goal = (int((self.area_rect[0] + self.area_rect[2])/2), int((self.area_rect[1] + self.area_rect[3])/2))

        self.col_p1, self.col_p2, self.col_area = (255, 120, 0), (0, 150, 255), (0, 255, 255)
        self.marker_pos, self.heading_angle, self.is_initialized = None, 0.0, False 

        # 영상 소스 설정
        self.cap0 = cv2.VideoCapture('../wheelchairdetect/rear.mp4')
        self.cap1 = cv2.VideoCapture('../wheelchairdetect/left.mp4')
        self.total_frames = int(min(self.cap0.get(cv2.CAP_PROP_FRAME_COUNT), self.cap1.get(cv2.CAP_PROP_FRAME_COUNT)))
        
        self.cams = {
            'cam1': { 'pos': np.array([200.0 + self.off_x, 270.0 + self.off_y]), 'h': 110.0, 'focal': 841.0, 'map_angle': 157, 'yaw': 1.0, 'fov': 45, 'color': (255, 120, 100), 'name': 'Left' },
            'cam0': { 'pos': np.array([301.4 + self.off_x, 540.0 + self.off_y]), 'h': 105.0, 'focal': 836.0, 'map_angle': 90, 'yaw': 1.0, 'fov': 45, 'color': (100, 120, 255), 'name': 'Rear' }
        }
        
        self.dist_gain, self.angle_gain, self.alpha = 1.03, 1.56, 0.75
        self.detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250), cv2.aruco.DetectorParameters())

        self.win_name = "Integrated UI Wheelchair Tracker"
        cv2.namedWindow(self.win_name)
        cv2.createTrackbar("Frame", self.win_name, 278, self.total_frames - 1, self.on_frame_change)
        
        self.on_frame_change(278)

    def on_frame_change(self, v):
        self.cap0.set(cv2.CAP_PROP_POS_FRAMES, v); self.cap1.set(cv2.CAP_PROP_POS_FRAMES, v)
        _, self.curr_f0 = self.cap0.read(); _, self.curr_f1 = self.cap1.read()

    def is_obstacle(self, px, py):
        wx, wy = px - self.off_x, py - self.off_y
        margin = 15.0
        if (200 - margin) <= wx <= (400 + margin) and (180 - margin) <= wy <= (540 + margin): return True
        return False

    def can_see_goal(self, start, goal):
        steps = 20
        for i in range(steps + 1):
            t = i / steps
            if self.is_obstacle(start[0]*(1-t)+goal[0]*t, start[1]*(1-t)+goal[1]*t): return False
        return True

    def simplify_path(self, path, epsilon=12.0):
        if len(path) < 3: return path
        def get_dist(p, a, b):
            ap, ab = p - a, b - a
            if np.array_equal(a, b): return np.linalg.norm(ap)
            return np.abs(np.cross(ab, ap)) / np.linalg.norm(ab)
        dmax, idx = 0, 0
        for i in range(1, len(path) - 1):
            d = get_dist(np.array(path[i]), np.array(path[0]), np.array(path[-1]))
            if d > dmax: idx, dmax = i, d
        if dmax > epsilon:
            return self.simplify_path(path[:idx+1], epsilon)[:-1] + self.simplify_path(path[idx:], epsilon)
        return [path[0], path[-1]]

    def astar_plan(self, start, goal):
        sn, gn = (int(start[0]), int(start[1])), (int(goal[0]), int(goal[1]))
        if self.can_see_goal(sn, gn): return [sn, gn]
        open_l = []
        heapq.heappush(open_l, (0, sn))
        came, g_s = {}, {sn: 0}
        while open_l:
            _, curr = heapq.heappop(open_l)
            if math.dist(curr, gn) < 20:
                res = []
                while curr in came: res.append(curr); curr = came[curr]
                return self.simplify_path(res[::-1])
            for dx, dy in [(0,15),(0,-15),(15,0),(-15,0),(11,11),(11,-11),(-11,11),(-11,-11)]:
                nb = (curr[0]+dx, curr[1]+dy)
                if not (0 <= nb[0] < self.map_w and 0 <= nb[1] < self.map_h) or self.is_obstacle(*nb): continue
                tg = g_s[curr] + math.dist(curr, nb)
                if nb not in g_s or tg < g_s[nb]:
                    came[nb], g_s[nb] = curr, tg
                    heapq.heappush(open_l, (tg + math.dist(nb, gn), nb))
        return [sn, gn]

    def draw_static_map(self, img):
        for i in range(0, 1001, 50): cv2.line(img, (i, 0), (i, 1000), (20, 20, 20), 1)
        cv2.rectangle(img, (200 + self.off_x, 180 + self.off_y), (400 + self.off_x, 540 + self.off_y), (30, 30, 40), -1)
        cv2.rectangle(img, (self.off_x, self.off_y), (self.off_x+self.grid_w, self.off_y+self.grid_h), (150, 150, 150), 2)
        cv2.rectangle(img, (self.area_rect[0], self.area_rect[1]), (self.area_rect[2], self.area_rect[3]), (0,40,0), -1)
        cv2.circle(img, self.area_goal, 7, self.col_area, -1)
        cv2.circle(img, self.p1, 8, self.col_p1, -1); cv2.circle(img, self.p2, 8, self.col_p2, -1)

    # [수정] 각도 계산 및 UI 출력 함수
    def draw_path_ui(self, img, path, color, label, pivot, offset_y):
        if not path or len(path) < 2: return
        
        # 1. 경로 그리기 (완벽한 직선 혹은 우회 경로)
        cv2.polylines(img, [np.array(path, np.int32)], False, color, 2, cv2.LINE_AA)
        
        # 2. [핵심] 타겟 설정: 내 발등 앞(path[0])이 아닌, 최종 목적지 혹은 충분히 먼 지점(path[-1])
        target = path[-1] 
        
        # 3. 각도 계산 (빨간 선 중앙 'pivot' 기준)
        dx, dy = target[0] - pivot[0], target[1] - pivot[1]
        target_abs_angle = math.atan2(dy, dx)
        
        # 4. 상대 각도 에러 (정면 화살표 기준 얼마나 틀어야 하는가)
        diff = target_abs_angle - self.heading_angle
        yaw_err = math.degrees(math.atan2(math.sin(diff), math.cos(diff)))
        
        # 5. UI 가이드 그리기
        cv2.ellipse(img, (int(pivot[0]), int(pivot[1])), (45, 45), 0, 
                   -math.degrees(self.heading_angle), -math.degrees(target_abs_angle), color, 2, cv2.LINE_AA)
        cv2.putText(img, f"{label} Rot: {yaw_err:+.1f}d", (int(pivot[0])+60, int(pivot[1])+offset_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    def run(self):
        play = False
        while True:
            if play:
                ret0, self.curr_f0 = self.cap0.read(); ret1, self.curr_f1 = self.cap1.read()
                if not ret0 or not ret1: self.on_frame_change(0); continue
                cv2.setTrackbarPos("Frame", self.win_name, int(self.cap0.get(cv2.CAP_PROP_POS_FRAMES)))

            m_map = np.ones((self.map_h, self.map_w, 3), dtype=np.uint8) * 12
            self.draw_static_map(m_map)
            mon0, mon1 = (f.copy() if f is not None else np.zeros((360,640,3),np.uint8) for f in [self.curr_f0, self.curr_f1])
            
            detected_data = []
            is_cam0, is_cam1 = False, False

            for frame, mon, side in [(self.curr_f0, mon0, 'cam0'), (self.curr_f1, mon1, 'cam1')]:
                if frame is None: continue
                corners, ids, _ = self.detector.detectMarkers(frame)
                if ids is not None:
                    if side == 'cam0': is_cam0 = True
                    if side == 'cam1': is_cam1 = True
                    cfg = self.cams[side]
                    c = corners[0].reshape(4, 2)
                    px_h = (np.linalg.norm(c[0]-c[3]) + np.linalg.norm(c[1]-c[2])) / 2.0
                    raw_dist = (self.marker_size * cfg['focal']) / px_h
                    corr_dist = raw_dist * (1 + (self.dist_gain - 1) * (raw_dist / 500)) 
                    d = math.sqrt(max(0, corr_dist**2 - abs(cfg['h'] - self.marker_h)**2))
                    rel_x = (np.mean(c[:, 0]) - frame.shape[1]/2) / (frame.shape[1]/2)
                    t_rad = math.radians(cfg['map_angle'] + cfg['yaw'] + (rel_x * cfg['fov'] * self.angle_gain))
                    pos = cfg['pos'] + np.array([d * self.map_scale * math.cos(t_rad), d * self.map_scale * math.sin(t_rad)])
                    h = t_rad + math.atan2(c[0][1]-c[3][1], c[0][0]-c[3][0]) - (math.pi/2)
                    if ids[0][0] == 1: h += math.pi
                    detected_data.append((pos, h))

            if detected_data:
                ap = np.mean([p[0] for p in detected_data], axis=0)
                ah = math.atan2(np.mean([math.sin(p[1]) for p in detected_data]), np.mean([math.cos(p[1]) for p in detected_data]))
                if not self.is_initialized: self.marker_pos, self.heading_angle, self.is_initialized = ap, ah, True
                else:
                    self.marker_pos = self.marker_pos*(1-self.alpha) + ap*self.alpha
                    self.heading_angle = math.atan2(math.sin(self.heading_angle)*(1-self.alpha)+math.sin(ah)*self.alpha, math.cos(self.heading_angle)*(1-self.alpha)+math.cos(ah)*self.alpha)

            if self.is_initialized:
                # [기준점] 빨간 줄의 중앙 (뒷바퀴 축 중심)
                pivot = self.marker_pos
                # [전방 중심] 시각화를 위한 휠체어 몸체 중심
                center = pivot + np.array([(self.wc_l/2)*self.map_scale*math.cos(self.heading_angle), (self.wc_l/2)*self.map_scale*math.sin(self.heading_angle)])
                
                # 경로 표시 (pivot 기준 각도 계산)
                if is_cam1: self.draw_path_ui(m_map, self.astar_plan(center, self.p1), self.col_p1, "P1", pivot, -40)
                if is_cam0: self.draw_path_ui(m_map, self.astar_plan(center, self.p2), self.col_p2, "P2", pivot, -15)
                if is_cam0 or is_cam1: self.draw_path_ui(m_map, self.astar_plan(center, self.area_goal), self.col_area, "AREA", pivot, 10)

                # 휠체어 시각화
                w, l = (self.wc_w*self.map_scale)/2, (self.wc_l*self.map_scale)/2
                pts = np.dot(np.array([[-l,-w],[l,-w],[l,w],[-l,w]]), np.array([[math.cos(self.heading_angle),-math.sin(self.heading_angle)],[math.sin(self.heading_angle),math.cos(self.heading_angle)]]).T) + center
                cv2.polylines(m_map, [pts.astype(np.int32)], True, (0,255,0), 2, cv2.LINE_AA)
                cv2.line(m_map, tuple(pts[0].astype(int)), tuple(pts[3].astype(int)), (0,0,255), 3) # 뒷바퀴 빨간줄
                cv2.arrowedLine(m_map, tuple(pivot.astype(int)), (int(pivot[0] + 45*math.cos(self.heading_angle)), int(pivot[1] + 45*math.sin(self.heading_angle))), (255,255,255), 2)

            cv2.imshow(self.win_name, m_map)
            cv2.imshow("Monitor", np.hstack([cv2.resize(mon1, (640, 360)), cv2.resize(mon0, (640, 360))]))
            key = cv2.waitKey(30) & 0xFF
            if key == ord(' '): play = not play
            elif key == ord('q'): break

        self.cap0.release(); self.cap1.release(); cv2.destroyAllWindows()

if __name__ == '__main__':
    FinalOptimizedTracker().run()