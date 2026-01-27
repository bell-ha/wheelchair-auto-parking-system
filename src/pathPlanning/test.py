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

        # 영상 소스 및 카메라 설정 (이전 코드의 정밀 수치 반영)
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
        cv2.createTrackbar("Smooth", self.win_name, 75, 100, lambda v: setattr(self, 'alpha', max(0.01, v/100.0)))
        cv2.createTrackbar("Dist_Gain", self.win_name, 103, 200, lambda v: setattr(self, 'dist_gain', v/100.0))
        
        self.on_frame_change(278)

    def on_frame_change(self, v):
        self.cap0.set(cv2.CAP_PROP_POS_FRAMES, v); self.cap1.set(cv2.CAP_PROP_POS_FRAMES, v)
        _, self.curr_f0 = self.cap0.read(); _, self.curr_f1 = self.cap1.read()

    def simplify_path(self, path, epsilon=12.0): # [수정] epsilon 상향으로 직선화 강화
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

    def is_obstacle(self, px, py):
        wx, wy = px - self.off_x, py - self.off_y
        m = 45.0 # 장애물 마진
        if (200 - m) <= wx <= (400 + m) and (180 - m) <= wy <= (540 + m): return True
        return False

    def astar_plan(self, start, goal):
        sn, gn = (int(start[0]), int(start[1])), (int(goal[0]), int(goal[1]))
        if self.is_obstacle(*sn): return []
        open_l = []
        heapq.heappush(open_l, (0, sn))
        came, g_s = {}, {sn: 0}
        while open_l:
            _, curr = heapq.heappop(open_l)
            if math.dist(curr, gn) < 10: # [수정] 도착 판정 거리 단축 (점과 선을 붙임)
                res = []
                while curr in came: res.append(curr); curr = came[curr]
                return self.simplify_path(res[::-1])
            # 8방향 탐색
            for dx, dy in [(0,7),(0,-7),(7,0),(-7,0),(5,5),(5,-5),(-5,5),(-5,-5)]:
                nb = (curr[0]+dx, curr[1]+dy)
                if not (0 <= nb[0] < self.map_w and 0 <= nb[1] < self.map_h) or self.is_obstacle(*nb): continue
                tg = g_s[curr] + math.dist((0,0),(dx,dy))
                if nb not in g_s or tg < g_s[nb]:
                    came[nb], g_s[nb] = curr, tg
                    heapq.heappush(open_l, (tg + math.dist(nb, gn), nb))
        return []

    def draw_static_map(self, img):
        # [복구] 배경 격자 및 작업 구역 격자
        for i in range(0, 1001, 50):
            cv2.line(img, (i, 0), (i, 1000), (20, 20, 20), 1)
        step = int(20 * self.map_scale * 2) 
        for x in range(0, self.grid_w + 1, step):
            c = (40, 40, 40) if x % 100 != 0 else (70, 70, 70)
            cv2.line(img, (self.off_x + x, self.off_y), (self.off_x + x, self.off_y + self.grid_h), c, 1)
        for y in range(0, self.grid_h + 1, step):
            c = (40, 40, 40) if y % 100 != 0 else (70, 70, 70)
            cv2.line(img, (self.off_x, self.off_y + y), (self.off_x + self.grid_w, self.off_y + y), c, 1)
        
        cv2.rectangle(img, (200 + self.off_x, 180 + self.off_y), (400 + self.off_x, 540 + self.off_y), (30, 30, 40), -1)
        cv2.rectangle(img, (self.off_x, self.off_y), (self.off_x+self.grid_w, self.off_y+self.grid_h), (150, 150, 150), 2)
        
        for side, cfg in self.cams.items():
            cp = tuple(cfg['pos'].astype(int))
            cv2.circle(img, cp, 7, cfg['color'], -1)
            cv2.putText(img, cfg['name'], (cp[0]-20, cp[1]+20), 0, 0.4, (200, 200, 200), 1)

        cv2.rectangle(img, (self.area_rect[0], self.area_rect[1]), (self.area_rect[2], self.area_rect[3]), (0,40,0), -1)
        cv2.circle(img, self.area_goal, 7, self.col_area, -1)
        cv2.circle(img, self.p1, 8, self.col_p1, -1); cv2.circle(img, self.p2, 8, self.col_p2, -1)

    def draw_path_ui(self, img, path, color, label, center, offset_y):
        if not path or len(path) < 2: return
        cv2.polylines(img, [np.array(path, np.int32)], False, color, 2, cv2.LINE_AA)
        target = path[0]
        dx, dy = target[0] - center[0], target[1] - center[1]
        target_yaw = math.atan2(dy, dx)
        yaw_err = math.degrees(math.atan2(math.sin(target_yaw - self.heading_angle), math.cos(target_yaw - self.heading_angle)))
        cv2.ellipse(img, (int(center[0]), int(center[1])), (40, 40), 0, -math.degrees(self.heading_angle), -math.degrees(target_yaw), color, 2, cv2.LINE_AA)
        cv2.putText(img, f"{label} Rot: {yaw_err:+.1f}d", (int(center[0])+50, int(center[1])+offset_y), 0, 0.4, color, 1)

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
            # [수정] 카메라별 감지 여부 플래그
            is_cam0_detected = False # Rear
            is_cam1_detected = False # Left

            for frame, mon, side in [(self.curr_f0, mon0, 'cam0'), (self.curr_f1, mon1, 'cam1')]:
                if frame is None: continue
                corners, ids, _ = self.detector.detectMarkers(frame)
                if ids is not None:
                    # 감지 플래그 업데이트
                    if side == 'cam0': is_cam0_detected = True
                    if side == 'cam1': is_cam1_detected = True

                    cv2.aruco.drawDetectedMarkers(mon, corners, ids)
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
                    cv2.line(m_map, tuple(cfg['pos'].astype(int)), tuple(pos.astype(int)), cfg['color'], 1)

            if detected_data:
                ap = np.mean([p[0] for p in detected_data], axis=0)
                ah = math.atan2(np.mean([math.sin(p[1]) for p in detected_data]), np.mean([math.cos(p[1]) for p in detected_data]))
                if not self.is_initialized: self.marker_pos, self.heading_angle, self.is_initialized = ap, ah, True
                else:
                    self.marker_pos = self.marker_pos*(1-self.alpha) + ap*self.alpha
                    self.heading_angle = math.atan2(math.sin(self.heading_angle)*(1-self.alpha)+math.sin(ah)*self.alpha, math.cos(self.heading_angle)*(1-self.alpha)+math.cos(ah)*self.alpha)

            if self.is_initialized:
                center = self.marker_pos + np.array([(self.wc_l/2)*self.map_scale*math.cos(self.heading_angle), (self.wc_l/2)*self.map_scale*math.sin(self.heading_angle)])
                
                # [수정] 조건부 경로 탐색 및 그리기
                # Cam1(Left)이 보일 때만 P1(주황색) 경로 표시
                if is_cam1_detected:
                    self.draw_path_ui(m_map, self.astar_plan(center, self.p1), self.col_p1, "P1", center, -40)
                
                # Cam0(Rear)이 보일 때만 P2(파란색) 경로 표시
                if is_cam0_detected:
                    self.draw_path_ui(m_map, self.astar_plan(center, self.p2), self.col_p2, "P2", center, -15)
                
                # AREA는 둘 중 하나만 보여도 출력 (데이터가 업데이트 중이므로)
                if is_cam0_detected or is_cam1_detected:
                    self.draw_path_ui(m_map, self.astar_plan(center, self.area_goal), self.col_area, "AREA", center, 10)

                # 휠체어 시각화
                w, l = (self.wc_w*self.map_scale)/2, (self.wc_l*self.map_scale)/2
                pts = np.dot(np.array([[-l,-w],[l,-w],[l,w],[-l,w]]), np.array([[math.cos(self.heading_angle),-math.sin(self.heading_angle)],[math.sin(self.heading_angle),math.cos(self.heading_angle)]]).T) + center
                cv2.polylines(m_map, [pts.astype(np.int32)], True, (0,255,0), 2, cv2.LINE_AA)
                cv2.line(m_map, tuple(pts[0].astype(int)), tuple(pts[3].astype(int)), (0,0,255), 3) 
                cv2.arrowedLine(m_map, tuple(self.marker_pos.astype(int)), (int(self.marker_pos[0] + 45*math.cos(self.heading_angle)), int(self.marker_pos[1] + 45*math.sin(self.heading_angle))), (255,255,255), 2)

            cv2.imshow(self.win_name, m_map)
            cv2.imshow("Monitor", np.hstack([cv2.resize(mon1, (640, 360)), cv2.resize(mon0, (640, 360))]))
            key = cv2.waitKey(80) & 0xFF
            if key == ord(' '): play = not play
            elif key == ord('q'): break

        self.cap0.release(); self.cap1.release(); cv2.destroyAllWindows()

if __name__ == '__main__':
    FinalOptimizedTracker().run()