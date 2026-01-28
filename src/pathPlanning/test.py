import cv2
import numpy as np
import math
import heapq

class FinalIntegratedTracker:
    def __init__(self):
        # 1. 물리 및 지도 설정 (사용자 수치 반영)
        self.marker_size, self.marker_h = 25.0, 72.0        
        self.map_w, self.map_h = 1000, 1000 
        self.grid_w, self.grid_h = 600, 720 
        self.map_scale = 0.5 
        self.off_x, self.off_y = 200, 150
        self.wc_w, self.wc_l = 57.0, 100.0           
        
        # 2. 상태 및 카메라 설정
        self.marker_pos, self.heading_angle, self.is_initialized = None, 0.0, False
        self.parking_mode = True
        self.path = []
        
        self.cams = {
            'cam1': {'pos': np.array([200.0 + self.off_x, 270.0 + self.off_y]), 'h': 110.0, 'focal': 841.0, 'map_angle': 157, 'yaw': 1.0, 'fov': 45, 'color': (255, 120, 100), 'name': 'Left'},
            'cam0': {'pos': np.array([301.4 + self.off_x, 540.0 + self.off_y]), 'h': 105.0, 'focal': 836.0, 'map_angle': 90, 'yaw': 1.0, 'fov': 45, 'color': (100, 120, 255), 'name': 'Rear'}
        }
        self.dist_gain, self.angle_gain, self.alpha = 1.03, 1.56, 0.75

        # 3. 목표 지점 설정 (좌표)
        self.goals = {
            'parking': {'pos': (300 + self.off_x, 600 + self.off_y), 'ang': math.radians(90)},
            'exit': {'pos': (150 + self.off_x, 360 + self.off_y), 'ang': math.radians(157)}
        }

        # 4. 파일 및 탐지기 초기화
        self.cap0 = cv2.VideoCapture('../wheelchairdetect/rear.mp4')
        self.cap1 = cv2.VideoCapture('../wheelchairdetect/left.mp4')
        self.detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250), cv2.aruco.DetectorParameters())

        self.win_name = "Integrated Wheelchair Tracker"
        cv2.namedWindow(self.win_name)
        self.setup_trackbars()

    def setup_trackbars(self):
        cv2.createTrackbar("Frame", self.win_name, 0, 1000, self.on_frame_change)
        cv2.createTrackbar("Mode: P(1)/E(0)", self.win_name, 1, 1, self.on_mode_change)
        cv2.createTrackbar("L_Focal", self.win_name, 841, 1500, lambda v: self.upd('cam1','focal',v))
        cv2.createTrackbar("L_Yaw", self.win_name, 91, 180, lambda v: self.upd('cam1','yaw',v-90))
        cv2.createTrackbar("R_Focal", self.win_name, 836, 1500, lambda v: self.upd('cam0','focal',v))
        cv2.createTrackbar("R_Yaw", self.win_name, 91, 180, lambda v: self.upd('cam0','yaw',v-90))

    def on_frame_change(self, v):
        self.cap0.set(cv2.CAP_PROP_POS_FRAMES, v)
        self.cap1.set(cv2.CAP_PROP_POS_FRAMES, v)

    def on_mode_change(self, v): 
        self.parking_mode = (v == 1)
        self.path = []

    def upd(self, side, key, val): self.cams[side][key] = float(val)

    def is_collision(self, x, y):
        # 자동차 금지 구역 및 맵 경계 체크
        gx, gy = x - self.off_x, y - self.off_y
        if 200 <= gx <= 400 and 180 <= gy <= 540: return True
        if not (self.off_x <= x <= self.off_x + self.grid_w and self.off_y <= y <= self.off_y + self.grid_h): return True
        return False

    def astar_plan(self, start, goal):
        # 최단 거리 및 장애물 회피를 위한 A* 알고리즘
        open_list = []
        heapq.heappush(open_list, (0, tuple(start)))
        came_from = {}
        g_score = {tuple(start): 0}
        
        while open_list:
            _, current = heapq.heappop(open_list)
            if math.dist(current, goal) < 20:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                return path[::-1]

            for dx, dy in [(0, 15), (0, -15), (15, 0), (-15, 0), (10, 10), (10, -10)]:
                neighbor = (current[0] + dx, current[1] + dy)
                if self.is_collision(neighbor[0], neighbor[1]): continue
                
                tentative_g = g_score[current] + math.dist(current, neighbor)
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + math.dist(neighbor, goal)
                    heapq.heappush(open_list, (f_score, neighbor))
        return []

    def draw_static_ui(self, img):
        # 자동차 영역 (진입 금지)
        cv2.rectangle(img, (200 + self.off_x, 180 + self.off_y), (400 + self.off_x, 540 + self.off_y), (35, 35, 45), -1)
        # 전체 그리드 테두리
        cv2.rectangle(img, (self.off_x, self.off_y), (self.off_x+self.grid_w, self.off_y+self.grid_h), (180, 180, 180), 2)
        
        # 목표 지점 시각화
        target = self.goals['parking'] if self.parking_mode else self.goals['exit']
        tp = tuple(map(int, target['pos']))
        cv2.circle(img, tp, 12, (0, 255, 255), -1)
        cv2.arrowedLine(img, tp, (int(tp[0]+45*math.cos(target['ang'])), int(tp[1]+45*math.sin(target['ang']))), (0, 255, 255), 2)
        
        status_txt = "PARKING MODE" if self.parking_mode else "EXIT MODE"
        cv2.putText(img, status_txt, (self.off_x, self.off_y - 20), 0, 0.7, (0, 255, 255), 2)

    def run(self):
        play = False
        while True:
            ret0, f0 = self.cap0.read(); ret1, f1 = self.cap1.read()
            if not ret0 or not ret1: break

            m_map = np.ones((self.map_h, self.map_w, 3), dtype=np.uint8) * 15
            self.draw_static_ui(m_map)
            
            # 모니터 영상용 복사본
            mon0, mon1 = f0.copy(), f1.copy()
            detected_data = []

            for frame, mon_frame, side in [(f0, mon0, 'cam0'), (f1, mon1, 'cam1')]:
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
                    
                    h = t_rad + math.atan2(c[0][1]-c[3][1], c[0][0]-c[3][0]) - (math.pi/2)
                    if ids[0][0] == 1: h += math.pi 
                    detected_data.append((raw_pos, h))

                    # 맵에 감지 광선 그리기
                    cv2.line(m_map, tuple(cfg['pos'].astype(int)), tuple(raw_pos.astype(int)), cfg['color'], 1)

            # 데이터 통합
            if len(detected_data) > 0:
                avg_pos = np.mean([p[0] for p in detected_data], axis=0)
                avg_h = math.atan2(np.mean([math.sin(p[1]) for p in detected_data]), np.mean([math.cos(p[1]) for p in detected_data]))
                
                if self.is_initialized:
                    self.marker_pos = self.marker_pos * (1 - self.alpha) + avg_pos * self.alpha
                    self.heading_angle = avg_h
                else:
                    self.marker_pos, self.heading_angle = avg_pos, avg_h
                    self.is_initialized = True

            # 휠체어 및 경로 표시
            if self.is_initialized:
                center_pos = self.marker_pos + np.array([(self.wc_l/2)*self.map_scale*math.cos(self.heading_angle), (self.wc_l/2)*self.map_scale*math.sin(self.heading_angle)])
                
                # 경로 계획 (실시간)
                target_pos = self.goals['parking']['pos'] if self.parking_mode else self.goals['exit']['pos']
                self.path = self.astar_plan(center_pos, target_pos)
                if self.path:
                    cv2.polylines(m_map, [np.array(self.path, np.int32)], False, (0, 255, 255), 2)

                # 휠체어 본체 그리기
                w_px, l_px = (self.wc_w * self.map_scale)/2, (self.wc_l * self.map_scale)/2
                base = np.array([[-l_px, -w_px], [l_px, -w_px], [l_px, w_px], [-l_px, w_px]])
                rot_m = np.array([[math.cos(self.heading_angle), -math.sin(self.heading_angle)], [math.sin(self.heading_angle), math.cos(self.heading_angle)]])
                rotated = np.dot(base, rot_m.T) + center_pos
                cv2.polylines(m_map, [rotated.astype(np.int32)], True, (0, 255, 0), 2)
                cv2.arrowedLine(m_map, tuple(center_pos.astype(int)), (int(center_pos[0]+50*math.cos(self.heading_angle)), int(center_pos[1]+50*math.sin(self.heading_angle))), (255, 255, 255), 2)

            cv2.imshow(self.win_name, m_map)
            cv2.imshow("Monitor", np.hstack([cv2.resize(mon0, (640, 360)), cv2.resize(mon1, (640, 360))]))
            
            key = cv2.waitKey(30) & 0xFF
            if key == ord(' '): play = not play
            elif key == ord('q'): break

        self.cap0.release(); self.cap1.release(); cv2.destroyAllWindows()

if __name__ == '__main__':
    FinalIntegratedTracker().run()