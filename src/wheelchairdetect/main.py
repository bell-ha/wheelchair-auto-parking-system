import cv2
import numpy as np
import math
import time

class UltimateWheelchairTracker:
    def __init__(self):
        # 1. 물리 및 지도 설정
        self.marker_size = 25.0
        self.marker_h = 40.0
        self.map_w, self.map_h = 1000, 1000 
        self.grid_w, self.grid_h = 600, 720
        self.map_scale = 0.5 
        self.off_x, self.off_y = 200, 150

        # --- [상태 및 분석 변수] ---
        self.box_size = 50
        self.alpha = 0.15          
        self.rel_pos = None        
        self.heading_angle = 0.0   
        
        self.path_history = []     # 이동 궤적 저장용
        # 30 FPS 기준 5초 = 150 프레임
        self.max_history = 150     

        # 고정 프리셋
        self.cams = {
            'left': {
                'pos': np.array([200.0 + self.off_x, 270.0 + self.off_y]), 
                'h': 115.0, 'focal': 797.0, 'map_angle': 135, 'yaw': 24.0, 
                'w_angle': 360.0, 'color': (0, 0, 255), 'name': 'L-Cam'
            },
            'rear': {
                'pos': np.array([301.4 + self.off_x, 540.0 + self.off_y]), 
                'h': 85.0, 'focal': 896.0, 'map_angle': 90, 'yaw': 1.0, 
                'w_angle': 360.0, 'color': (255, 0, 0), 'name': 'R-Cam'
            }
        }
        self.dist_gain = 1.19      
        self.angle_gain = 1.56     

        self.cap_left = cv2.VideoCapture('left.mp4')
        self.cap_rear = cv2.VideoCapture('rear.mp4')
        self.total_frames = int(self.cap_left.get(cv2.CAP_PROP_FRAME_COUNT))
        self.detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50))

        self.win_name = "5s Trail Wheelchair Tracker"
        cv2.namedWindow(self.win_name)
        cv2.createTrackbar("Progress", self.win_name, 0, self.total_frames - 1, self.on_frame)

        self.is_paused = False
        self.last_f = None

    def on_frame(self, v): 
        self.cap_left.set(cv2.CAP_PROP_POS_FRAMES, v)
        self.cap_rear.set(cv2.CAP_PROP_POS_FRAMES, v)
        self.path_history = [] # 재생바 이동 시 궤적 초기화

    def draw_grid(self, img):
        start_pt, end_pt = (self.off_x, self.off_y), (self.off_x + self.grid_w, self.off_y + self.grid_h)
        cv2.rectangle(img, start_pt, end_pt, (100, 100, 100), 2)
        for i in range(1, 3):
            x = self.off_x + (self.grid_w // 3) * i
            y = self.off_y + (self.grid_h // 3) * i
            cv2.line(img, (x, self.off_y), (x, self.off_y + self.grid_h), (60, 60, 60), 1)
            cv2.line(img, (self.off_x, y), (self.off_x + self.grid_w, y), (60, 60, 60), 1)
        
        for side, cfg in self.cams.items():
            cp = tuple(cfg['pos'].astype(int))
            cv2.circle(img, cp, 7, cfg['color'], -1)
            cv2.putText(img, cfg['name'], (cp[0]-25, cp[1]+25), 0, 0.4, (200, 200, 200), 1)

    def get_corrected_dist(self, corners, cfg):
        c = corners.reshape(4, 2)
        px_h = (np.linalg.norm(c[0]-c[3]) + np.linalg.norm(c[1]-c[2])) / 2.0
        if px_h < 2: return None
        raw_dist = (self.marker_size * cfg['focal']) / px_h
        corr_dist = raw_dist * (1 + (self.dist_gain - 1) * (raw_dist / 500)) 
        return math.sqrt(max(0, corr_dist**2 - abs(cfg['h'] - self.marker_h)**2))

    def draw_wheelchair_and_path(self, img, pos, angle):
        # 1. 5초 이동 궤적 (Trail)
        if len(self.path_history) > 2:
            pts = np.array(self.path_history, np.int32)
            # 궤적을 조금 더 눈에 띄게 초록색으로 표시
            cv2.polylines(img, [pts], False, (0, 200, 0), 2, cv2.LINE_AA)

        # 2. 휠체어 사각형 (Object)
        half = self.box_size / 2
        base_pts = np.array([[-half, -half], [half, -half], [half, half], [-half, half]])
        c, s = math.cos(angle), math.sin(angle)
        rot_m = np.array([[c, -s], [s, c]])
        rotated_pts = np.dot(base_pts, rot_m.T) + pos
        
        cv2.polylines(img, [rotated_pts.astype(np.int32)], True, (0, 255, 0), 2)
        back_pts = np.array([[-half, -half], [-half, half]])
        rot_back = np.dot(back_pts, rot_m.T) + pos
        cv2.line(img, tuple(rot_back[0].astype(int)), tuple(rot_back[1].astype(int)), (0, 0, 255), 4)

        # 3. 정면 화살표
        front_pt = pos + np.array([40 * math.cos(angle), 40 * math.sin(angle)])
        cv2.arrowedLine(img, tuple(pos.astype(int)), tuple(front_pt.astype(int)), (255, 255, 255), 2, tipLength=0.3)

    def run(self):
        while True:
            if not self.is_paused:
                ret_l, f_l = self.cap_left.read(); ret_r, f_r = self.cap_rear.read()
                if not ret_l or not ret_r: break
                self.last_f = (f_l.copy(), f_r.copy())
                cv2.setTrackbarPos("Progress", self.win_name, int(self.cap_left.get(cv2.CAP_PROP_POS_FRAMES)))
            else:
                if self.last_f: f_l, f_r = self.last_f
                else: continue

            m_map = np.ones((self.map_h, self.map_w, 3), dtype=np.uint8) * 15
            self.draw_grid(m_map)
            cv2.rectangle(m_map, (200 + self.off_x, 180 + self.off_y), (400 + self.off_x, 540 + self.off_y), (35, 35, 35), -1)

            detected_data = []
            for frame, side in [(f_l, 'left'), (f_r, 'rear')]:
                corners, ids, _ = self.detector.detectMarkers(frame)
                if ids is not None:
                    cfg = self.cams[side]
                    d = self.get_corrected_dist(corners[0], cfg)
                    if d:
                        rel_x = (np.mean(corners[0].reshape(4,2)[:, 0]) - frame.shape[1]/2) / (frame.shape[1]/2)
                        angle_offset = (rel_x * math.radians(45)) * self.angle_gain
                        t_rad = math.radians(cfg['map_angle'] + cfg['yaw']) + angle_offset
                        raw_pos = cfg['pos'] + np.array([d * self.map_scale * math.cos(t_rad), d * self.map_scale * math.sin(t_rad)])
                        
                        c = corners[0].reshape(4, 2)
                        vec = c[0] - c[3]
                        h = t_rad + math.atan2(vec[1], vec[0]) - (math.pi / 2) + math.radians(cfg['w_angle'])
                        if ids[0][0] == 1: h += math.pi
                        
                        detected_data.append((raw_pos, h, d))
                        
                        cp, rp = tuple(cfg['pos'].astype(int)), tuple(raw_pos.astype(int))
                        cv2.line(m_map, cp, rp, cfg['color'], 1, cv2.LINE_AA)
                        cv2.putText(m_map, f"{d:.0f}cm", ((cp[0]+rp[0])//2, (cp[1]+rp[1])//2), 0, 0.4, (150, 150, 150), 1)
                        cv2.circle(m_map, rp, 4, cfg['color'], -1)

            if len(detected_data) > 0:
                new_pos = np.mean([p[0] for p in detected_data], axis=0)
                new_h = math.atan2(np.mean([math.sin(p[1]) for p in detected_data]), 
                                   np.mean([math.cos(p[1]) for p in detected_data]))

                if self.rel_pos is not None:
                    # 위치/방향 EMA 필터 적용
                    self.rel_pos = self.rel_pos * (1 - self.alpha) + new_pos * self.alpha
                    s = math.sin(self.heading_angle)*(1-self.alpha) + math.sin(new_h)*self.alpha
                    c = math.cos(self.heading_angle)*(1-self.alpha) + math.cos(new_h)*self.alpha
                    self.heading_angle = math.atan2(s, c)
                else:
                    self.rel_pos, self.heading_angle = new_pos, new_h
                
                # 궤적 업데이트
                self.path_history.append(self.rel_pos.astype(int))
                if len(self.path_history) > self.max_history: 
                    self.path_history.pop(0)

            if self.rel_pos is not None:
                self.draw_wheelchair_and_path(m_map, self.rel_pos, self.heading_angle)
                wx, wy = int(self.rel_pos[0]), int(self.rel_pos[1])
                gx, gy = wx - self.off_x, wy - self.off_y
                cv2.putText(m_map, f"GRID POS: ({gx}, {gy})", (wx+35, wy-20), 0, 0.4, (0, 255, 0), 1)

            cv2.imshow(self.win_name, m_map)
            cv2.imshow("Cam Stream", np.hstack([cv2.resize(f_l, (480, 270)), cv2.resize(f_r, (480, 270))]))

            key = cv2.waitKey(15) & 0xFF 
            if key == ord(' '): self.is_paused = not self.is_paused
            elif key == ord('q'): break

        self.cap_left.release(); self.cap_rear.release(); cv2.destroyAllWindows()

if __name__ == '__main__':
    UltimateWheelchairTracker().run()