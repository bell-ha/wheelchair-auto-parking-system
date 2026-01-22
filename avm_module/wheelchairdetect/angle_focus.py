import cv2
import numpy as np
import math

class FinalSmoothTracker:
    def __init__(self):
        # 1. 물리 및 지도 설정
        self.marker_size = 25.0
        self.marker_h = 40.0
        self.map_w, self.map_h = 1000, 1000 
        self.grid_w, self.grid_h = 600, 720
        self.map_scale = 0.5 
        self.off_x, self.off_y = 200, 150

        # --- [부드러운 이동(EMA 필터) 설정] ---
        self.alpha = 0.15          # 0.1 ~ 0.2 사이가 가장 적당합니다.
        self.rel_pos = None        # 필터링된 현재 위치
        self.heading_angle = 0.0   # 필터링된 현재 각도

        # 2. [최신 프리셋 적용] 카메라 설정
        self.cams = {
            'left': {
                'pos': np.array([200.0 + self.off_x, 270.0 + self.off_y]), 
                'h': 115.0, 
                'focal': 797.0,    # 스크린샷 수치 반영
                'map_angle': 135, 
                'yaw': 24.0,       # 114 - 90
                'w_angle': 360.0, 
                'color': (0, 0, 255) 
            },
            'rear': {
                'pos': np.array([301.4 + self.off_x, 540.0 + self.off_y]), 
                'h': 85.0, 
                'focal': 896.0,    # 스크린샷 수치 반영
                'map_angle': 90, 
                'yaw': 1.0,        # 91 - 90
                'w_angle': 360.0, 
                'color': (255, 0, 0) 
            }
        }
        
        # [최신 프리셋 적용] 보정 계수
        self.dist_gain = 1.19      # 스크린샷 수치 반영
        self.angle_gain = 1.56     # 스크린샷 수치 반영

        # 3. 비디오 소스 및 탐지기 설정
        self.cap_left = cv2.VideoCapture('left.mp4')
        self.cap_rear = cv2.VideoCapture('rear.mp4')
        self.total_frames = int(self.cap_left.get(cv2.CAP_PROP_FRAME_COUNT))
        self.detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50))

        self.win_name = "Final Smooth Sensor Fusion"
        cv2.namedWindow(self.win_name)
        
        # --- 슬라이더 생성 (스크린샷 프리셋 순서 유지) ---
        cv2.createTrackbar("Progress", self.win_name, 0, self.total_frames - 1, self.on_frame)
        cv2.createTrackbar("L_Focal", self.win_name, 797, 1500, lambda v: self.upd('left','focal',v))
        cv2.createTrackbar("L_Yaw", self.win_name, 114, 180, lambda v: self.upd('left','yaw',v-90))
        cv2.createTrackbar("R_Focal", self.win_name, 896, 1500, lambda v: self.upd('rear','focal',v))
        cv2.createTrackbar("R_Yaw", self.win_name, 91, 180, lambda v: self.upd('rear','yaw',v-90))
        cv2.createTrackbar("Dist_Gain", self.win_name, 119, 200, self.on_dist_gain)
        cv2.createTrackbar("Angle_Gain", self.win_name, 156, 200, self.on_angle_gain)
        cv2.createTrackbar("Smooth(%)", self.win_name, 15, 100, self.on_alpha)

        self.is_paused = False
        self.last_f = None

    def on_frame(self, v): 
        self.cap_left.set(cv2.CAP_PROP_POS_FRAMES, v)
        self.cap_rear.set(cv2.CAP_PROP_POS_FRAMES, v)

    def on_alpha(self, v): self.alpha = max(0.01, v / 100.0)
    def upd(self, side, key, val): self.cams[side][key] = float(val)
    def on_dist_gain(self, v): self.dist_gain = v / 100.0
    def on_angle_gain(self, v): self.angle_gain = v / 100.0

    def draw_grid(self, img):
        start_pt, end_pt = (self.off_x, self.off_y), (self.off_x + self.grid_w, self.off_y + self.grid_h)
        cv2.rectangle(img, start_pt, end_pt, (200, 200, 200), 2)
        for i in range(1, 3):
            x = self.off_x + (self.grid_w // 3) * i
            y = self.off_y + (self.grid_h // 3) * i
            cv2.line(img, (x, self.off_y), (x, self.off_y + self.grid_h), (80, 80, 80), 1)
            cv2.line(img, (self.off_x, y), (self.off_x + self.grid_w, y), (80, 80, 80), 1)
        cv2.putText(img, "(0,0)", (self.off_x-40, self.off_y-10), 0, 0.4, (255,255,255), 1)

    def get_corrected_dist(self, corners, cfg):
        c = corners.reshape(4, 2)
        px_h = (np.linalg.norm(c[0]-c[3]) + np.linalg.norm(c[1]-c[2])) / 2.0
        if px_h < 2: return None
        raw_dist = (self.marker_size * cfg['focal']) / px_h
        corr_dist = raw_dist * (1 + (self.dist_gain - 1) * (raw_dist / 500)) 
        return math.sqrt(max(0, corr_dist**2 - abs(cfg['h'] - self.marker_h)**2))

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

            m_map = np.ones((self.map_h, self.map_w, 3), dtype=np.uint8) * 20
            self.draw_grid(m_map)
            cv2.rectangle(m_map, (200 + self.off_x, 180 + self.off_y), (400 + self.off_x, 540 + self.off_y), (60, 60, 60), -1)

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
                        
                        # 헤딩 계산
                        c = corners[0].reshape(4, 2)
                        vec = c[0] - c[3]
                        h = t_rad + math.atan2(vec[1], vec[0]) - (math.pi / 2) + math.radians(cfg['w_angle'])
                        if ids[0][0] == 1: h += math.pi
                        
                        detected_data.append((raw_pos, h))
                        
                        # 각 카메라 탐지 호(실시간)
                        cv2.ellipse(m_map, tuple(cfg['pos'].astype(int)), (int(d*self.map_scale), int(d*self.map_scale)), 0, 
                                   math.degrees(t_rad)-15, math.degrees(t_rad)+15, cfg['color'], 1)
                        cv2.circle(m_map, tuple(raw_pos.astype(int)), 4, cfg['color'], -1)

            # --- [데이터 퓨전 및 EMA 필터링] ---
            if len(detected_data) > 0:
                t_pos = np.mean([p[0] for p in detected_data], axis=0)
                t_sin = np.mean([math.sin(p[1]) for p in detected_data])
                t_cos = np.mean([math.cos(p[1]) for p in detected_data])
                t_h = math.atan2(t_sin, t_cos)

                if self.rel_pos is None:
                    self.rel_pos, self.heading_angle = t_pos, t_h
                else:
                    # 위치와 각도를 부드럽게 갱신
                    self.rel_pos = self.rel_pos * (1 - self.alpha) + t_pos * self.alpha
                    s = math.sin(self.heading_angle)*(1-self.alpha) + t_sin*self.alpha
                    c = math.cos(self.heading_angle)*(1-self.alpha) + t_cos*self.alpha
                    self.heading_angle = math.atan2(s, c)

            if self.rel_pos is not None:
                wx, wy = int(self.rel_pos[0]), int(self.rel_pos[1])
                cv2.circle(m_map, (wx, wy), 10, (0, 255, 0), -1)
                ax, ay = int(wx + 60 * math.cos(self.heading_angle)), int(wy + 60 * math.sin(self.heading_angle))
                cv2.arrowedLine(m_map, (wx, wy), (ax, ay), (255, 255, 255), 2, tipLength=0.3)
                cv2.putText(m_map, f"Grid: {wx-self.off_x}, {wy-self.off_y}", (wx+15, wy-10), 0, 0.4, (0,255,0), 1)

            cv2.imshow(self.win_name, m_map)
            cv2.imshow("Monitoring", np.hstack([cv2.resize(f_l, (480, 270)), cv2.resize(f_r, (480, 270))]))

            key = cv2.waitKey(15) & 0xFF 
            if key == ord(' '): self.is_paused = not self.is_paused
            elif key == ord('q'): break

        self.cap_left.release(); self.cap_rear.release(); cv2.destroyAllWindows()

if __name__ == '__main__':
    FinalSmoothTracker().run()