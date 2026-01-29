import cv2
import numpy as np
import math

class TotalIntegrationTracker:
    def __init__(self):
        # 1. 물리 및 지도 설정
        self.marker_size = 25.0
        self.marker_h = 72.0        
        self.map_w, self.map_h = 1000, 1000 
        self.grid_w, self.grid_h = 600, 720
        self.map_scale = 0.5 
        self.off_x, self.off_y = 200, 150

        # --- [상태 변수] ---
        self.wc_w = 57.0            
        self.wc_l = 100.0           
        self.alpha = 0.15          
        self.rel_pos = None        
        self.heading_angle = 0.0   

        # 2. 카메라 설정 (0: Rear, 1: Left)
        self.cap0 = cv2.VideoCapture(0) 
        self.cap1 = cv2.VideoCapture(1) 
        for c in [self.cap0, self.cap1]:
            c.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            c.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.cams = {
            'cam1': { 
                'pos': np.array([200.0 + self.off_x, 270.0 + self.off_y]), 
                'h': 110.0, 'focal': 797.0, 'map_angle': 157, 
                'yaw': 0.0, 'fov': 45, 'color': (255, 100, 100), 'name': 'Left-Cam'
            },
            'cam0': { 
                'pos': np.array([301.4 + self.off_x, 540.0 + self.off_y]), 
                'h': 105.0, 'focal': 896.0, 'map_angle': 90, 
                'yaw': 0.0, 'fov': 45, 'color': (100, 100, 255), 'name': 'Rear-Cam'
            }
        }
        
        self.dist_gain = 1.19      
        self.angle_gain = 1.56     
        self.detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250),
            cv2.aruco.DetectorParameters()
        )

        self.win_name = "Final Total System"
        cv2.namedWindow(self.win_name)
        
        cv2.createTrackbar("L_Focal", self.win_name, 797, 1500, lambda v: self.upd('cam1','focal',v))
        cv2.createTrackbar("L_Yaw", self.win_name, 90, 180, lambda v: self.upd('cam1','yaw',v-90))
        cv2.createTrackbar("R_Focal", self.win_name, 896, 1500, lambda v: self.upd('cam0','focal',v))
        cv2.createTrackbar("R_Yaw", self.win_name, 90, 180, lambda v: self.upd('cam0','yaw',v-90))
        cv2.createTrackbar("Dist_Gain", self.win_name, 119, 200, self.on_dist_gain)
        cv2.createTrackbar("Smooth", self.win_name, 15, 100, self.on_alpha)

    def on_alpha(self, v): self.alpha = max(0.01, v / 100.0)
    def on_dist_gain(self, v): self.dist_gain = v / 100.0
    def upd(self, side, key, val): self.cams[side][key] = float(val)

    def draw_static_map(self, img):
        # 1. 세부 격자 그리기 (20cm 단위 보조선, 100cm 단위 주선)
        step_cm = 20
        for x in range(0, self.grid_w + 1, int(step_cm * self.map_scale * 2)): # 20cm 간격
            color = (60, 60, 60) if x % 50 != 0 else (100, 100, 100)
            thickness = 1 if x % 50 != 0 else 2
            cv2.line(img, (self.off_x + x, self.off_y), (self.off_x + x, self.off_y + self.grid_h), color, thickness)
        for y in range(0, self.grid_h + 1, int(step_cm * self.map_scale * 2)):
            color = (60, 60, 60) if y % 50 != 0 else (100, 100, 100)
            thickness = 1 if y % 50 != 0 else 2
            cv2.line(img, (self.off_x, self.off_y + y), (self.off_x + self.grid_w, self.off_y + y), color, thickness)

        # 2. 메인 외곽선 및 장애물
        cv2.rectangle(img, (self.off_x, self.off_y), (self.off_x+self.grid_w, self.off_y+self.grid_h), (180, 180, 180), 2)
        cv2.rectangle(img, (200 + self.off_x, 180 + self.off_y), (400 + self.off_x, 540 + self.off_y), (40, 40, 40), -1)

        # 3. 카메라 위치 및 탐지 범위(호) 그리기
        for side in self.cams:
            cfg = self.cams[side]
            center = tuple(cfg['pos'].astype(int))
            # 카메라 위치 점
            cv2.circle(img, center, 8, cfg['color'], -1)
            cv2.putText(img, cfg['name'], (center[0]-30, center[1]-20), 0, 0.5, cfg['color'], 1)
            
            # 탐지 범위 호(Arc) 시각화 (반경 150px)
            angle_center = cfg['map_angle'] + cfg['yaw']
            fov = cfg['fov']
            axes = (100, 100) # 호의 반지름
            cv2.ellipse(img, center, axes, 0, angle_center - fov, angle_center + fov, cfg['color'], 1, cv2.LINE_AA)
            
            # 방향선 추가
            rad = math.radians(angle_center)
            p2 = (int(center[0] + 100 * math.cos(rad)), int(center[1] + 100 * math.sin(rad)))
            cv2.line(img, center, p2, cfg['color'], 1, cv2.LINE_AA)

    def get_corrected_dist(self, corners, cfg):
        c = corners.reshape(4, 2)
        px_h = (np.linalg.norm(c[0]-c[3]) + np.linalg.norm(c[1]-c[2])) / 2.0
        if px_h < 2: return None
        raw_dist = (self.marker_size * cfg['focal']) / px_h
        corr_dist = raw_dist * (1 + (self.dist_gain - 1) * (raw_dist / 500)) 
        h_diff = abs(cfg['h'] - self.marker_h)
        return math.sqrt(max(0, corr_dist**2 - h_diff**2))

    def draw_wheelchair(self, img, pos, angle):
        w_px = (self.wc_w * self.map_scale) / 2
        l_px = (self.wc_l * self.map_scale) / 2
        base_pts = np.array([[-l_px, -w_px], [l_px, -w_px], [l_px, w_px], [-l_px, w_px]])
        c, s = math.cos(angle), math.sin(angle)
        rot_m = np.array([[c, -s], [s, c]])
        rotated_pts = np.dot(base_pts, rot_m.T) + pos
        cv2.polylines(img, [rotated_pts.astype(np.int32)], True, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.arrowedLine(img, tuple(pos.astype(int)), 
                        (int(pos[0]+(l_px+20)*c), int(pos[1]+(l_px+20)*s)), (255, 255, 255), 2, tipLength=0.3)

    def run(self):
        while True:
            ret0, f0 = self.cap0.read(); ret1, f1 = self.cap1.read()
            if not ret0 or not ret1: break
            m_map = np.ones((self.map_h, self.map_w, 3), dtype=np.uint8) * 20
            self.draw_static_map(m_map)
            detected_data = []

            for i, (frame, side) in enumerate([(f0, 'cam0'), (f1, 'cam1')]):
                corners, ids, _ = self.detector.detectMarkers(frame)
                if ids is not None:
                    cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                    cfg = self.cams[side]
                    d = self.get_corrected_dist(corners[0], cfg)
                    if d:
                        rel_x = (np.mean(corners[0].reshape(4,2)[:, 0]) - frame.shape[1]/2) / (frame.shape[1]/2)
                        angle_offset = (rel_x * math.radians(cfg['fov'])) * self.angle_gain
                        t_rad = math.radians(cfg['map_angle'] + cfg['yaw']) + angle_offset
                        raw_pos = cfg['pos'] + np.array([d * self.map_scale * math.cos(t_rad), d * self.map_scale * math.sin(t_rad)])
                        
                        c_pts = corners[0].reshape(4, 2)
                        h = t_rad + math.atan2(c_pts[0,1]-c_pts[3,1], c_pts[0,0]-c_pts[3,0]) - (math.pi/2)
                        if ids[0][0] == 1: h += math.pi
                        detected_data.append((raw_pos, h))

                        # 실시간 탐지 거리 표시
                        cv2.circle(m_map, tuple(raw_pos.astype(int)), 5, cfg['color'], -1)
                        cv2.putText(m_map, f"{int(d)}cm", (int(raw_pos[0]+10), int(raw_pos[1]+10)), 0, 0.4, cfg['color'], 1)

            if len(detected_data) > 0:
                t_pos = np.mean([p[0] for p in detected_data], axis=0)
                t_h = math.atan2(np.mean([math.sin(p[1]) for p in detected_data]), np.mean([math.cos(p[1]) for p in detected_data]))
                if self.rel_pos is None: self.rel_pos, self.heading_angle = t_pos, t_h
                else:
                    self.rel_pos = self.rel_pos * (1 - self.alpha) + t_pos * self.alpha
                    self.heading_angle = math.atan2(math.sin(self.heading_angle)*(1-self.alpha) + math.sin(t_h)*self.alpha,
                                                    math.cos(self.heading_angle)*(1-self.alpha) + math.cos(t_h)*self.alpha)

            if self.rel_pos is not None:
                self.draw_wheelchair(m_map, self.rel_pos, self.heading_angle)
                wx, wy = int(self.rel_pos[0]), int(self.rel_pos[1])
                cv2.putText(m_map, f"Pos: {int(wx-self.off_x)}, {int(wy-self.off_y)}", (wx+20, wy-20), 0, 0.5, (0, 255, 0), 1)

            cv2.imshow(self.win_name, m_map)
            combined = np.hstack([cv2.resize(f0, (640, 360)), cv2.resize(f1, (640, 360))])
            cv2.imshow("Dual-View Monitor", combined)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        self.cap0.release(); self.cap1.release(); cv2.destroyAllWindows()

if __name__ == '__main__':
    TotalIntegrationTracker().run()