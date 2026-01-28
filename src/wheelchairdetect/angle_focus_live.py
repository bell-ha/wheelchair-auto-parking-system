import cv2
import numpy as np
import math

class FinalOptimizedTracker:
    def __init__(self):
        # 1. 물리 및 지도 설정 (원본 그대로 유지)
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

        self.cap0 = cv2.VideoCapture(0)  # rear
        self.cap1 = cv2.VideoCapture(1)  # left
        for c in [self.cap0, self.cap1]:
            c.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            c.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                
        self.curr_f0 = None
        self.curr_f1 = None

        # 카메라 설정 (원본 수치 유지)
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

        self.win_name = "Integrated Wheelchair Tracker"
        cv2.namedWindow(self.win_name)
        
        # [복구] 모든 트랙바 기능 원상복구
        # cv2.createTrackbar("Frame", self.win_name, 278, self.total_frames - 1, self.on_frame_change)
        cv2.createTrackbar("L_Focal", self.win_name, 841, 1500, lambda v: self.upd('cam1','focal',v))
        cv2.createTrackbar("L_Yaw", self.win_name, 91, 180, lambda v: self.upd('cam1','yaw',v-90))
        cv2.createTrackbar("R_Focal", self.win_name, 836, 1500, lambda v: self.upd('cam0','focal',v))
        cv2.createTrackbar("R_Yaw", self.win_name, 91, 180, lambda v: self.upd('cam0','yaw',v-90))
        cv2.createTrackbar("Dist_Gain", self.win_name, 103, 200, self.on_dist_gain)
        cv2.createTrackbar("Smooth", self.win_name, 75, 100, self.on_alpha)

        # self.on_frame_change(278)

    def on_frame_change(self, v):
        self.cap0.set(cv2.CAP_PROP_POS_FRAMES, v)
        self.cap1.set(cv2.CAP_PROP_POS_FRAMES, v)
        _, self.curr_f0 = self.cap0.read()
        _, self.curr_f1 = self.cap1.read()

    def on_alpha(self, v): self.alpha = max(0.01, v / 100.0)
    def on_dist_gain(self, v): self.dist_gain = v / 100.0
    def upd(self, side, key, val): self.cams[side][key] = float(val)

    def draw_static_map(self, img):
        step = int(20 * self.map_scale * 2) 
        for x in range(0, self.grid_w + 1, step):
            c = (45, 45, 45) if x % 100 != 0 else (80, 80, 80)
            cv2.line(img, (self.off_x + x, self.off_y), (self.off_x + x, self.off_y + self.grid_h), c, 1)
        for y in range(0, self.grid_h + 1, step):
            c = (45, 45, 45) if y % 100 != 0 else (80, 80, 80)
            cv2.line(img, (self.off_x, self.off_y + y), (self.off_x + self.grid_w, self.off_y + y), c, 1)
        
        cv2.rectangle(img, (200 + self.off_x, 180 + self.off_y), (400 + self.off_x, 540 + self.off_y), (35, 35, 45), -1)
        cv2.rectangle(img, (self.off_x, self.off_y), (self.off_x+self.grid_w, self.off_y+self.grid_h), (180, 180, 180), 2)
        
        for side in self.cams:
            cfg = self.cams[side]
            cp = tuple(cfg['pos'].astype(int))
            cv2.circle(img, cp, 7, cfg['color'], -1)
            cv2.putText(img, cfg['name'], (cp[0]-25, cp[1]+25), 0, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    def run(self):
        play = True
        while True:
            if play:
                ret0, self.curr_f0 = self.cap0.read()
                ret1, self.curr_f1 = self.cap1.read()
                if not ret0 or not ret1:
                    self.on_frame_change(0); continue
                cv2.setTrackbarPos("Frame", self.win_name, int(self.cap0.get(cv2.CAP_PROP_POS_FRAMES)))

            m_map = np.ones((self.map_h, self.map_w, 3), dtype=np.uint8) * 15
            self.draw_static_map(m_map)
            
            detected_data = [] # (pos, h, weight) 형태로 저장
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
                    
                    # 수평 각도 및 가중치 계산 (중심부 신뢰도 강화)
                    rel_x = (np.mean(c[:, 0]) - frame.shape[1]/2) / (frame.shape[1]/2)
                    weight = max(0.1, 1.0 - abs(rel_x)) # 중심에서 멀어질수록 가중치 감소
                    
                    m_yaw_deg = (rel_x * cfg['fov']) * self.angle_gain
                    t_rad = math.radians(cfg['map_angle'] + cfg['yaw'] + m_yaw_deg)
                    raw_pos = cfg['pos'] + np.array([d * self.map_scale * math.cos(t_rad), d * self.map_scale * math.sin(t_rad)])
                    
                    marker_vec = c[0] - c[3]
                    h = t_rad + math.atan2(marker_vec[1], marker_vec[0]) - (math.pi/2)
                    if ids[0][0] == 1: h += math.pi 
                    
                    detected_data.append((raw_pos, h, weight))

                    # [복구] 개별 카메라별 거리 호, 연결선, 텍스트 표시
                    cp, rp = tuple(cfg['pos'].astype(int)), tuple(raw_pos.astype(int))
                    dist_px = int(d * self.map_scale)
                    cv2.ellipse(m_map, cp, (dist_px, dist_px), 0, math.degrees(t_rad)-5, math.degrees(t_rad)+5, cfg['color'], 2, cv2.LINE_AA)
                    cv2.line(m_map, cp, rp, cfg['color'], 1, cv2.LINE_AA)
                    txt_pos = ((cp[0]+rp[0])//2, (cp[1]+rp[1])//2)
                    cv2.putText(m_map, f"{d:.0f}cm / {m_yaw_deg:+.1f}deg (w:{weight:.1f})", (txt_pos[0]+5, txt_pos[1]-5), 0, 0.4, (180, 180, 180), 1, cv2.LINE_AA)

            # [개선] 가중 평균을 이용한 데이터 통합
            if len(detected_data) > 0:
                total_w = sum(p[2] for p in detected_data)
                avg_pos = sum(p[0] * p[2] for p in detected_data) / total_w
                
                # 각도 벡터 합산 (튐 방지)
                avg_sin = sum(math.sin(p[1]) * p[2] for p in detected_data) / total_w
                avg_cos = sum(math.cos(p[1]) * p[2] for p in detected_data) / total_w
                avg_h = math.atan2(avg_sin, avg_cos)
                
                if play and self.is_initialized:
                    # Smoothing 적용
                    self.marker_pos = self.marker_pos * (1 - self.alpha) + avg_pos * self.alpha
                    diff = (avg_h - self.heading_angle + math.pi) % (2 * math.pi) - math.pi
                    self.heading_angle += diff * self.alpha
                else:
                    self.marker_pos, self.heading_angle, self.is_initialized = avg_pos, avg_h, True

            if self.is_initialized:
                # [원본] 휠체어 렌더링 로직 그대로 유지
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

            cv2.imshow(self.win_name, m_map)
            # [복구] 모니터 창 예외 처리 및 출력
            m0_res = cv2.resize(mon0, (640, 360)) if mon0 is not None else np.zeros((360, 640, 3), np.uint8)
            m1_res = cv2.resize(mon1, (640, 360)) if mon1 is not None else np.zeros((360, 640, 3), np.uint8)
            cv2.imshow("Monitor", np.hstack([m0_res, m1_res]))
            
            key = cv2.waitKey(30) & 0xFF
            if key == ord(' '): play = not play
            elif key == ord('q'): break

        self.cap0.release(); self.cap1.release(); cv2.destroyAllWindows()

if __name__ == '__main__':
    FinalOptimizedTracker().run()