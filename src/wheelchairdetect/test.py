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

        # 휠체어 규격 (cm)
        self.wc_w, self.wc_l = 57.0, 100.0           
        self.alpha = 0.15          # Smooth 필터 강도
        
        # 상태 변수
        self.marker_pos = None     # 등받이 마커의 현재 위치
        self.rel_pos = None        # 보정된 휠체어 중심 위치
        self.heading_angle = 0.0   

        # 2. 비디오 파일 로드
        self.cap0 = cv2.VideoCapture('rear.mp4')
        self.cap1 = cv2.VideoCapture('left.mp4')
        self.total_frames = int(min(self.cap0.get(cv2.CAP_PROP_FRAME_COUNT), 
                                    self.cap1.get(cv2.CAP_PROP_FRAME_COUNT)))
        
        self.curr_f0 = None
        self.curr_f1 = None

        # 카메라 설정 (Calibration 값)
        self.cams = {
            'cam1': { 
                'pos': np.array([200.0 + self.off_x, 270.0 + self.off_y]), 
                'h': 110.0, 'focal': 873.0, 'map_angle': 157, 
                'yaw': -42.0, 'fov': 45, 'color': (255, 120, 100), 'name': 'Left'
            },
            'cam0': { 
                'pos': np.array([301.4 + self.off_x, 540.0 + self.off_y]), 
                'h': 105.0, 'focal': 896.0, 'map_angle': 90, 
                'yaw': 0.0, 'fov': 45, 'color': (100, 120, 255), 'name': 'Rear'
            }
        }
        
        self.dist_gain = 1.19      
        self.angle_gain = 1.56     
        self.detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250),
            cv2.aruco.DetectorParameters()
        )

        self.win_name = "Integrated Wheelchair Tracker"
        cv2.namedWindow(self.win_name)
        
        # 슬라이더 구성
        cv2.createTrackbar("Frame", self.win_name, 0, self.total_frames - 1, self.on_frame_change)
        cv2.createTrackbar("L_Focal", self.win_name, 873, 1500, lambda v: self.upd('cam1','focal',v))
        cv2.createTrackbar("L_Yaw", self.win_name, 48, 180, lambda v: self.upd('cam1','yaw',v-90))
        cv2.createTrackbar("R_Focal", self.win_name, 896, 1500, lambda v: self.upd('cam0','focal',v))
        cv2.createTrackbar("R_Yaw", self.win_name, 90, 180, lambda v: self.upd('cam0','yaw',v-90))
        cv2.createTrackbar("Dist_Gain", self.win_name, 119, 200, self.on_dist_gain)
        cv2.createTrackbar("Smooth", self.win_name, 15, 100, self.on_alpha)

    def on_frame_change(self, v):
        self.cap0.set(cv2.CAP_PROP_POS_FRAMES, v)
        self.cap1.set(cv2.CAP_PROP_POS_FRAMES, v)
        _, self.curr_f0 = self.cap0.read()
        _, self.curr_f1 = self.cap1.read()

    def on_alpha(self, v): self.alpha = max(0.01, v / 100.0)
    def on_dist_gain(self, v): self.dist_gain = v / 100.0
    def upd(self, side, key, val): self.cams[side][key] = float(val)

    def draw_static_map(self, img):
        # 배경 격자
        step = int(20 * self.map_scale * 2) 
        for x in range(0, self.grid_w + 1, step):
            c = (45, 45, 45) if x % 100 != 0 else (80, 80, 80)
            cv2.line(img, (self.off_x + x, self.off_y), (self.off_x + x, self.off_y + self.grid_h), c, 1)
        for y in range(0, self.grid_h + 1, step):
            c = (45, 45, 45) if y % 100 != 0 else (80, 80, 80)
            cv2.line(img, (self.off_x, self.off_y + y), (self.off_x + self.grid_w, self.off_y + y), c, 1)
        # 차량 구역 및 경계
        cv2.rectangle(img, (200 + self.off_x, 180 + self.off_y), (400 + self.off_x, 540 + self.off_y), (35, 35, 45), -1)
        cv2.rectangle(img, (self.off_x, self.off_y), (self.off_x+self.grid_w, self.off_y+self.grid_h), (180, 180, 180), 2)
        # 카메라 위치 표시
        for side in self.cams:
            cv2.circle(img, tuple(self.cams[side]['pos'].astype(int)), 7, self.cams[side]['color'], -1)

    def run(self):
        play = True
        ret0, self.curr_f0 = self.cap0.read()
        ret1, self.curr_f1 = self.cap1.read()

        while True:
            if play:
                ret0, self.curr_f0 = self.cap0.read()
                ret1, self.curr_f1 = self.cap1.read()
                if not ret0 or not ret1:
                    self.on_frame_change(0)
                    continue
                cv2.setTrackbarPos("Frame", self.win_name, int(self.cap0.get(cv2.CAP_PROP_POS_FRAMES)))

            m_map = np.ones((self.map_h, self.map_w, 3), dtype=np.uint8) * 15
            self.draw_static_map(m_map)
            
            detected_data = []
            # 모니터링용 영상 (박스 그리기용)
            mon0 = self.curr_f0.copy() if self.curr_f0 is not None else None
            mon1 = self.curr_f1.copy() if self.curr_f1 is not None else None

            # 3. 마커 검출 및 개별 점 계산
            for i, (frame, mon_frame, side) in enumerate([(self.curr_f0, mon0, 'cam0'), (self.curr_f1, mon1, 'cam1')]):
                if frame is None: continue
                corners, ids, _ = self.detector.detectMarkers(frame)
                
                if ids is not None:
                    cv2.aruco.drawDetectedMarkers(mon_frame, corners, ids)
                    cfg = self.cams[side]
                    c = corners[0].reshape(4, 2)
                    
                    # 거리 및 맵 좌표 계산
                    px_h = (np.linalg.norm(c[0]-c[3]) + np.linalg.norm(c[1]-c[2])) / 2.0
                    raw_dist = (self.marker_size * cfg['focal']) / px_h
                    corr_dist = raw_dist * (1 + (self.dist_gain - 1) * (raw_dist / 500)) 
                    d = math.sqrt(max(0, corr_dist**2 - abs(cfg['h'] - self.marker_h)**2))

                    rel_x = (np.mean(c[:, 0]) - frame.shape[1]/2) / (frame.shape[1]/2)
                    t_rad = math.radians(cfg['map_angle'] + cfg['yaw']) + (rel_x * math.radians(cfg['fov'])) * self.angle_gain
                    raw_pos = cfg['pos'] + np.array([d * self.map_scale * math.cos(t_rad), d * self.map_scale * math.sin(t_rad)])
                    
                    # 방향(Heading) 계산: 마커가 등받이 평면에 있으므로 로컬 각도 추출
                    marker_vec = c[0] - c[3]
                    marker_rot = math.atan2(marker_vec[1], marker_vec[0]) - (math.pi/2)
                    h = t_rad + marker_rot
                    if ids[0][0] == 1: h += math.pi # 바깥쪽 마커(ID 1) 반전 보정
                    
                    detected_data.append((raw_pos, h))

                    # 카메라 시야 시각화 (개별 점과 호)
                    dist_px = int(d * self.map_scale)
                    cv2.circle(m_map, tuple(raw_pos.astype(int)), 6, cfg['color'], -1)
                    cv2.ellipse(m_map, tuple(cfg['pos'].astype(int)), (dist_px, dist_px), 0, 
                               math.degrees(t_rad)-5, math.degrees(t_rad)+5, cfg['color'], 2, cv2.LINE_AA)

            # 4. 데이터 융합 및 휠체어 몸체 중심 보정
            if len(detected_data) > 0:
                # 마커들이 가리키는 등받이의 평균 위치
                marker_avg_pos = np.mean([p[0] for p in detected_data], axis=0)
                t_h = math.atan2(np.mean([math.sin(p[1]) for p in detected_data]), 
                                 np.mean([math.cos(p[1]) for p in detected_data]))
                
                # 필터링 (재생 중일 때만 부드럽게)
                if play and self.marker_pos is not None:
                    self.marker_pos = self.marker_pos * (1 - self.alpha) + marker_avg_pos * self.alpha
                    self.heading_angle = math.atan2(math.sin(self.heading_angle)*(1-self.alpha) + math.sin(t_h)*self.alpha,
                                                    math.cos(self.heading_angle)*(1-self.alpha) + math.cos(t_h)*self.alpha)
                else:
                    self.marker_pos, self.heading_angle = marker_avg_pos, t_h

                # [중요] 등받이 마커 위치에서 앞방향으로 50cm(L/2) 이동하여 실제 중심점(rel_pos) 계산
                offset_dist = (self.wc_l / 2) * self.map_scale
                self.rel_pos = self.marker_pos + np.array([
                    offset_dist * math.cos(self.heading_angle),
                    offset_dist * math.sin(self.heading_angle)
                ])

            # 5. 최종 휠체어 렌더링
            if self.rel_pos is not None:
                w_px, l_px = (self.wc_w * self.map_scale) / 2, (self.wc_l * self.map_scale) / 2
                base_pts = np.array([[-l_px, -w_px], [l_px, -w_px], [l_px, w_px], [-l_px, w_px]])
                rot_m = np.array([[math.cos(self.heading_angle), -math.sin(self.heading_angle)], 
                                  [math.sin(self.heading_angle), math.cos(self.heading_angle)]])
                rotated_pts = np.dot(base_pts, rot_m.T) + self.rel_pos
                
                # 몸체 (초록색)
                cv2.polylines(m_map, [rotated_pts.astype(np.int32)], True, (0, 255, 0), 2, cv2.LINE_AA)
                
                # 등받이 라인 강조 (뒤쪽 변을 빨간색으로 표시하여 앞뒤 구분)
                back_line = rotated_pts[[0, 3]] 
                cv2.line(m_map, tuple(back_line[0].astype(int)), tuple(back_line[1].astype(int)), (0, 0, 255), 3)

                # 전방 화살표
                p1 = tuple(self.rel_pos.astype(int))
                arrow_len = 50
                p2 = (int(p1[0] + arrow_len * math.cos(self.heading_angle)), 
                      int(p1[1] + arrow_len * math.sin(self.heading_angle)))
                cv2.arrowedLine(m_map, p1, p2, (255, 255, 255), 3, tipLength=0.3)

            # 결과 화면 출력
            cv2.imshow(self.win_name, m_map)
            disp0 = cv2.resize(mon0, (480, 270)) if mon0 is not None else np.zeros((270,480,3), np.uint8)
            disp1 = cv2.resize(mon1, (480, 270)) if mon1 is not None else np.zeros((270,480,3), np.uint8)
            cv2.imshow("Detection Monitor", np.hstack([disp0, disp1]))

            key = cv2.waitKey(30) & 0xFF
            if key == ord(' '): play = not play
            elif key == ord('q'): break

        self.cap0.release(); self.cap1.release(); cv2.destroyAllWindows()

if __name__ == '__main__':
    TotalIntegrationTracker().run()