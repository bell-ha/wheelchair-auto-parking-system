import cv2
import numpy as np
import math

class FinalOptimizedTracker:
    def __init__(self):
        # 1. 카메라 광학 데이터 (제공해주신 값 고정)
        self.K = np.array([[601.71923, 0.0, 630.477007], [0.0, 601.3453, 367.2122], [0.0, 0.0, 1.0]], dtype=np.float32)
        self.D = np.array([-0.18495647, 0.02541005, -0.01068433, 0.00321714], dtype=np.float32)
        
        # 2. 물리 및 지도 설정 (원본 유지)
        self.marker_size = 25.0 # cm
        self.marker_h = 72.0        
        self.map_w, self.map_h = 1000, 1000 
        self.grid_w, self.grid_h = 600, 720 
        self.map_scale = 0.5 
        self.off_x, self.off_y = 200, 150
        self.wc_w, self.wc_l = 57.0, 100.0           
        
        self.marker_pos = None     
        self.heading_angle = 0.0   
        self.is_initialized = False 

        # 동영상 로드
        self.cap0 = cv2.VideoCapture('rear.mp4')
        self.cap1 = cv2.VideoCapture('left.mp4')
        self.total_frames = int(min(self.cap0.get(cv2.CAP_PROP_FRAME_COUNT), 
                                    self.cap1.get(cv2.CAP_PROP_FRAME_COUNT)))
        
        self.curr_f0 = None
        self.curr_f1 = None

        # 카메라 설치 정보 (초기값)
        self.cams = {
            'cam1': { # Left
                'pos': np.array([200.0 + self.off_x, 270.0 + self.off_y]), 
                'h': 110.0, 'map_angle': 0, 'sens': 1.6, 'inst': 113, 'offset': 50,
                'color': (255, 120, 100), 'name': 'Left'
            },
            'cam0': { # Rear
                'pos': np.array([301.4 + self.off_x, 540.0 + self.off_y]), 
                'h': 105.0, 'map_angle': 90, 'sens': 1.6, 'inst': 0, 'offset': 0,
                'color': (100, 120, 255), 'name': 'Rear'
            }
        }
        
        self.alpha = 0.75          
        self.detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250),
            cv2.aruco.DetectorParameters()
        )

        self.win_name = "Integrated Wheelchair Tracker"
        cv2.namedWindow(self.win_name)
        
        # [복구 및 확장] 트랙바 구성
        cv2.createTrackbar("Frame", self.win_name, 0, self.total_frames - 1, self.on_frame_change)
        cv2.createTrackbar("L_Sens(x10)", self.win_name, 16, 30, lambda v: self.upd('cam1','sens',v/10.0))
        cv2.createTrackbar("L_Inst", self.win_name, 113, 180, lambda v: self.upd('cam1','inst',v))
        cv2.createTrackbar("L_Offset", self.win_name, 50, 100, lambda v: self.upd('cam1','offset',v))
        cv2.createTrackbar("Smooth", self.win_name, 75, 100, self.on_alpha)

        self.on_frame_change(0)
        
        # 3D 마커 좌표 (solvePnP용)
        s = self.marker_size / 2
        self.obj_pts = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], dtype=np.float32)

    def on_frame_change(self, v):
        self.cap0.set(cv2.CAP_PROP_POS_FRAMES, v)
        self.cap1.set(cv2.CAP_PROP_POS_FRAMES, v)
        _, self.curr_f0 = self.cap0.read()
        _, self.curr_f1 = self.cap1.read()

    def on_alpha(self, v): self.alpha = max(0.01, v / 100.0)
    def upd(self, side, key, val): self.cams[side][key] = float(val)

    def draw_static_map(self, img):
        # (기존 격자 및 맵 렌더링 유지)
        cv2.rectangle(img, (self.off_x, self.off_y), (self.off_x+self.grid_w, self.off_y+self.grid_h), (180, 180, 180), 2)
        for side, cfg in self.cams.items():
            cp = tuple(cfg['pos'].astype(int))
            cv2.circle(img, cp, 7, cfg['color'], -1)

    def run(self):
        play = False
        while True:
            if play:
                ret0, self.curr_f0 = self.cap0.read()
                ret1, self.curr_f1 = self.cap1.read()
                if not ret0 or not ret1: self.on_frame_change(0); continue
                cv2.setTrackbarPos("Frame", self.win_name, int(self.cap0.get(cv2.CAP_PROP_POS_FRAMES)))

            m_map = np.ones((self.map_h, self.map_w, 3), dtype=np.uint8) * 15
            self.draw_static_map(m_map)
            
            detected_data = [] 
            mon_frames = [self.curr_f0.copy() if self.curr_f0 is not None else None,
                          self.curr_f1.copy() if self.curr_f1 is not None else None]

            for i, (frame, side) in enumerate([(self.curr_f0, 'cam0'), (self.curr_f1, 'cam1')]):
                if frame is None: continue
                corners, ids, _ = self.detector.detectMarkers(frame)
                if ids is not None:
                    cfg = self.cams[side]
                    # [핵심] 어안 보정 적용
                    undistorted = cv2.fisheye.undistortPoints(corners[0].reshape(-1, 1, 2), self.K, self.D, P=self.K)
                    _, rvec, tvec = cv2.solvePnP(self.obj_pts, undistorted, self.K, None)
                    
                    dist_cm = np.linalg.norm(tvec)
                    floor_d = math.sqrt(max(0, dist_cm**2 - abs(cfg['h'] - self.marker_h)**2))
                    
                    rmat, _ = cv2.Rodrigues(rvec)
                    raw_yaw = math.atan2(-rmat[2,0], math.sqrt(rmat[0,0]**2 + rmat[1,0]**2)) * 180 / math.pi
                    
                    # 보정 수식
                    final_yaw_deg = (raw_yaw * cfg['sens']) + cfg['inst'] - cfg['offset']
                    t_rad = math.radians(cfg['map_angle'] + final_yaw_deg)
                    
                    raw_pos = cfg['pos'] + np.array([floor_d * self.map_scale * math.cos(t_rad), 
                                                     floor_d * self.map_scale * math.sin(t_rad)])
                    
                    # 가중치 (중앙 신뢰도)
                    rel_x = (np.mean(corners[0][0][:, 0]) - frame.shape[1]/2) / (frame.shape[1]/2)
                    weight = max(0.1, 1.0 - abs(rel_x))
                    
                    detected_data.append((raw_pos, t_rad, weight))
                    
                    # 지도 시각화
                    cp, rp = tuple(cfg['pos'].astype(int)), tuple(raw_pos.astype(int))
                    cv2.line(m_map, cp, rp, cfg['color'], 1, cv2.LINE_AA)
                    cv2.putText(m_map, f"{dist_cm:.0f}cm", (rp[0], rp[1]-10), 0, 0.4, (200, 200, 200), 1)

            # 데이터 통합
            if len(detected_data) > 0:
                total_w = sum(p[2] for p in detected_data)
                avg_pos = sum(p[0] * p[2] for p in detected_data) / total_w
                avg_h = sum(p[1] * p[2] for p in detected_data) / total_w
                
                if self.is_initialized:
                    self.marker_pos = self.marker_pos * (1 - self.alpha) + avg_pos * self.alpha
                    self.heading_angle = avg_h
                else:
                    self.marker_pos, self.heading_angle, self.is_initialized = avg_pos, avg_h, True

            if self.is_initialized:
                # 휠체어 렌더링 (기존 로직)
                center_pos = self.marker_pos + np.array([(self.wc_l/2)*self.map_scale*math.cos(self.heading_angle), 
                                                         (self.wc_l/2)*self.map_scale*math.sin(self.heading_angle)])
                w_px, l_px = (self.wc_w*self.map_scale)/2, (self.wc_l*self.map_scale)/2
                base_pts = np.array([[-l_px, -w_px], [l_px, -w_px], [l_px, w_px], [-l_px, w_px]])
                rot_m = np.array([[math.cos(self.heading_angle), -math.sin(self.heading_angle)], 
                                  [math.sin(self.heading_angle), math.cos(self.heading_angle)]])
                rotated_pts = np.dot(base_pts, rot_m.T) + center_pos
                cv2.polylines(m_map, [rotated_pts.astype(np.int32)], True, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.arrowedLine(m_map, tuple(center_pos.astype(int)), 
                                (int(center_pos[0]+30*math.cos(self.heading_angle)), 
                                 int(center_pos[1]+30*math.sin(self.heading_angle))), (255, 255, 255), 2)

            cv2.imshow(self.win_name, m_map)
            # 모니터 출력
            m0 = cv2.resize(mon_frames[0], (640, 360)) if mon_frames[0] is not None else np.zeros((360,640,3),np.uint8)
            m1 = cv2.resize(mon_frames[1], (640, 360)) if mon_frames[1] is not None else np.zeros((360,640,3),np.uint8)
            cv2.imshow("Monitor", np.hstack([m0, m1]))
            
            key = cv2.waitKey(30) & 0xFF
            if key == ord(' '): play = not play
            elif key == ord('q'): break

        self.cap0.release(); self.cap1.release(); cv2.destroyAllWindows()

if __name__ == '__main__':
    FinalOptimizedTracker().run()