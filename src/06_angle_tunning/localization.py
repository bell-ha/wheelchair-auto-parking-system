import cv2
import numpy as np
import math

class PoseEstimator:
    """ArUco 마커 기반 위치 및 방향 추정 모듈 (실시간 정밀 튜닝 버전)"""
    
    def __init__(self, K, D, cams, marker_size, alpha):
        self.K = K
        self.D = D
        self.cams = cams
        self.marker_size = marker_size # 25.0 (cm)
        # 튜닝 중에는 alpha 값을 무시하고 1.0을 사용하여 즉각적인 반응을 유도합니다.
        self.alpha = alpha 
        
        # ID별 마커 물리 데이터
        # ID 0: 앞(70cm), ID 1: 뒤(56cm)
        self.marker_info = {
            0: {'h_cm': 70.0, 'tilt_deg': 13.0},
            1: {'h_cm': 56.0, 'tilt_deg': 0.0}
        }
        
        self.detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250), 
            cv2.aruco.DetectorParameters()
        )
        
        self.marker_pos = None
        self.heading_angle = 0.0
        self.is_initialized = False
        self.subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)

    def detect_and_estimate(self, frames, current_dist_gain):
        """
        current_dist_gain: main의 트랙바에서 전달받은 미세 보정 값
        """
        detected_data = []
        
        for cam_name, frame in frames.items():
            if frame is None or cam_name not in self.cams:
                continue
            
            cfg = self.cams[cam_name]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = self.detector.detectMarkers(gray)
            
            if ids is not None:
                for i in range(len(ids)):
                    mid = int(ids[i][0])
                    if mid not in self.marker_info:
                        continue
                    
                    # 1. 서브픽셀 정밀화 및 시각화
                    cv2.cornerSubPix(gray, corners[i], (5, 5), (-1, -1), self.subpix_criteria)
                    c_pts = corners[i].astype(np.int32).reshape((-1, 2))
                    cv2.polylines(frame, [c_pts], True, (0, 255, 0), 2)
                    
                    # 2. 왜곡 보정 및 SolvePnP
                    pts_2d = corners[i].reshape(-1, 1, 2)
                    undist = cv2.fisheye.undistortPoints(pts_2d, self.K, self.D, P=self.K)
                    
                    ms = self.marker_size
                    obj_points = np.array([
                        [-ms/2,  ms/2, 0], [ ms/2,  ms/2, 0],
                        [ ms/2, -ms/2, 0], [-ms/2, -ms/2, 0]
                    ], dtype=np.float32)
                    
                    ret, rvec, tvec = cv2.solvePnP(obj_points, undist, self.K, None, flags=cv2.SOLVEPNP_SQPNP)
                    
                    if ret:
                        tvec = tvec.flatten()
                        
                        # [보정] DistGain 적용
                        # 근거: 어안 렌즈의 외곽부 거리 왜곡을 보정 계수로 상쇄
                        dist_cm = np.linalg.norm(tvec) * current_dist_gain
                        
                        # 3. 바닥 수평 거리(ground_d) 계산
                        # 근거: 마커 높이차(dh_m)가 빗변(dist_cm)보다 클 경우 제곱근 에러 방지를 위해 0.01cm 최소값 할당
                        m_h = self.marker_info[mid]['h_cm']
                        dh_cm = abs(cfg['h_cm'] - m_h)
                        
                        dist_sq_diff = dist_cm**2 - dh_cm**2
                        ground_cm = math.sqrt(dist_sq_diff) if dist_sq_diff > 0 else 0.01
                        
                        # 4. 각도(Yaw) 계산
                        rmat, _ = cv2.Rodrigues(rvec)
                        sy = math.sqrt(rmat[0,0]**2 + rmat[1,0]**2)
                        raw_yaw = math.atan2(-rmat[2,0], sy) * 180.0 / math.pi
                        
                        final_yaw_compass = (raw_yaw * cfg['sens']) + cfg['install_angle'] - cfg['install_offset']
                        if mid == 1: # 뒤쪽 마커 방향 반전
                            final_yaw_compass = (final_yaw_compass + 180.0) % 360.0
                            
                        h_rad = math.radians((final_yaw_compass + 270.0) % 360.0)
                        
                        # 5. 글로벌 좌표 투영
                        bearing_deg = math.degrees(math.atan2(tvec[0], tvec[2]))
                        ray_deg = cfg['map_angle_deg'] + cfg.get('yaw_trim_deg', 0) + bearing_deg
                        ray_rad = math.radians(ray_deg)
                        
                        # 마커의 맵상 좌표 (px)
                        m_pos_px = cfg['pos_px'] + np.array([
                            ground_cm * cfg['map_scale'] * math.cos(ray_rad),
                            ground_cm * cfg['map_scale'] * math.sin(ray_rad)
                        ])
                        
                        # 6. 마커 위치 -> 휠체어 중심(Center) 변환 (50cm 이동)
                        shift_px = (100.0 / 2.0) * cfg['map_scale']
                        if mid == 0: # 앞 마커는 뒤로 50cm
                            c_pos_px = m_pos_px - np.array([shift_px * math.cos(h_rad), shift_px * math.sin(h_rad)])
                        else:        # 뒤 마커는 앞으로 50cm
                            c_pos_px = m_pos_px + np.array([shift_px * math.cos(h_rad), shift_px * math.sin(h_rad)])

                        # 가중치 계산
                        weight = max(0.05, 1.0 - abs((np.mean(corners[i][:, 0, 0]) - frame.shape[1]/2) / (frame.shape[1]/2)))
                        detected_data.append((c_pos_px, h_rad, weight))
                        
                        # 모니터링 텍스트 그리기 (단위 m 표시)
                        cv2.putText(frame, f"ID:{mid} Dist:{dist_cm/100.0:.2f}m", (c_pts[0][0], c_pts[0][1]-10),
                                    0, 0.5, (0, 255, 0), 2)

        # 7. 데이터 통합
        if len(detected_data) > 0:
            total_w = sum(p[2] for p in detected_data)
            avg_pos = sum(p[0] * p[2] for p in detected_data) / total_w
            avg_sin = sum(math.sin(p[1]) * p[2] for p in detected_data) / total_w
            avg_cos = sum(math.cos(p[1]) * p[2] for p in detected_data) / total_w
            avg_h = math.atan2(avg_sin, avg_cos)
            
            # 튜닝 기간이므로 alpha를 1.0으로 간주하여 필터 없이 즉각 갱신
            tuning_alpha = 1.0 
            if not self.is_initialized:
                self.marker_pos = avg_pos
                self.heading_angle = avg_h
                self.is_initialized = True
            else:
                self.marker_pos = self.marker_pos * (1 - tuning_alpha) + avg_pos * tuning_alpha
                diff = (avg_h - self.heading_angle + math.pi) % (2 * math.pi) - math.pi
                self.heading_angle += diff * tuning_alpha
        
        return self.marker_pos, self.heading_angle, self.is_initialized
    
    def get_center_position(self, wc_l, map_scale):
        """휠체어 중심점 반환"""
        if self.marker_pos is None:
            return None
        return self.marker_pos