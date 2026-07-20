
import cv2
import numpy as np
import math


class PoseEstimator:
    """ArUco 마커 기반 위치 및 방향 추정 모듈"""
    
    def __init__(self, K, D, cams, marker_size, marker_h, dist_gain, alpha):
        self.K = K
        self.D = D
        self.cams = cams
        self.marker_size = marker_size
        self.marker_h = marker_h
        self.dist_gain = dist_gain
        self.alpha = alpha
        
        self.detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250), 
            cv2.aruco.DetectorParameters()
        )
        
        self.marker_pos = None
        self.heading_angle = 0.0
        self.is_initialized = False
    
    def detect_and_estimate(self, frames):
        """
        프레임에서 마커를 감지하고 위치/방향 추정
        
        Args:
            frames: {camera_name: frame} 딕셔너리
        
        Returns:
            (marker_pos, heading_angle, is_initialized) 튜플
        """
        detected_data = []
        
        for cam_name, frame in frames.items():
            if frame is None or cam_name not in self.cams:
                continue
            
            corners, ids, _ = self.detector.detectMarkers(frame)
            
            if ids is not None:
                cfg = self.cams[cam_name]
                
                # 1. 캘리브레이션 데이터 적용
                pts_2d = corners[0].reshape(-1, 1, 2)
                
                # 렌즈 왜곡 보정
                undistorted_pts = cv2.fisheye.undistortPoints(pts_2d, self.K, self.D, P=self.K)
                
                # 2. 3D 마커 객체 좌표
                ms = self.marker_size
                obj_points = np.array([
                    [-ms/2,  ms/2, 0],
                    [ ms/2,  ms/2, 0],
                    [ ms/2, -ms/2, 0],
                    [-ms/2, -ms/2, 0]
                ], dtype=np.float32)
                
                # 3. SolvePnP
                ret, rvec, tvec = cv2.solvePnP(
                    obj_points, undistorted_pts, self.K, None, 
                    flags=cv2.SOLVEPNP_IPPE_SQUARE
                )
                
                if ret:
                    tvec = tvec.flatten()
                    x_offset, y_offset, z_dist = tvec
                    
                    # 거리 보정
                    d_raw = np.linalg.norm(tvec)
                    d = d_raw * (1 + (self.dist_gain - 1) * (d_raw / 500))
                    
                    # 바닥 거리 계산
                    dh = abs(cfg['h'] - self.marker_h)
                    ground_d = math.sqrt(max(0, d**2 - dh**2))
                    
                    # 4. 글로벌 좌표 변환
                    ray_angle = math.atan2(x_offset, z_dist)
                    cam_global_angle = math.radians(cfg['map_angle'] + cfg['yaw'])
                    t_rad = cam_global_angle + ray_angle
                    
                    pos = cfg['pos'] + np.array([
                        ground_d * cfg.get('map_scale', 0.5) * math.cos(t_rad), 
                        ground_d * cfg.get('map_scale', 0.5) * math.sin(t_rad)
                    ])
                    
                    # 5. 헤딩(Yaw) 계산
                    R, _ = cv2.Rodrigues(rvec)
                    sy = math.sqrt(R[0, 0]**2 + R[1, 0]**2)
                    
                    if sy < 1e-6:
                        local_yaw = math.atan2(-R[1, 2], R[1, 1])
                    else:
                        local_yaw = math.atan2(R[1, 0], R[0, 0])
                    
                    # 방향 보정
                    h = cam_global_angle + local_yaw + math.pi
                    if ids[0][0] == 1:
                        h += math.pi
                    
                    # 가중치 계산
                    rel_x = (np.mean(corners[0][:, 0, 0]) - frame.shape[1]/2) / (frame.shape[1]/2)
                    weight = max(0.1, 1.0 - abs(rel_x))
                    
                    detected_data.append((pos, h, weight))
        
        # 데이터 통합
        if len(detected_data) > 0:
            total_w = sum(p[2] for p in detected_data)
            avg_pos = sum(p[0] * p[2] for p in detected_data) / total_w
            avg_sin = sum(math.sin(p[1]) * p[2] for p in detected_data) / total_w
            avg_cos = sum(math.cos(p[1]) * p[2] for p in detected_data) / total_w
            avg_h = math.atan2(avg_sin, avg_cos)
            
            if not self.is_initialized:
                self.marker_pos = avg_pos
                self.heading_angle = avg_h
                self.is_initialized = True
            else:
                self.marker_pos = self.marker_pos * (1 - self.alpha) + avg_pos * self.alpha
                diff = (avg_h - self.heading_angle + math.pi) % (2 * math.pi) - math.pi
                self.heading_angle += diff * self.alpha
        
        return self.marker_pos, self.heading_angle, self.is_initialized
    
    def get_center_position(self, wc_l, map_scale):
        """휠체어 중심 위치 계산"""
        if self.marker_pos is None:
            return None
        
        center = self.marker_pos + np.array([
            (wc_l/2) * map_scale * math.cos(self.heading_angle), 
            (wc_l/2) * map_scale * math.sin(self.heading_angle)
        ])
        return center