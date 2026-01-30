import cv2
import numpy as np
import math

class PoseEstimator:
    """ArUco 마커 기반 위치 및 방향 추정 (각도 튐 방지 버전)"""

    def __init__(self, K, D, cams, marker_size, alpha):
        self.K = K
        self.D = D
        self.cams = cams
        self.marker_size = marker_size

        # ===== 튐 방지 파라미터 =====
        # alpha는 0~1, 작을수록 더 부드러움. (main에서 0.2~0.35 추천)
        self.pos_alpha = float(alpha)
        self.ang_alpha = float(alpha)

        self.reproj_err_th = 4.0      # px: 재투영오차 이 이상이면 버림 (3~6 사이 튜닝)
        self.min_area_px2 = 300.0     # 마커 면적 너무 작으면 버림
        self.max_outlier_deg = 60.0   # 이전 헤딩과 60도 이상 차이나면 outlier로 버림
        self.flip_guard = True        # 180 플립 방지

        # ID별 마커 물리 데이터
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

    # --------- helpers ---------
    def _ang_diff(self, a, b):
        """a-b 를 [-pi, pi]로"""
        return (a - b + math.pi) % (2 * math.pi) - math.pi

    def _choose_flip_near_prev(self, h):
        """h 와 h+pi 중 이전 heading에 더 가까운 것 선택 (180 플립 방지)"""
        if not self.is_initialized:
            return h
        h2 = (h + math.pi) % (2 * math.pi)
        d1 = abs(self._ang_diff(h, self.heading_angle))
        d2 = abs(self._ang_diff(h2, self.heading_angle))
        return h2 if d2 < d1 else h

    def _reproj_err(self, obj_points, rvec, tvec, undist):
        proj, _ = cv2.projectPoints(obj_points, rvec, tvec.reshape(3, 1), self.K, None)
        e = np.mean(np.linalg.norm(proj.reshape(-1, 2) - undist.reshape(-1, 2), axis=1))
        return float(e)

    def detect_and_estimate(self, frames, current_dist_gain):
        detected_data = []

        for cam_name, frame in frames.items():
            if frame is None or cam_name not in self.cams:
                continue

            cfg = self.cams[cam_name]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = self.detector.detectMarkers(gray)

            if ids is None:
                continue

            for i in range(len(ids)):
                mid = int(ids[i][0])
                if mid not in self.marker_info:
                    continue

                # 1) 서브픽셀
                cv2.cornerSubPix(gray, corners[i], (5, 5), (-1, -1), self.subpix_criteria)
                c_pts = corners[i].astype(np.int32).reshape((-1, 2))
                cv2.polylines(frame, [c_pts], True, (0, 255, 0), 2)

                # (품질) 면적 체크
                area = abs(cv2.contourArea(corners[i].reshape(-1, 2).astype(np.float32)))
                if area < self.min_area_px2:
                    continue

                # 2) 왜곡 보정
                pts_2d = corners[i].reshape(-1, 1, 2)
                undist = cv2.fisheye.undistortPoints(pts_2d, self.K, self.D, P=self.K)

                ms = self.marker_size
                obj_points = np.array([
                    [-ms/2,  ms/2, 0], [ ms/2,  ms/2, 0],
                    [ ms/2, -ms/2, 0], [-ms/2, -ms/2, 0]
                ], dtype=np.float32)

                # 3) solvePnP (평면 마커면 IPPE_SQUARE가 보통 더 안정)
                try:
                    ret, rvec, tvec = cv2.solvePnP(obj_points, undist, self.K, None,
                                                   flags=cv2.SOLVEPNP_IPPE_SQUARE)
                except Exception:
                    ret, rvec, tvec = cv2.solvePnP(obj_points, undist, self.K, None,
                                                   flags=cv2.SOLVEPNP_SQPNP)

                if not ret:
                    continue

                tvec = tvec.flatten()

                # (품질) 재투영 오차 체크
                err = self._reproj_err(obj_points, rvec, tvec, undist)
                if err > self.reproj_err_th:
                    continue

                # 4) 거리
                dist_cm = np.linalg.norm(tvec) * current_dist_gain
                m_h = self.marker_info[mid]['h_cm']
                dh_cm = abs(cfg['h_cm'] - m_h)
                dist_sq_diff = dist_cm**2 - dh_cm**2
                ground_cm = math.sqrt(dist_sq_diff) if dist_sq_diff > 0 else 0.01

                # 5) 각도 (당신 기존 방식 유지)
                rmat, _ = cv2.Rodrigues(rvec)
                sy = math.sqrt(rmat[0,0]**2 + rmat[1,0]**2)
                raw_yaw = math.atan2(-rmat[2,0], sy) * 180.0 / math.pi

                final_yaw_compass = (raw_yaw * cfg['sens']) + cfg['install_angle'] - cfg['install_offset']
                if mid == 1:
                    final_yaw_compass = (final_yaw_compass + 180.0) % 360.0

                h_rad = math.radians((final_yaw_compass + 270.0) % 360.0)

                # (핵심) 180 플립 방지: 이전 heading과 가까운 해 선택
                if self.flip_guard:
                    h_rad = self._choose_flip_near_prev(h_rad)

                # (추가) 이전 heading에서 너무 멀면 outlier로 버리기
                if self.is_initialized:
                    if abs(self._ang_diff(h_rad, self.heading_angle)) > math.radians(self.max_outlier_deg):
                        continue

                # 6) 글로벌 투영(위치)
                bearing_deg = math.degrees(math.atan2(tvec[0], tvec[2]))
                ray_deg = cfg['map_angle_deg'] + cfg.get('yaw_trim_deg', 0) + bearing_deg
                ray_rad = math.radians(ray_deg)

                m_pos_px = cfg['pos_px'] + np.array([
                    ground_cm * cfg['map_scale'] * math.cos(ray_rad),
                    ground_cm * cfg['map_scale'] * math.sin(ray_rad)
                ])

                shift_px = (100.0 / 2.0) * cfg['map_scale']
                if mid == 0:
                    c_pos_px = m_pos_px - np.array([shift_px * math.cos(h_rad), shift_px * math.sin(h_rad)])
                else:
                    c_pos_px = m_pos_px + np.array([shift_px * math.cos(h_rad), shift_px * math.sin(h_rad)])

                # 7) 가중치 (중앙 + reproj + 면적)
                center_w = max(0.05, 1.0 - abs((np.mean(corners[i][:, 0, 0]) - frame.shape[1]/2) / (frame.shape[1]/2)))
                err_w = 1.0 / (1.0 + err * err)
                area_w = min(1.0, area / 1500.0)   # 면적 커질수록 가중↑ (대충 스케일)
                weight = center_w * err_w * area_w

                detected_data.append((c_pos_px, h_rad, weight))

                # 디버그
                cv2.putText(frame, f"ID:{mid} Dist:{dist_cm/100.0:.2f}m err:{err:.1f}px",
                            (c_pts[0][0], c_pts[0][1]-10), 0, 0.5, (0,255,0), 2)

        # 8) 통합 + 필터
        if len(detected_data) > 0:
            total_w = sum(p[2] for p in detected_data)
            avg_pos = sum(p[0] * p[2] for p in detected_data) / max(total_w, 1e-6)
            avg_sin = sum(math.sin(p[1]) * p[2] for p in detected_data) / max(total_w, 1e-6)
            avg_cos = sum(math.cos(p[1]) * p[2] for p in detected_data) / max(total_w, 1e-6)
            avg_h = math.atan2(avg_sin, avg_cos)

            if not self.is_initialized:
                self.marker_pos = avg_pos
                self.heading_angle = avg_h
                self.is_initialized = True
            else:
                # 위치 smoothing
                self.marker_pos = self.marker_pos * (1 - self.pos_alpha) + avg_pos * self.pos_alpha
                # 각도 smoothing (wrap 보정)
                diff = self._ang_diff(avg_h, self.heading_angle)
                self.heading_angle += diff * self.ang_alpha

        return self.marker_pos, self.heading_angle, self.is_initialized

    def get_center_position(self, wc_l, map_scale):
        if self.marker_pos is None:
            return None
        return self.marker_pos
