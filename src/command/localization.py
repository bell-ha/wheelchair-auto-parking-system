import cv2
import numpy as np
import math


class PoseEstimator:
    """
    ArUco 마커 기반 위치 및 방향 추정 모듈 (개선 버전)
    - 180도 플립 방지
    - Outlier 제거
    - 품질 필터링 (재투영 오차, 마커 면적)
    - 서브픽셀 정확도
    - 개선된 가중치 시스템
    - cam별 추정치 제공
    """
    
    def __init__(self, K, D, cams, marker_size, marker_h, dist_gain, alpha):
        self.K = K
        self.D = D
        self.cams = cams
        self.marker_size = marker_size
        self.marker_h = marker_h
        self.dist_gain = dist_gain
        
        # ===== 튐 방지/스무딩 파라미터 =====
        self.set_alpha(alpha)
        
        # ===== 품질 필터링 파라미터 =====
        self.reproj_err_th = 4.0          # px: 재투영 오차 임계치
        self.min_area_px2 = 300.0         # 마커 면적 최소값
        self.max_outlier_deg = 60.0       # 이전 헤딩과 최대 허용 차이
        self.flip_guard = True            # 180도 플립 방지 활성화
        
        # ArUco 검출기
        self.detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250), 
            cv2.aruco.DetectorParameters()
        )
        
        # 상태 변수
        self.marker_pos = None
        self.heading_angle = 0.0
        self.is_initialized = False
        
        # 디버그용 - 마지막 검출 데이터
        self.last_det_all = []
        self.last_det_by_cam = {}
        self.last_per_cam_est = {}
        self.last_fused_raw = None
        
        # 서브픽셀 정확도 개선용 파라미터
        self.subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
    
    # ---------- public setters ----------
    def set_alpha(self, alpha):
        """스무딩 계수 설정"""
        a = float(alpha)
        a = max(0.01, min(1.0, a))
        self.pos_alpha = a
        self.ang_alpha = a
    
    # ---------- helpers ----------
    @staticmethod
    def _ang_diff(a, b):
        """각도 차이를 [-pi, pi] 범위로 정규화"""
        return (a - b + math.pi) % (2 * math.pi) - math.pi
    
    def _choose_flip_near_prev(self, h):
        """
        180도 플립 방지: h와 h+pi 중 이전 heading에 더 가까운 것 선택
        """
        if not self.is_initialized:
            return h
        
        h2 = (h + math.pi) % (2 * math.pi)
        d1 = abs(self._ang_diff(h, self.heading_angle))
        d2 = abs(self._ang_diff(h2, self.heading_angle))
        
        return h2 if d2 < d1 else h
    
    def _reproj_err(self, obj_points, rvec, tvec, undist):
        """재투영 오차 계산"""
        proj, _ = cv2.projectPoints(obj_points, rvec, tvec.reshape(3, 1), self.K, None)
        e = np.mean(np.linalg.norm(proj.reshape(-1, 2) - undist.reshape(-1, 2), axis=1))
        return float(e)
    
    def _fuse_estimates(self, det_list):
        """
        가중치 기반 추정치 융합
        - 위치: 가중 평균
        - 각도: sin/cos 가중 평균으로 처리 (각도 wrapping 문제 해결)
        """
        if not det_list:
            return None
        
        total_w = sum(d["weight"] for d in det_list)
        if total_w <= 1e-6:
            return None
        
        # 위치 융합
        avg_pos = sum(d["pos"] * d["weight"] for d in det_list) / total_w
        
        # 각도 융합 (sin/cos 평균)
        avg_sin = sum(math.sin(d["heading"]) * d["weight"] for d in det_list) / total_w
        avg_cos = sum(math.cos(d["heading"]) * d["weight"] for d in det_list) / total_w
        avg_heading = math.atan2(avg_sin, avg_cos)
        
        return avg_pos, avg_heading, float(total_w)
    
    def detect_and_estimate(self, frames, apply_smoothing=True):
        """
        프레임에서 마커를 감지하고 위치/방향 추정
        
        Args:
            frames: {camera_name: frame} 딕셔너리
            apply_smoothing: 스무딩 적용 여부
        
        Returns:
            (marker_pos, heading_angle, is_initialized, per_cam_est, det_all) 튜플
            - per_cam_est: 카메라별 추정치
            - det_all: 모든 검출 데이터 리스트
        """
        det_all = []
        det_by_cam = {k: [] for k in self.cams.keys()}
        
        # 3D 마커 객체 좌표
        ms = self.marker_size
        obj_points = np.array([
            [-ms/2,  ms/2, 0],
            [ ms/2,  ms/2, 0],
            [ ms/2, -ms/2, 0],
            [-ms/2, -ms/2, 0]
        ], dtype=np.float32)
        
        for cam_name, frame in frames.items():
            if frame is None or cam_name not in self.cams:
                continue
            
            cfg = self.cams[cam_name]
            
            # 그레이스케일 변환
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 마커 검출
            corners, ids, _ = self.detector.detectMarkers(gray)
            
            if ids is None:
                continue
            
            for i in range(len(ids)):
                marker_id = int(ids[i][0])
                
                # ===== 1. 서브픽셀 정확도 개선 =====
                try:
                    cv2.cornerSubPix(gray, corners[i], (5, 5), (-1, -1), self.subpix_criteria)
                except Exception:
                    pass
                
                # 시각화용 - 마커 테두리 그리기
                c_pts = corners[i].astype(np.int32).reshape((-1, 2))
                cv2.polylines(frame, [c_pts], True, (0, 255, 0), 2)
                
                # ===== 2. 품질 체크: 마커 면적 =====
                area = abs(cv2.contourArea(corners[i].reshape(-1, 2).astype(np.float32)))
                if area < self.min_area_px2:
                    continue
                
                # ===== 3. 렌즈 왜곡 보정 =====
                pts_2d = corners[i].reshape(-1, 1, 2)
                undist = cv2.fisheye.undistortPoints(pts_2d, self.K, self.D, P=self.K)
                
                # ===== 4. SolvePnP =====
                try:
                    ret, rvec, tvec = cv2.solvePnP(
                        obj_points, undist, self.K, None,
                        flags=cv2.SOLVEPNP_IPPE_SQUARE
                    )
                except Exception:
                    ret, rvec, tvec = cv2.solvePnP(
                        obj_points, undist, self.K, None,
                        flags=cv2.SOLVEPNP_SQPNP
                    )
                
                if not ret:
                    continue
                
                tvec = tvec.flatten()
                
                # ===== 5. 품질 체크: 재투영 오차 =====
                err = self._reproj_err(obj_points, rvec, tvec, undist)
                if err > self.reproj_err_th:
                    continue
                
                # ===== 6. 거리 계산 =====
                d_raw = np.linalg.norm(tvec)
                
                # 거리 보정 적용
                d = d_raw * (1 + (self.dist_gain - 1) * (d_raw / 500))
                
                # 바닥 거리 계산
                dh = abs(cfg['h'] - self.marker_h)
                dist_sq_diff = d**2 - dh**2
                ground_d = math.sqrt(dist_sq_diff) if dist_sq_diff > 1e-6 else 0.01
                
                # ===== 7. 헤딩(Yaw) 계산 =====
                rmat, _ = cv2.Rodrigues(rvec)
                sy = math.sqrt(rmat[0, 0]**2 + rmat[1, 0]**2)
                
                if sy < 1e-6:
                    local_yaw = math.atan2(-rmat[1, 2], rmat[1, 1])
                else:
                    local_yaw = math.atan2(rmat[1, 0], rmat[0, 0])
                
                # 카메라 글로벌 각도
                cam_global_angle = math.radians(cfg['map_angle'] + cfg['yaw'])
                
                # 헤딩 계산
                h_rad = cam_global_angle + local_yaw + math.pi
                if marker_id == 1:
                    h_rad += math.pi
                
                # ===== 8. 180도 플립 방지 =====
                if self.flip_guard:
                    h_rad = self._choose_flip_near_prev(h_rad)
                
                # ===== 9. Outlier 제거 =====
                if self.is_initialized:
                    if abs(self._ang_diff(h_rad, self.heading_angle)) > math.radians(self.max_outlier_deg):
                        continue
                
                # ===== 10. 글로벌 좌표 변환 =====
                ray_angle = math.atan2(tvec[0], tvec[2])
                t_rad = cam_global_angle + ray_angle
                
                pos = cfg['pos'] + np.array([
                    ground_d * cfg.get('map_scale', 0.5) * math.cos(t_rad),
                    ground_d * cfg.get('map_scale', 0.5) * math.sin(t_rad)
                ], dtype=np.float32)
                
                # ===== 11. 개선된 가중치 계산 =====
                # (a) 화면 중앙에 가까울수록 높은 가중치
                cx = float(np.mean(corners[i][:, 0, 0]))
                rel_x = (cx - frame.shape[1] / 2) / (frame.shape[1] / 2)
                w_center = max(0.05, 1.0 - abs(rel_x))
                
                # (b) 재투영 오차가 작을수록 높은 가중치
                err_w = 1.0 / (1.0 + err * err)
                
                # (c) 마커 면적이 클수록 높은 가중치
                area_w = min(1.0, area / 1500.0)
                
                # (d) 거리가 가까울수록 높은 가중치
                dist_w = 1.0 / (1.0 + ground_d / 200.0)
                
                # 종합 가중치
                weight = float(max(0.02, w_center * err_w * area_w * dist_w))
                
                # ===== 12. 검출 데이터 저장 =====
                det = {
                    "cam": cam_name,
                    "marker_id": marker_id,
                    "pos": pos,
                    "heading": h_rad,
                    "weight": weight,
                    "dbg": {
                        "ground_d": float(ground_d),
                        "ray_angle": float(math.degrees(ray_angle)),
                        "reproj_err": float(err),
                        "area": float(area)
                    }
                }
                det_all.append(det)
                det_by_cam[cam_name].append(det)
                
                # 시각화용 - 정보 표시
                cv2.putText(frame,
                            f"{cam_name} ID:{marker_id} d:{ground_d/100.0:.2f}m err:{err:.1f}px w:{weight:.2f}",
                            (c_pts[0][0], c_pts[0][1] - 10),
                            0, 0.5, (0, 255, 255), 2)
        
        # ===== 13. 카메라별 추정치 계산 =====
        per_cam_est = {cam: self._fuse_estimates(lst) for cam, lst in det_by_cam.items()}
        
        # ===== 14. 전체 융합 추정치 계산 =====
        fused_raw = self._fuse_estimates(det_all)
        
        # 디버그 데이터 저장
        self.last_det_all = det_all
        self.last_det_by_cam = det_by_cam
        self.last_per_cam_est = per_cam_est
        self.last_fused_raw = fused_raw
        
        # ===== 15. 상태 업데이트 (스무딩 적용) =====
        if fused_raw is not None:
            fused_pos, fused_heading, _ = fused_raw
            
            if (not self.is_initialized) or (not apply_smoothing):
                # 초기화 또는 스무딩 비활성화
                self.marker_pos = fused_pos.astype(np.float32)
                self.heading_angle = float(fused_heading)
                self.is_initialized = True
            else:
                # EMA 필터 적용
                # 위치 스무딩
                self.marker_pos = self.marker_pos * (1.0 - self.pos_alpha) + fused_pos * self.pos_alpha
                
                # 각도 스무딩 (wrapping 고려)
                diff = self._ang_diff(fused_heading, self.heading_angle)
                self.heading_angle = self.heading_angle + diff * self.ang_alpha
        
        return self.marker_pos, self.heading_angle, self.is_initialized, per_cam_est, det_all
    
    def get_center_position(self, wc_l, map_scale):
        """휠체어 중심 위치 계산"""
        if self.marker_pos is None:
            return None
        
        center = self.marker_pos + np.array([
            (wc_l/2) * map_scale * math.cos(self.heading_angle), 
            (wc_l/2) * map_scale * math.sin(self.heading_angle)
        ])
        return center
    
    def get_debug_info(self):
        """디버그 정보 반환"""
        return {
            "det_all": self.last_det_all,
            "det_by_cam": self.last_det_by_cam,
            "per_cam_est": self.last_per_cam_est,
            "fused_raw": self.last_fused_raw
        }