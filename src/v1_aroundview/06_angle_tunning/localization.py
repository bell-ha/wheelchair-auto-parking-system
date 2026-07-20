# localization.py
import cv2
import numpy as np
import math


class PoseEstimator:
    """
    ArUco 마커 기반 위치 및 방향 추정 (cam별/전체 fused + 튐 방지 + 스무딩)
    - cam별 추정치(센터/헤딩) + 전체 fused(센터/헤딩) 둘 다 제공
    - trackbar에서 cams 파라미터를 바꾸면 즉시 반영 (cams dict를 참조)
    """

    def __init__(self, K, D, cams, marker_size_cm, alpha, wc_l_cm=100.0):
        self.K = K
        self.D = D
        self.cams = cams                  # main에서 넘겨준 dict를 그대로 참조
        self.marker_size_cm = float(marker_size_cm)
        self.wc_l_cm = float(wc_l_cm)

        # ===== 튐 방지/스무딩 파라미터 =====
        self.set_alpha(alpha)             # pos_alpha, ang_alpha 동기화

        self.reproj_err_th = 4.0          # px: 재투영 오차 임계치
        self.min_area_px2 = 300.0         # 마커 면적 너무 작으면 버림
        self.max_outlier_deg = 60.0       # 이전 헤딩과 이 이상 차이면 outlier
        self.flip_guard = True            # 180 플립 방지

        # ID별 마커 물리 정보
        self.marker_info = {
            0: {'h_cm': 70.0},
            1: {'h_cm': 56.0},
        }

        self.detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250),
            cv2.aruco.DetectorParameters()
        )

        # 상태(센터 기준으로 유지)
        self.center_pos = None
        self.heading_angle = 0.0
        self.is_initialized = False

        # 기존 코드 호환용
        self.marker_pos = None

        # 디버그 저장
        self.last_det_all = []
        self.last_det_by_cam = {}
        self.last_per_cam_est = {}
        self.last_fused_raw = None

        self.subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)

    # ---------- public setters ----------
    def set_alpha(self, alpha):
        a = float(alpha)
        a = max(0.01, min(1.0, a))
        self.pos_alpha = a
        self.ang_alpha = a

    def set_wheelchair_length(self, wc_l_cm):
        self.wc_l_cm = float(wc_l_cm)

    # ---------- helpers ----------
    @staticmethod
    def _ang_diff(a, b):
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

    def _fuse_estimates(self, det_list):
        if not det_list:
            return None
        total_w = sum(d["weight"] for d in det_list)
        if total_w <= 1e-6:
            return None

        avg_center = sum(d["center_pos"] * d["weight"] for d in det_list) / total_w
        avg_sin = sum(math.sin(d["heading"]) * d["weight"] for d in det_list) / total_w
        avg_cos = sum(math.cos(d["heading"]) * d["weight"] for d in det_list) / total_w
        avg_heading = math.atan2(avg_sin, avg_cos)
        return avg_center, avg_heading, float(total_w)

    # ---------- main ----------
    def detect_and_estimate(self, frames, dist_gain, apply_smoothing=True):
        det_all = []
        det_by_cam = {k: [] for k in self.cams.keys()}

        # object points (cm)
        ms = self.marker_size_cm
        obj_points = np.array([
            [-ms/2,  ms/2, 0], [ ms/2,  ms/2, 0],
            [ ms/2, -ms/2, 0], [-ms/2, -ms/2, 0]
        ], dtype=np.float32)

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
                try:
                    cv2.cornerSubPix(gray, corners[i], (5, 5), (-1, -1), self.subpix_criteria)
                except Exception:
                    pass

                c_pts = corners[i].astype(np.int32).reshape((-1, 2))
                cv2.polylines(frame, [c_pts], True, (0, 255, 0), 2)

                # (품질) 면적 체크
                area = abs(cv2.contourArea(corners[i].reshape(-1, 2).astype(np.float32)))
                if area < self.min_area_px2:
                    continue

                # 2) 왜곡 보정
                pts_2d = corners[i].reshape(-1, 1, 2)
                undist = cv2.fisheye.undistortPoints(pts_2d, self.K, self.D, P=self.K)

                # 3) solvePnP
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

                # 4) 거리 (tvec 단위=cm)
                dist_cm = float(np.linalg.norm(tvec)) * float(dist_gain)
                m_h = float(self.marker_info[mid]['h_cm'])
                dh_cm = abs(float(cfg['h_cm']) - m_h)
                dist_sq_diff = dist_cm**2 - dh_cm**2
                ground_cm = math.sqrt(dist_sq_diff) if dist_sq_diff > 1e-6 else 0.01

                # 5) 헤딩
                rmat, _ = cv2.Rodrigues(rvec)
                sy = math.sqrt(rmat[0, 0]**2 + rmat[1, 0]**2)
                raw_yaw = math.atan2(-rmat[2, 0], sy) * 180.0 / math.pi

                final_yaw_compass = (raw_yaw * float(cfg['sens'])) + float(cfg['install_angle']) - float(cfg['install_offset'])
                if mid == 1:
                    final_yaw_compass = (final_yaw_compass + 180.0) % 360.0

                h_rad = math.radians((final_yaw_compass + 270.0) % 360.0)

                # 180 플립 방지
                if self.flip_guard:
                    h_rad = self._choose_flip_near_prev(h_rad)

                # outlier 제거
                if self.is_initialized:
                    if abs(self._ang_diff(h_rad, self.heading_angle)) > math.radians(self.max_outlier_deg):
                        continue

                # 6) 글로벌 투영(마커 위치)
                bearing_deg = math.degrees(math.atan2(tvec[0], tvec[2]))
                ray_deg = float(cfg['map_angle_deg']) + float(cfg.get('yaw_trim_deg', 0.0)) + bearing_deg
                ray_rad = math.radians(ray_deg)

                m_pos_px = cfg['pos_px'] + np.array([
                    ground_cm * float(cfg['map_scale']) * math.cos(ray_rad),
                    ground_cm * float(cfg['map_scale']) * math.sin(ray_rad)
                ], dtype=np.float32)

                # 7) 마커->센터 변환
                shift_px = (self.wc_l_cm / 2.0) * float(cfg['map_scale'])
                dx = shift_px * math.cos(h_rad)
                dy = shift_px * math.sin(h_rad)

                if mid == 0:  # front marker
                    c_pos_px = m_pos_px - np.array([dx, dy], dtype=np.float32)
                else:         # rear marker
                    c_pos_px = m_pos_px + np.array([dx, dy], dtype=np.float32)

                # 8) 가중치
                cx = float(np.mean(corners[i][:, 0, 0]))
                rel_x = (cx - frame.shape[1] / 2) / (frame.shape[1] / 2)
                w_center = max(0.05, 1.0 - abs(rel_x))

                err_w = 1.0 / (1.0 + err * err)
                area_w = min(1.0, area / 1500.0)
                dist_w = 1.0 / (1.0 + ground_cm / 200.0)
                weight = float(max(0.02, w_center * err_w * area_w * dist_w))

                det = {
                    "cam": cam_name,
                    "marker_id": mid,
                    "marker_pos": m_pos_px,
                    "center_pos": c_pos_px,
                    "heading": h_rad,
                    "weight": weight,
                    "dbg": {
                        "ground_cm": float(ground_cm),
                        "bearing_deg": float(bearing_deg),
                        "ray_deg": float(ray_deg),
                        "reproj_err": float(err)
                    }
                }
                det_all.append(det)
                det_by_cam[cam_name].append(det)

                cv2.putText(frame,
                            f"{cam_name} ID:{mid} g:{ground_cm/100.0:.2f}m err:{err:.1f}px w:{weight:.2f}",
                            (c_pts[0][0], c_pts[0][1] - 10),
                            0, 0.5, (0, 255, 255), 2)

        per_cam_est = {cam: self._fuse_estimates(lst) for cam, lst in det_by_cam.items()}
        fused_raw = self._fuse_estimates(det_all)

        self.last_det_all = det_all
        self.last_det_by_cam = det_by_cam
        self.last_per_cam_est = per_cam_est
        self.last_fused_raw = fused_raw

        if fused_raw is not None:
            fused_center, fused_heading, _ = fused_raw

            if (not self.is_initialized) or (not apply_smoothing):
                self.center_pos = fused_center.astype(np.float32)
                self.heading_angle = float(fused_heading)
                self.is_initialized = True
            else:
                self.center_pos = self.center_pos * (1.0 - self.pos_alpha) + fused_center * self.pos_alpha
                diff = self._ang_diff(fused_heading, self.heading_angle)
                self.heading_angle = self.heading_angle + diff * self.ang_alpha

            self.marker_pos = self.center_pos

        return self.center_pos, self.heading_angle, self.is_initialized, per_cam_est, det_all

    def get_center_position(self, wc_l, map_scale):
        return self.center_pos
