# localization.py
import cv2
import numpy as np
import math


class AngleKalman:
    """
    1D angle Kalman filter (theta, omega)
    - theta wrap(-pi~pi)
    - measurement residual uses shortest angle diff
    """
    def __init__(self, q_theta=2e-3, q_omega=6e-3, r_base=3e-2):
        self.x = np.zeros((2, 1), dtype=np.float32)  # [theta, omega]
        self.P = np.eye(2, dtype=np.float32) * 1.0
        self.q_theta = float(q_theta)
        self.q_omega = float(q_omega)
        self.r_base = float(r_base)
        self.initialized = False

    @staticmethod
    def _wrap(a):
        return (a + math.pi) % (2 * math.pi) - math.pi

    @staticmethod
    def _ang_diff(a, b):
        return (a - b + math.pi) % (2 * math.pi) - math.pi

    def reset(self, theta):
        self.x[:] = 0.0
        self.x[0, 0] = float(theta)
        self.P = np.eye(2, dtype=np.float32) * 1.0
        self.initialized = True

    def predict(self, dt=1.0):
        if not self.initialized:
            return
        dt = float(max(1e-3, dt))
        F = np.array([[1.0, dt],
                      [0.0, 1.0]], dtype=np.float32)
        Q = np.array([[self.q_theta * dt * dt, 0.0],
                      [0.0, self.q_omega * dt]], dtype=np.float32)
        self.x = F @ self.x
        self.x[0, 0] = self._wrap(self.x[0, 0])
        self.P = F @ self.P @ F.T + Q

    def update(self, z_theta, quality=1.0, gate_rad=math.radians(75.0)):
        z_theta = float(z_theta)
        q = float(np.clip(quality, 0.05, 1.0))

        if not self.initialized:
            self.reset(z_theta)
            return True

        y = self._ang_diff(z_theta, float(self.x[0, 0]))

        # gate: 큰 점프 + 낮은 quality는 버림
        if abs(y) > gate_rad and q < 0.35:
            return False

        R = np.array([[self.r_base / (q * q)]], dtype=np.float32)
        H = np.array([[1.0, 0.0]], dtype=np.float32)

        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K * np.array([[y]], dtype=np.float32)
        self.x[0, 0] = self._wrap(self.x[0, 0])
        self.P = (np.eye(2, dtype=np.float32) - K @ H) @ self.P
        return True

    def get(self):
        return float(self.x[0, 0]), float(self.x[1, 0])


class PoseEstimator:
    """
    ArUco 마커 기반 위치/각도 추정 (점프 방지 강화 버전)

    핵심:
    - fisheye undistortPoints
    - solvePnPGeneric(IPPE_SQUARE)로 후보 자세(2개) 중
      '이전 heading에 가까운 해' 선택
    - 품질게이트(면적/가장자리/재투영오차)로 bad 측정 제거
    - heading은 Kalman(각도+각속도) + 게이팅으로 점프 억제
    - 마커가 휠체어 "뒤쪽" 영역에 붙은 상황에 맞춰:
        center_pos = marker_pos + marker_to_center * forward_dir
      (ID0/ID1 모두 동일한 방식)
    - ID1(뒤 마커)은 마커 정면이 뒤를 보게 붙어있으므로 heading +180 보정
    - left가 멀고 가장자리에서 튈 때 rear가 보이면 left를 자동으로 약화
    """

    def __init__(self, K, D, cams, marker_size_cm, alpha,
                 wc_l_cm=100.0, marker_to_center_cm=50.0):
        self.K = K
        self.D = D
        self.cams = cams
        self.marker_size_cm = float(marker_size_cm)

        self.wc_l_cm = float(wc_l_cm)
        self.marker_to_center_cm = float(marker_to_center_cm)

        self.set_alpha(alpha)
        self.pos_alpha_min = 0.03

        # quality gate
        self.min_area_px2 = 450.0
        self.reproj_err_th = 3.8
        self.min_edge_px = 25.0
        self.max_jump_deg = 75.0

        # "rear가 잘 보이면 left 약화" 옵션
        self.auto_left_suppress = True
        self.rear_strong_th = 0.35     # rear total weight가 이 이상이면 "rear strong"
        self.left_suppress_gain = 0.35 # left weight에 곱해지는 억제 배수

        # marker heights (cm)
        self.marker_info = {
            0: {'h_cm': 70.0},
            1: {'h_cm': 56.0},
        }

        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        params.cornerRefinementWinSize = 5
        params.cornerRefinementMaxIterations = 30
        params.cornerRefinementMinAccuracy = 0.01

        self.detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250),
            params
        )

        self.subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)

        # state
        self.center_pos = None
        self.heading_angle = 0.0
        self.is_initialized = False
        self.marker_pos = None

        # heading filter
        self.kf = AngleKalman(q_theta=2e-3, q_omega=6e-3, r_base=3e-2)
        self.use_kalman = True

        # debug
        self.last_det_all = []
        self.last_per_cam_est = {}

    # ---------------- setters ----------------
    def set_alpha(self, alpha):
        a = float(np.clip(alpha, 0.01, 1.0))
        self.pos_alpha = a

    def set_marker_to_center(self, cm):
        self.marker_to_center_cm = float(cm)

    def set_quality_gates(self, min_area_px2=None, reproj_err_th=None, min_edge_px=None, max_jump_deg=None):
        if min_area_px2 is not None:
            self.min_area_px2 = float(min_area_px2)
        if reproj_err_th is not None:
            self.reproj_err_th = float(reproj_err_th)
        if min_edge_px is not None:
            self.min_edge_px = float(min_edge_px)
        if max_jump_deg is not None:
            self.max_jump_deg = float(max_jump_deg)

    def set_heading_filter(self, use_kalman: bool):
        self.use_kalman = bool(use_kalman)

    # ---------------- helpers ----------------
    @staticmethod
    def _ang_diff(a, b):
        return (a - b + math.pi) % (2 * math.pi) - math.pi

    @staticmethod
    def _wrap(a):
        return (a + math.pi) % (2 * math.pi) - math.pi

    def _edge_distance(self, corners, w, h):
        xs = corners[:, 0]
        ys = corners[:, 1]
        d_left = xs.min()
        d_right = (w - 1) - xs.max()
        d_top = ys.min()
        d_bottom = (h - 1) - ys.max()
        return float(min(d_left, d_right, d_top, d_bottom))

    def _reproj_err(self, obj_points, rvec, tvec, undist):
        proj, _ = cv2.projectPoints(obj_points, rvec, tvec.reshape(3, 1), self.K, None)
        e = np.mean(np.linalg.norm(proj.reshape(-1, 2) - undist.reshape(-1, 2), axis=1))
        return float(e)

    def _heading_from_pose(self, rvec, cfg, marker_id):
        """
        rvec -> heading(rad) in "map drawing coordinates"
        - 먼저 compass 기준(북=0)을 만든 다음
        - map rad(0=+x, +y=아래)로 변환: rad = radians((compass + 270) % 360)
        """
        rmat, _ = cv2.Rodrigues(rvec)

        # ✅ yaw (수평 회전) : atan2(r10, r00)
        raw_yaw_deg = math.degrees(math.atan2(float(rmat[1, 0]), float(rmat[0, 0])))

        # 사용자 보정식 유지 (+ yaw_trim은 heading에도 적용)
        compass = (raw_yaw_deg * float(cfg['sens'])) + float(cfg['install_angle']) - float(cfg['install_offset'])
        compass += float(cfg.get('yaw_trim_deg', 0.0))
        compass = compass % 360.0

        # ID1(뒤 마커는 정면이 뒤) => 전방 기준 통일 위해 +180
        if marker_id == 1:
            compass = (compass + 180.0) % 360.0

        # compass(북=0) -> map heading rad (0=+x, +y=아래)
        h_rad = math.radians((compass + 270.0) % 360.0)
        return float(h_rad)

    def _choose_best_pnp(self, obj_points, undist, cfg, marker_id, pred_heading):
        """
        solvePnPGeneric(IPPE_SQUARE) 후보 중:
        - reproj err 낮고
        - pred_heading과 가까운 후보 선택
        """
        candidates = []

        try:
            ok, rvecs, tvecs, reproj = cv2.solvePnPGeneric(
                obj_points, undist, self.K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE
            )
            if ok and rvecs is not None and tvecs is not None:
                for i in range(len(rvecs)):
                    rvec = rvecs[i]
                    tvec = tvecs[i].reshape(-1)
                    err = self._reproj_err(obj_points, rvec, tvec, undist)
                    h = self._heading_from_pose(rvec, cfg, marker_id)
                    candidates.append((err, rvec, tvec, h))
        except Exception:
            candidates = []

        if not candidates:
            # fallback
            try:
                ret, rvec, tvec = cv2.solvePnP(obj_points, undist, self.K, None,
                                               flags=cv2.SOLVEPNP_IPPE_SQUARE)
            except Exception:
                ret, rvec, tvec = cv2.solvePnP(obj_points, undist, self.K, None,
                                               flags=cv2.SOLVEPNP_SQPNP)
            if not ret:
                return None
            tvec = tvec.reshape(-1)
            err = self._reproj_err(obj_points, rvec, tvec, undist)
            h = self._heading_from_pose(rvec, cfg, marker_id)
            candidates.append((err, rvec, tvec, h))

        if pred_heading is None:
            candidates.sort(key=lambda x: x[0])
            return candidates[0]

        best = None
        best_score = 1e9
        for err, rvec, tvec, h in candidates:
            ang = abs(self._ang_diff(h, pred_heading))
            score = (err * 1.2) + (ang * 180.0 / math.pi) * 0.7
            if score < best_score:
                best_score = score
                best = (err, rvec, tvec, h)
        return best

    def _fuse(self, dets):
        if not dets:
            return None
        tw = sum(d["weight"] for d in dets)
        if tw < 1e-6:
            return None
        center = sum(d["center_pos"] * d["weight"] for d in dets) / tw
        s = sum(math.sin(d["heading"]) * d["weight"] for d in dets) / tw
        c = sum(math.cos(d["heading"]) * d["weight"] for d in dets) / tw
        h = math.atan2(s, c)
        return center, float(h), float(tw)

    # ---------------- main ----------------
    def detect_and_estimate(self, frames, dist_gain, apply_smoothing=True):
        det_all = []
        det_by_cam = {k: [] for k in self.cams.keys()}

        ms = self.marker_size_cm
        obj_points = np.array([
            [-ms/2,  ms/2, 0], [ ms/2,  ms/2, 0],
            [ ms/2, -ms/2, 0], [-ms/2, -ms/2, 0]
        ], dtype=np.float32)

        pred_heading = None
        if self.is_initialized:
            pred_heading = float(self.heading_angle)

        # Kalman predict
        if apply_smoothing and self.use_kalman and self.kf.initialized:
            self.kf.predict(dt=1.0)
            pred_heading = self.kf.get()[0]

        # --- detect per cam ---
        for cam_name, frame in frames.items():
            if frame is None or cam_name not in self.cams:
                continue

            cfg = self.cams[cam_name]
            h_img, w_img = frame.shape[:2]

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = self.detector.detectMarkers(gray)
            if ids is None:
                continue

            for i in range(len(ids)):
                mid = int(ids[i][0])
                if mid not in self.marker_info:
                    continue

                # subpix
                try:
                    cv2.cornerSubPix(gray, corners[i], (5, 5), (-1, -1), self.subpix_criteria)
                except Exception:
                    pass

                pts = corners[i].reshape(-1, 2).astype(np.float32)

                area = abs(cv2.contourArea(pts))
                if area < self.min_area_px2:
                    continue

                edge_dist = self._edge_distance(pts, w_img, h_img)
                if edge_dist < self.min_edge_px:
                    continue

                undist = cv2.fisheye.undistortPoints(corners[i].reshape(-1, 1, 2), self.K, self.D, P=self.K)

                best = self._choose_best_pnp(obj_points, undist, cfg, mid, pred_heading)
                if best is None:
                    continue
                err, rvec, tvec, h_rad = best

                if err > self.reproj_err_th:
                    continue

                # distance (cm)
                dist_cm = float(np.linalg.norm(tvec)) * float(dist_gain)
                m_h = float(self.marker_info[mid]['h_cm'])
                dh_cm = abs(float(cfg['h_cm']) - m_h)
                dist_sq_diff = dist_cm**2 - dh_cm**2
                ground_cm = math.sqrt(dist_sq_diff) if dist_sq_diff > 1e-6 else 0.01

                # bearing -> map ray
                bearing_deg = math.degrees(math.atan2(float(tvec[0]), float(tvec[2])))
                ray_deg = float(cfg['map_angle_deg']) + float(cfg.get('yaw_trim_deg', 0.0)) + bearing_deg
                ray_rad = math.radians(ray_deg)

                # marker pos
                m_pos_px = cfg['pos_px'] + np.array([
                    ground_cm * float(cfg['map_scale']) * math.cos(ray_rad),
                    ground_cm * float(cfg['map_scale']) * math.sin(ray_rad)
                ], dtype=np.float32)

                # center pos (marker is on rear area -> always push forward)
                shift_px = float(self.marker_to_center_cm) * float(cfg['map_scale'])
                c_pos_px = m_pos_px + np.array([
                    shift_px * math.cos(h_rad),
                    shift_px * math.sin(h_rad)
                ], dtype=np.float32)

                # weight
                cx = float(np.mean(pts[:, 0]))
                rel_x = (cx - w_img / 2) / (w_img / 2)
                w_center = max(0.08, 1.0 - abs(rel_x))

                w_area = float(np.clip(area / 2000.0, 0.05, 1.0))
                w_edge = float(np.clip(edge_dist / 90.0, 0.05, 1.0))
                w_err = float(1.0 / (1.0 + err * err))
                w_dist = float(1.0 / (1.0 + (ground_cm / 140.0) ** 2))

                cam_base = float(cfg.get("base_weight", 1.0))
                weight = float(max(0.02, cam_base * w_center * w_area * w_edge * w_err * w_dist))

                det = {
                    "cam": cam_name,
                    "marker_id": mid,
                    "marker_pos": m_pos_px,
                    "center_pos": c_pos_px,
                    "heading": float(h_rad),
                    "weight": weight,
                    "dbg": {
                        "ground_cm": float(ground_cm),
                        "bearing_deg": float(bearing_deg),
                        "ray_deg": float(ray_deg),
                        "reproj_err": float(err),
                        "area": float(area),
                        "edge_dist": float(edge_dist),
                    }
                }
                det_all.append(det)
                det_by_cam[cam_name].append(det)

        # ===== auto suppress left when rear is strong =====
        if self.auto_left_suppress and det_all:
            rear_w = sum(d["weight"] for d in det_all if d["cam"] == "cam0")
            left_w = sum(d["weight"] for d in det_all if d["cam"] == "cam1")

            # rear가 충분히 강하면 left는 억제 (너가 말한 상황에 최적)
            if rear_w >= self.rear_strong_th and left_w > 0:
                for d in det_all:
                    if d["cam"] == "cam1":
                        d["weight"] *= self.left_suppress_gain
                # det_by_cam도 weight 반영된 det 참조라 그대로 OK

        # per-cam (debug)
        per_cam_est = {cam: self._fuse(lst) for cam, lst in det_by_cam.items()}
        self.last_per_cam_est = per_cam_est
        self.last_det_all = det_all

        fused_raw = self._fuse(det_all)
        if fused_raw is None:
            return self.center_pos, self.heading_angle, self.is_initialized, per_cam_est, det_all

        meas_center, meas_heading, total_w = fused_raw
        quality = float(np.clip(total_w, 0.05, 1.0))

        if not self.is_initialized:
            self.center_pos = meas_center.astype(np.float32)
            self.heading_angle = float(meas_heading)
            self.is_initialized = True
            self.kf.reset(self.heading_angle)
        else:
            # position EMA (quality low -> very slow)
            a_pos = max(self.pos_alpha_min, self.pos_alpha * quality) if apply_smoothing else 1.0
            self.center_pos = self.center_pos * (1.0 - a_pos) + meas_center * a_pos

            # heading
            if apply_smoothing and self.use_kalman:
                self.kf.update(meas_heading, quality=quality, gate_rad=math.radians(self.max_jump_deg))
                self.heading_angle = self.kf.get()[0]
            else:
                diff = self._ang_diff(meas_heading, self.heading_angle)
                a_ang = max(0.05, 0.35 * quality) if apply_smoothing else 1.0
                self.heading_angle = self._wrap(self.heading_angle + diff * a_ang)

        self.marker_pos = self.center_pos
        return self.center_pos, self.heading_angle, self.is_initialized, per_cam_est, det_all

    def get_center_position(self, wc_l, map_scale):
        return self.center_pos
