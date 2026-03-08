#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# dual_rear_jitter_aware_viewer.py
#
# 역할 분리:
#   ID0 (83cm, 상단) → 주 위치 추정
#   ID1 (53cm, 하단) → jittering 감지용
#
# 동작:
#   ID0만 보임           → quality=1.0 으로 KF update
#   ID0+ID1 둘 다 보임   → 두 추정값 위치 차이 계산
#                            차이 < JITTER_THRESH_CM  → 정상, quality=1.2 (두 마커 동의)
#                            차이 >= JITTER_THRESH_CM → jittering, quality=0.3
#   ID1만 보임           → ID0 없으므로 quality=0.5 로 KF update (신뢰도 낮음)
#   둘 다 없음           → KF predict만
#
#   rear 카메라 우선: rear quality *= 1.5, left quality *= 0.7
#   3초 이상 미검출 → 화살표 숨김
#
# 키: Q/ESC=종료   [/] = jitter 임계값 조절

import math, time
from dataclasses import dataclass

import cv2
import numpy as np

# =======================
# SETTINGS
# =======================
DIST_GAIN_FIXED   = 0.90
CENTER_OFFSET_CM  = 23.0
MARKER_H_CM       = {0: 83.0, 1: 53.0}
HIDE_TIMEOUT_S    = 3.0
JITTER_THRESH_CM  = 15.0   # [ / ] 키로 조절

CAM_TRUST = {"rear": 1.5, "left": 0.7}   # rear 카메라 신뢰 배율

# =======================
# ArUco Board 오브젝트 포인트
# =======================
MARKER_SIZE_M = 0.25
HALF          = MARKER_SIZE_M / 2
ID1_Y         = -(MARKER_H_CM[0] - MARKER_H_CM[1]) / 100.0   # -0.30m

def _corners(cx, cy):
    return np.array([
        [cx-HALF, cy+HALF, 0.],
        [cx+HALF, cy+HALF, 0.],
        [cx+HALF, cy-HALF, 0.],
        [cx-HALF, cy-HALF, 0.],
    ], dtype=np.float32)

BOARD_OBJ = {
    0: _corners(0., 0.),      # 주 추정용
    1: _corners(0., ID1_Y),   # jitter 감지용
}

aruco_dict   = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
aruco_params = cv2.aruco.DetectorParameters()
detector     = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

# =======================
# Camera intrinsics
# =======================
K = np.array([[601.71923257, 0.,          630.47700714],
              [0.,          601.34529853, 367.21223657],
              [0.,          0.,           1.          ]], dtype=np.float32)
D = np.array([-0.18495647, 0.02541005, -0.01068433, 0.00321714], dtype=np.float32)

# =======================
# Helpers
# =======================
def wrap360(d):
    d = d % 360.
    return d + 360. if d < 0 else d

def wrap_pi(r):
    return (r + math.pi) % (2*math.pi) - math.pi

def compass_to_map(deg):
    return math.radians((deg + 270.) % 360.)

# =======================
# CamCfg
# =======================
@dataclass
class CamCfg:
    key:            str
    index:          int
    pos_world_px:   np.ndarray
    h_cm:           float
    map_angle_deg:  float
    sens:           float
    install_angle:  float
    install_offset: float
    yaw_trim_deg:   float = 0.
    dist_gain:      float = DIST_GAIN_FIXED

# =======================
# 단일 마커 PnP (역할 구분용)
# =======================
def single_marker_pnp(corners_i, cam: CamCfg, marker_id: int):
    """마커 하나로 위치·방향 추정. 반환: (center_world, heading_rad, reproj) or None"""
    und = cv2.fisheye.undistortPoints(
        corners_i.reshape(-1,1,2), K, D, P=K
    ).reshape(-1,2).astype(np.float32)

    ok, rvec, tvec = cv2.solvePnP(
        BOARD_OBJ[marker_id], und, K, None,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        return None
    tvec = tvec.reshape(3).astype(np.float32)
    if float(tvec[2]) <= 0.01:
        return None

    dist_m    = float(np.linalg.norm(tvec))
    dh_m      = abs(cam.h_cm - MARKER_H_CM[marker_id]) / 100.
    ground_m  = math.sqrt(max(0., dist_m**2 - dh_m**2))
    ground_cm = ground_m * 100. * cam.dist_gain

    bearing = math.degrees(math.atan2(float(tvec[0]), float(tvec[2])))
    ray_rad = math.radians(cam.map_angle_deg + cam.yaw_trim_deg + bearing)

    mw = cam.pos_world_px + np.array([
        ground_cm * math.cos(ray_rad),
        ground_cm * math.sin(ray_rad),
    ], dtype=np.float32)

    rmat, _ = cv2.Rodrigues(rvec)
    sy  = math.sqrt(rmat[0,0]**2 + rmat[1,0]**2)
    yaw = math.degrees(math.atan2(-rmat[2,0], sy))
    yc  = wrap360((yaw * cam.sens) + cam.install_angle
                  - cam.install_offset + 180.)
    hdg = compass_to_map(yc)

    center = mw + np.array([
        CENTER_OFFSET_CM * math.cos(hdg),
        CENTER_OFFSET_CM * math.sin(hdg),
    ], dtype=np.float32)

    proj, _ = cv2.projectPoints(BOARD_OBJ[marker_id], rvec, tvec, K, None)
    reproj   = float(np.mean(np.linalg.norm(
        proj.reshape(-1,2) - und, axis=1)))

    return {"center": center, "heading": hdg, "reproj": reproj}

# =======================
# 프레임에서 ID0/ID1 각각 추정
# =======================
def estimate_frame(frame_bgr, cam: CamCfg):
    """반환: {"id0": ..., "id1": ...}  각각 None 가능"""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    result = {"id0": None, "id1": None}
    if ids is None:
        return result
    for i, mid_arr in enumerate(ids):
        mid = int(mid_arr[0])
        if mid not in (0, 1):
            continue
        est = single_marker_pnp(corners[i], cam, mid)
        if est is None:
            continue
        key = "id0" if mid == 0 else "id1"
        # 같은 ID 여러 개 검출 시 reproj 낮은 것 선택
        if result[key] is None or est["reproj"] < result[key]["reproj"]:
            result[key] = est
    return result

# =======================
# jitter 감지 + quality 결정
# =======================
def assess_quality(id0_est, id1_est, jitter_thresh_cm, cam_key):
    """
    id0 기반 추정값과 quality를 반환.
    id0 없으면 id1 폴백.
    """
    cam_mul = CAM_TRUST[cam_key]

    if id0_est is None and id1_est is None:
        return None, "없음", False

    if id0_est is None:
        # ID1만 보임 → 신뢰도 낮음
        return id1_est, "ID1만", False, 0.5 * cam_mul

    if id1_est is None:
        # ID0만 보임 → 정상
        return id0_est, "ID0만", False, 1.0 * cam_mul

    # 둘 다 보임 → 위치 차이로 jitter 판단
    dx = float(id0_est["center"][0]) - float(id1_est["center"][0])
    dy = float(id0_est["center"][1]) - float(id1_est["center"][1])
    dist_cm = math.hypot(dx, dy)

    if dist_cm < jitter_thresh_cm:
        # 두 마커 동의 → 신뢰도 높음
        return id0_est, f"동의({dist_cm:.0f}cm)", False, 1.2 * cam_mul
    else:
        # jittering 의심 → quality 낮춤
        return id0_est, f"jitter({dist_cm:.0f}cm)", True, 0.3 * cam_mul

# =======================
# Kalman Filter [x, y, θ, vx, vy, vθ]
# =======================
class RobotKF:
    def __init__(self):
        self.x  = np.zeros((6,1), dtype=float)
        self.P  = np.diag([100.**2, 100.**2,
                           (30.*math.pi/180)**2,
                           50.**2, 50.**2,
                           (20.*math.pi/180)**2]).astype(float)
        self.initialized = False
        self.last_seen   = None

        self.q_xy = 5.0;  self.q_th  = 8.*math.pi/180
        self.q_v  = 20.0; self.q_vth = 15.*math.pi/180
        self.r_xy = 15.0; self.r_th  = 10.*math.pi/180

    def predict(self, dt):
        if not self.initialized or dt <= 0:
            return
        dt = min(dt, 0.5)
        x,y,th,vx,vy,vth = [float(v) for v in self.x.flatten()]
        self.x = np.array([[x+vx*dt],[y+vy*dt],[wrap_pi(th+vth*dt)],
                            [vx],[vy],[vth]], dtype=float)
        F = np.eye(6); F[0,3]=dt; F[1,4]=dt; F[2,5]=dt
        Q = np.diag([(self.q_xy**2)*dt]*2 + [(self.q_th**2)*dt] +
                    [(self.q_v**2)*dt]*2   + [(self.q_vth**2)*dt])
        self.P = F @ self.P @ F.T + Q

    def update(self, cx, cy, cth, now, quality=1.0):
        z = np.array([[cx],[cy],[wrap_pi(cth)]], dtype=float)
        H = np.zeros((3,6)); H[0,0]=H[1,1]=H[2,2]=1.
        R = np.diag([(self.r_xy/quality)**2]*2 + [(self.r_th/quality)**2])
        if not self.initialized:
            self.x[0,0]=cx; self.x[1,0]=cy; self.x[2,0]=wrap_pi(cth)
            self.P = np.diag([25.**2,25.**2,(8.*math.pi/180)**2,
                               40.**2,40.**2,(12.*math.pi/180)**2])
            self.initialized = True
        else:
            inn = z - H @ self.x
            inn[2,0] = wrap_pi(float(inn[2,0]))
            S  = H @ self.P @ H.T + R
            Kg = self.P @ H.T @ np.linalg.inv(S)
            self.x = self.x + Kg @ inn
            self.x[2,0] = wrap_pi(float(self.x[2,0]))
            self.P = (np.eye(6) - Kg @ H) @ self.P
        self.last_seen = now

    def get(self):
        return float(self.x[0,0]), float(self.x[1,0]), float(self.x[2,0])

    def is_visible(self, now):
        if not self.initialized or self.last_seen is None:
            return False
        return (now - self.last_seen) < HIDE_TIMEOUT_S

# =======================
# App
# =======================
class App:
    def __init__(self):
        self.canvas_w, self.canvas_h = 2000, 2000
        self.draw_off_x, self.draw_off_y = 700, 500
        self.grid_world_origin = np.array([200., 150.], dtype=np.float32)
        self.grid_w, self.grid_h = 600, 720
        gx,gy = float(self.grid_world_origin[0]), float(self.grid_world_origin[1])
        self.car_zone_world = ((200+gx,180+gy),(400+gx,540+gy))

        self.cams = {
            "rear": CamCfg("rear",0,
                np.array([301.4,540.],np.float32)+self.grid_world_origin,
                h_cm=105.5, map_angle_deg=90., sens=1.6,
                install_angle=0., install_offset=0., yaw_trim_deg=3.),
            "left": CamCfg("left",1,
                np.array([200.,270.],np.float32)+self.grid_world_origin,
                h_cm=110., map_angle_deg=157., sens=1.6,
                install_angle=113., install_offset=50.84, yaw_trim_deg=8.),
        }

        self.kf           = RobotKF()
        self.last_t       = time.time()
        self.jitter_thresh = JITTER_THRESH_CM

        self.cap0 = cv2.VideoCapture(self.cams["rear"].index)
        self.cap1 = cv2.VideoCapture(self.cams["left"].index)
        if not self.cap0.isOpened() or not self.cap1.isOpened():
            raise RuntimeError("Camera open failed")

        cv2.namedWindow("MAP | jitter-aware KF", cv2.WINDOW_NORMAL)
        cv2.namedWindow("MONITOR", cv2.WINDOW_NORMAL)
        print(f"[INFO] jitter_thresh={self.jitter_thresh}cm  [ / ] 키로 조절")

    def close(self):
        for c in (self.cap0, self.cap1):
            try: c.release()
            except: pass
        cv2.destroyAllWindows()

    def w2c(self, p):
        return p + np.array([self.draw_off_x, self.draw_off_y], np.float32)

    def draw_grid(self, img, x0,y0,w,h,s,cm,cM,ms):
        for x in range(0,w+1,s):
            cv2.line(img,(x0+x,y0),(x0+x,y0+h),cM if x%ms==0 else cm,1)
        for y in range(0,h+1,s):
            cv2.line(img,(x0,y0+y),(x0+w,y0+y),cM if y%ms==0 else cm,1)

    def draw_static(self, canvas):
        self.draw_grid(canvas,0,0,self.canvas_w-1,self.canvas_h-1,
                       20,(25,25,25),(45,45,45),100)
        g0=self.w2c(self.grid_world_origin).astype(int)
        gx,gy=int(g0[0]),int(g0[1])
        self.draw_grid(canvas,gx,gy,self.grid_w,self.grid_h,
                       20,(45,45,45),(80,80,80),100)
        cv2.rectangle(canvas,(gx,gy),(gx+self.grid_w,gy+self.grid_h),(200,200,200),2)
        x0w,y0w=self.car_zone_world[0]; x1w,y1w=self.car_zone_world[1]
        cv2.rectangle(canvas,
            tuple(self.w2c(np.array([x0w,y0w],np.float32)).astype(int)),
            tuple(self.w2c(np.array([x1w,y1w],np.float32)).astype(int)),
            (35,35,45),-1)
        for key,cam in self.cams.items():
            cp=self.w2c(cam.pos_world_px).astype(int)
            cv2.circle(canvas,tuple(cp),6,(220,220,220),-1)
            cv2.putText(canvas,key,(int(cp[0])+8,int(cp[1])-8),
                        0,0.5,(220,220,220),1,cv2.LINE_AA)

    def draw_arrow(self, canvas, center, heading, color, label, r, t, alen):
        c=self.w2c(center).astype(int)
        cv2.circle(canvas,tuple(c),r,color,-1)
        e=self.w2c(center+np.array([alen*math.cos(heading),
                                     alen*math.sin(heading)],np.float32)).astype(int)
        cv2.arrowedLine(canvas,tuple(c),tuple(e),color,t,cv2.LINE_AA,tipLength=0.22)
        cv2.putText(canvas,label,(int(c[0])+10,int(c[1])+5),
                    0,0.48,color,2,cv2.LINE_AA)

    def run(self):
        COL_ROBOT   = (  0, 255, 255)   # 최종 ROBOT (굵음)
        COL_REAR_ID0= ( 80,  80, 180)   # rear id0 raw
        COL_REAR_ID1= ( 80, 160,  80)   # rear id1 raw
        COL_LEFT_ID0= ( 40,  40, 120)   # left id0 raw (더 흐리게)
        COL_JITTER  = (  0,  80, 255)   # jitter 경고색
        MON_W, MON_H = 640, 360

        def make_mon(ok, fr):
            if not ok or fr is None:
                return np.zeros((MON_H,MON_W,3),np.uint8)
            mon = cv2.resize(fr,(MON_W,MON_H))
            gray = cv2.cvtColor(fr,cv2.COLOR_BGR2GRAY)
            corners,ids,_ = detector.detectMarkers(gray)
            if ids is not None and len(corners)>0:
                sx,sy = MON_W/fr.shape[1], MON_H/fr.shape[0]
                scaled = [(c.reshape(-1,1,2)*[sx,sy]).astype(np.float32)
                          for c in corners]
                cv2.aruco.drawDetectedMarkers(mon,scaled,ids)
            return mon

        try:
            while True:
                now = time.time()
                dt  = min(now - self.last_t, 0.5)
                self.last_t = now

                ok0,fr0 = self.cap0.read()
                ok1,fr1 = self.cap1.read()

                rear_f = estimate_frame(fr0, self.cams["rear"]) \
                         if (ok0 and fr0 is not None) else {"id0":None,"id1":None}
                left_f = estimate_frame(fr1, self.cams["left"]) \
                         if (ok1 and fr1 is not None) else {"id0":None,"id1":None}

                # quality 평가
                rear_res = assess_quality(rear_f["id0"], rear_f["id1"],
                                          self.jitter_thresh, "rear")
                left_res = assess_quality(left_f["id0"], left_f["id1"],
                                          self.jitter_thresh, "left")
                # assess_quality 반환: (est, status_str, is_jitter, quality)
                # 없음인 경우 3-tuple이므로 처리
                def unpack(res):
                    if len(res) == 3:   # 없음
                        return None, res[1], False, 0.
                    return res

                rear_est, rear_status, rear_jit, rear_q = unpack(rear_res)
                left_est, left_status, left_jit, left_q = unpack(left_res)

                # KF predict (항상)
                self.kf.predict(dt)

                # 유효한 추정값들로 KF update
                # rear와 left 중 더 신뢰도 높은 것을 먼저, 낮은 것은 보조
                updates = [(e,q) for e,q in [(rear_est,rear_q),(left_est,left_q)]
                           if e is not None and q > 0]
                for est, q in updates:
                    self.kf.update(
                        float(est["center"][0]),
                        float(est["center"][1]),
                        float(est["heading"]),
                        now, quality=q
                    )

                # ── canvas ──────────────────────────────────
                canvas = np.ones((self.canvas_h,self.canvas_w,3),dtype=np.uint8)*15
                self.draw_static(canvas)

                visible   = self.kf.is_visible(now)
                seen_ago  = (now-self.kf.last_seen) if self.kf.last_seen else 999.
                is_jitter = rear_jit or left_jit

                # HUD
                jit_col = COL_JITTER if is_jitter else (180,255,180)
                cv2.putText(canvas,
                    f"rear: {rear_status}  |  left: {left_status}",
                    (10,30),0,0.60,(230,230,230),2,cv2.LINE_AA)
                cv2.putText(canvas,
                    f"jitter thresh={self.jitter_thresh:.0f}cm  "
                    f"({'JITTER!' if is_jitter else '정상'})   [ / ] 조절",
                    (10,58),0,0.55,jit_col,2,cv2.LINE_AA)
                cv2.putText(canvas,
                    f"마지막 검출: {seen_ago:.1f}s 전   "
                    f"화살표: {'표시' if visible else f'숨김(>{HIDE_TIMEOUT_S:.0f}s)'}",
                    (10,84),0,0.50,(200,200,200),1,cv2.LINE_AA)
                cv2.putText(canvas,"Q/ESC=종료",
                    (10,106),0,0.46,(160,160,160),1,cv2.LINE_AA)

                # 참고용 raw 포인트 (작게, 흐리게)
                if rear_f["id0"]:
                    self.draw_arrow(canvas,
                        rear_f["id0"]["center"], rear_f["id0"]["heading"],
                        COL_REAR_ID0, "r_id0", 3, 1, 30)
                if rear_f["id1"]:
                    self.draw_arrow(canvas,
                        rear_f["id1"]["center"], rear_f["id1"]["heading"],
                        COL_REAR_ID1, "r_id1", 3, 1, 30)
                if left_f["id0"]:
                    self.draw_arrow(canvas,
                        left_f["id0"]["center"], left_f["id0"]["heading"],
                        COL_LEFT_ID0, "l_id0", 3, 1, 30)

                # 최종 ROBOT 화살표
                if visible:
                    kx,ky,kth = self.kf.get()
                    arrow_col = COL_JITTER if is_jitter else COL_ROBOT
                    self.draw_arrow(canvas,
                        np.array([kx,ky],np.float32), kth,
                        arrow_col, "ROBOT", 9, 3, 60)

                cv2.imshow("MONITOR",
                           np.hstack([make_mon(ok0,fr0), make_mon(ok1,fr1)]))
                cv2.imshow("MAP | jitter-aware KF", canvas)

                k = cv2.waitKey(1) & 0xFF
                if k in (27, ord('q')):
                    break
                elif k == ord(']'):
                    self.jitter_thresh = min(self.jitter_thresh + 5., 100.)
                    print(f"[INFO] jitter_thresh → {self.jitter_thresh:.0f}cm")
                elif k == ord('['):
                    self.jitter_thresh = max(self.jitter_thresh - 5., 5.)
                    print(f"[INFO] jitter_thresh → {self.jitter_thresh:.0f}cm")

        except KeyboardInterrupt:
            pass
        finally:
            self.close()


if __name__ == "__main__":
    App().run()