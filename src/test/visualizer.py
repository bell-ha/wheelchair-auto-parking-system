# visualizer.py
import cv2
import numpy as np
import math


class Visualizer:
    """맵 및 경로 시각화 모듈 (cam별/퓨즈 표시 지원)"""

    def __init__(self, map_w, map_h, car_dim, car_pos, wc_w, wc_l, map_scale):
        self.map_w = map_w
        self.map_h = map_h
        self.car_dim = car_dim
        self.car_x, self.car_y = car_pos
        self.wc_w = wc_w
        self.wc_l = wc_l
        self.map_scale = map_scale

    def create_map(self):
        img = np.ones((self.map_h, self.map_w, 3), dtype=np.uint8) * 15
        for i in range(0, self.map_w, 50):
            cv2.line(img, (i, 0), (i, self.map_h), (25, 25, 25), 1)
        for i in range(0, self.map_h, 50):
            cv2.line(img, (0, i), (self.map_w, i), (25, 25, 25), 1)
        return img

    def draw_car(self, img):
        cv2.rectangle(
            img,
            (int(self.car_x), int(self.car_y)),
            (int(self.car_x + self.car_dim[0]), int(self.car_y + self.car_dim[1])),
            (35, 35, 45),
            -1
        )

        ext_w, ext_h = 600, 720
        car_center_x = self.car_x + self.car_dim[0] / 2
        car_center_y = self.car_y + self.car_dim[1] / 2

        x1 = int(car_center_x - ext_w / 2)
        y1 = int(car_center_y - ext_h / 2)
        x2 = int(car_center_x + ext_w / 2)
        y2 = int(car_center_y + ext_h / 2)

        cv2.rectangle(img, (x1, y1), (x2, y2), (60, 60, 60), 2)
        cv2.putText(img, "Boundary (600x720)", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1)

    def draw_obstacles(self, img, obstacles):
        for ox, oy, r in obstacles:
            cv2.circle(img, (ox, oy), r, (0, 0, 150), -1)
            cv2.circle(img, (ox, oy), r, (0, 0, 255), 2)

    def draw_goals(self, img, goals, stage, goal_idx, parking_mode):
        for si, stage_goals in enumerate(goals):
            for gi, g in enumerate(stage_goals):
                gp = (int(g[0]), int(g[1]))
                is_curr = (si == stage and gi == goal_idx)
                col = (0, 255, 0) if is_curr else (100, 100, 100)
                cv2.circle(img, gp, 10, col, -1 if is_curr else 2)
                cv2.putText(img, f"S{si}", (gp[0]-8, gp[1]-15), 0, 0.4, col, 1)

                if g[2] is not None:
                    ax = int(gp[0] + 25 * math.cos(math.radians(g[2])))
                    ay = int(gp[1] + 25 * math.sin(math.radians(g[2])))
                    cv2.arrowedLine(img, gp, (ax, ay), (150, 150, 255), 2, tipLength=0.4)

    def draw_exit_goals(self, img, exit_goals, exit_choice):
        for i, g in enumerate(exit_goals[2]):
            gp = (int(g[0]), int(g[1]))
            col = (255, 100, 0) if i == exit_choice else (80, 80, 80)
            cv2.circle(img, gp, 8, col, -1 if i == exit_choice else 2)

    def draw_angle_info(self, img, pos, heading_angle):
        raw_deg = math.degrees(heading_angle) + 90.0
        display_angle = raw_deg % 360.0
        cv2.putText(
            img,
            f"Angle(N=0): {display_angle:.1f}deg",
            (int(pos[0]) + 50, int(pos[1]) - 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1
        )

    def draw_path(self, img, path, pos, heading_angle):
        if len(path) < 2:
            return
        cv2.polylines(img, [np.array(path, np.int32)], False, (0, 255, 255), 2)

        pivot = pos
        target = path[-1]
        dx, dy = target[0] - pivot[0], target[1] - pivot[1]
        target_yaw = math.atan2(dy, dx)

        yaw_err = math.degrees(math.atan2(
            math.sin(target_yaw - heading_angle),
            math.cos(target_yaw - heading_angle)
        ))

        cv2.ellipse(
            img,
            (int(pivot[0]), int(pivot[1])),
            (45, 45),
            0,
            -math.degrees(heading_angle),
            -math.degrees(target_yaw),
            (0, 200, 255) if yaw_err > 0 else (255, 150, 0),
            2
        )

        cv2.putText(
            img,
            f"Rot: {yaw_err:+.1f}deg",
            (int(pivot[0]) + 50, int(pivot[1]) - 70),
            0, 0.5, (0, 255, 255), 1
        )

    def draw_stage_info(self, img, pos, stage):
        cv2.putText(
            img,
            f"Stage: {stage}",
            (int(pos[0]) + 50, int(pos[1]) - 55),
            0, 0.4, (255, 200, 100), 1
        )

    def draw_wheelchair(self, img, center_pos, heading_angle,
                        body_color=(0, 255, 0), front_color=(0, 255, 255),
                        thickness=2, label=None):
        if center_pos is None:
            return

        center = center_pos.astype(np.float32)
        w_px = (self.wc_w * self.map_scale) / 2.0
        l_px = (self.wc_l * self.map_scale) / 2.0

        base_pts = np.array([[-l_px, -w_px], [l_px, -w_px], [l_px, w_px], [-l_px, w_px]], dtype=np.float32)
        rot_m = np.array([[math.cos(heading_angle), -math.sin(heading_angle)],
                          [math.sin(heading_angle),  math.cos(heading_angle)]], dtype=np.float32)
        pts = (base_pts @ rot_m.T) + center

        cv2.polylines(img, [pts.astype(np.int32)], True, body_color, thickness, cv2.LINE_AA)
        cv2.line(img, tuple(pts[0].astype(int)), tuple(pts[3].astype(int)), front_color, thickness + 1)

        cv2.arrowedLine(
            img,
            tuple(center.astype(int)),
            (int(center[0] + 45 * math.cos(heading_angle)),
             int(center[1] + 45 * math.sin(heading_angle))),
            body_color, thickness
        )

        if label is not None:
            cv2.putText(img, label, (int(center[0]) + 10, int(center[1]) + 15),
                        0, 0.5, body_color, 2, cv2.LINE_AA)

    def draw_rays_and_markers(self, img, detections, cams):
        for d in detections:
            cam = d["cam"]
            if cam not in cams:
                continue
            cfg = cams[cam]
            cp = tuple(cfg["pos_px"].astype(int))
            mp = tuple(d["marker_pos"].astype(int))
            col = cfg.get("color", (180, 180, 180))

            cv2.line(img, cp, mp, col, 1, cv2.LINE_AA)
            cv2.circle(img, mp, 4, (255, 255, 0), -1)
            cv2.putText(img, f"ID{d['marker_id']}", (mp[0] + 6, mp[1] - 6),
                        0, 0.45, (255, 255, 0), 1, cv2.LINE_AA)

    def draw_help_text(self, img):
        cv2.putText(
            img,
            "L-Click: Add | R-Click: Remove | SPACE: Play/Pause | q: Quit | d: Angle log",
            (10, 30),
            0, 0.5, (200, 200, 200), 1
        )
