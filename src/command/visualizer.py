import cv2
import numpy as np
import math


class Visualizer:
    """맵 및 경로 시각화 모듈 (카메라별/융합 추정치 표시 지원)"""
    
    def __init__(self, map_w, map_h, car_dim, car_pos, wc_w, wc_l, map_scale):
        self.map_w = map_w
        self.map_h = map_h
        self.car_dim = car_dim
        self.car_x, self.car_y = car_pos
        self.wc_w = wc_w
        self.wc_l = wc_l
        self.map_scale = map_scale
    
    def create_map(self):
        """빈 맵 이미지 생성"""
        img = np.ones((self.map_h, self.map_w, 3), dtype=np.uint8) * 15
        
        # 그리드
        for i in range(0, self.map_w, 50):
            cv2.line(img, (i, 0), (i, self.map_h), (25, 25, 25), 1)
        for i in range(0, self.map_h, 50):
            cv2.line(img, (0, i), (self.map_w, i), (25, 25, 25), 1)
        
        return img
    
    def draw_car(self, img):
        """차량 그리기"""
        cv2.rectangle(
            img, 
            (int(self.car_x), int(self.car_y)), 
            (int(self.car_x + self.car_dim[0]), int(self.car_y + self.car_dim[1])), 
            (35, 35, 45), 
            -1
        )
        
        # 주차 경계선 표시 (옵션)
        ext_w, ext_h = 600, 720
        car_center_x = self.car_x + self.car_dim[0] / 2
        car_center_y = self.car_y + self.car_dim[1] / 2
        
        x1 = int(car_center_x - ext_w / 2)
        y1 = int(car_center_y - ext_h / 2)
        x2 = int(car_center_x + ext_w / 2)
        y2 = int(car_center_y + ext_h / 2)
        
        cv2.rectangle(img, (x1, y1), (x2, y2), (60, 60, 60), 2)
        cv2.putText(img, "Parking Area (600x720)", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1)
    
    def draw_obstacles(self, img, obstacles):
        """동적 장애물 그리기"""
        for ox, oy, r in obstacles:
            cv2.circle(img, (ox, oy), r, (0, 0, 150), -1)
            cv2.circle(img, (ox, oy), r, (0, 0, 255), 2)
    
    def draw_goals(self, img, goals, stage, goal_idx, parking_mode):
        """목표 지점 그리기"""
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
        """출차 최종 목표 그리기"""
        for i, g in enumerate(exit_goals[2]):
            gp = (int(g[0]), int(g[1]))
            col = (255, 100, 0) if i == exit_choice else (80, 80, 80)
            cv2.circle(img, gp, 8, col, -1 if i == exit_choice else 2)
    
    def draw_path(self, img, path, marker_pos, heading_angle):
        """경로 및 회전 정보 그리기"""
        if len(path) < 2:
            return
        
        cv2.polylines(img, [np.array(path, np.int32)], False, (0, 255, 255), 2)
        
        # 각도 정보
        pivot = marker_pos
        target = path[-1]
        dx, dy = target[0] - pivot[0], target[1] - pivot[1]
        target_yaw = math.atan2(dy, dx)
        yaw_err = math.degrees(math.atan2(
            math.sin(target_yaw - heading_angle), 
            math.cos(target_yaw - heading_angle)
        ))
        
        # 호
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
        
        # 텍스트
        cv2.putText(
            img, 
            f"Rot: {yaw_err:+.1f}deg", 
            (int(pivot[0]) + 50, int(pivot[1]) - 70), 
            0, 0.5, (0, 255, 255), 1
        )
    
    def draw_stage_info(self, img, marker_pos, stage):
        """스테이지 정보 그리기"""
        cv2.putText(
            img, 
            f"Stage: {stage}", 
            (int(marker_pos[0]) + 50, int(marker_pos[1]) - 55), 
            0, 0.4, (255, 200, 100), 1
        )
    
    def draw_angle_info(self, img, pos, heading_angle):
        """각도 정보 표시 (북쪽 기준)"""
        raw_deg = math.degrees(heading_angle) + 90.0
        display_angle = raw_deg % 360.0
        cv2.putText(
            img,
            f"Angle(N=0): {display_angle:.1f}deg",
            (int(pos[0]) + 50, int(pos[1]) - 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1
        )
    
    def draw_wheelchair(self, img, center_pos, heading_angle,
                        body_color=(0, 255, 0), front_color=(0, 255, 255),
                        thickness=2, label=None):
        """
        휠체어 그리기 (개선 버전)
        
        Args:
            img: 이미지
            center_pos: 중심 위치 (numpy array)
            heading_angle: 헤딩 각도 (라디안)
            body_color: 테두리 색상
            front_color: 앞면 강조 색상
            thickness: 선 두께
            label: 표시할 라벨 (예: "Fused", "Front", "Rear")
        """
        if center_pos is None:
            return
        
        center = center_pos.astype(np.float32)
        w_px = (self.wc_w * self.map_scale) / 2.0
        l_px = (self.wc_l * self.map_scale) / 2.0
        
        # 회전 변환
        base_pts = np.array([
            [-l_px, -w_px], [l_px, -w_px], 
            [l_px, w_px], [-l_px, w_px]
        ], dtype=np.float32)
        
        rot_m = np.array([
            [math.cos(heading_angle), -math.sin(heading_angle)],
            [math.sin(heading_angle),  math.cos(heading_angle)]
        ], dtype=np.float32)
        
        pts = (base_pts @ rot_m.T) + center
        
        # 휠체어 테두리
        cv2.polylines(img, [pts.astype(np.int32)], True, body_color, thickness, cv2.LINE_AA)
        
        # 앞면 강조
        cv2.line(img, tuple(pts[0].astype(int)), tuple(pts[3].astype(int)), 
                 front_color, thickness + 1)
        
        # 방향 화살표
        cv2.arrowedLine(
            img,
            tuple(center.astype(int)),
            (int(center[0] + 45 * math.cos(heading_angle)),
             int(center[1] + 45 * math.sin(heading_angle))),
            body_color, thickness
        )
        
        # 라벨 표시
        if label is not None:
            cv2.putText(img, label, 
                       (int(center[0]) + 10, int(center[1]) + 15),
                       0, 0.5, body_color, 2, cv2.LINE_AA)
    
    def draw_camera_estimates(self, img, per_cam_est, cams):
        """
        카메라별 추정치 그리기 (반투명 휠체어)
        
        Args:
            img: 이미지
            per_cam_est: {cam_name: (center_pos, heading, weight)} 딕셔너리
            cams: 카메라 설정 딕셔너리
        """
        # 카메라별 색상 정의
        cam_colors = {
            'front': (255, 100, 100),  # 빨강
            'rear': (100, 100, 255),   # 파랑
            'left': (100, 255, 100),   # 초록
            'right': (255, 255, 100),  # 노랑
        }
        
        for cam_name, est in per_cam_est.items():
            if est is None:
                continue
            
            cam_pos, cam_heading, weight = est
            color = cam_colors.get(cam_name, (150, 150, 150))
            
            # 가중치에 따라 투명도 조절 (알파 블렌딩 대신 얇은 선으로)
            thickness = max(1, int(weight * 2))
            
            # 카메라별 휠체어 그리기
            self.draw_wheelchair(
                img, 
                cam_pos, 
                cam_heading,
                body_color=color,
                front_color=color,
                thickness=thickness,
                label=f"{cam_name[:1].upper()}"  # F, R, L, R 등
            )
    
    def draw_rays_and_markers(self, img, detections, cams):
        """
        카메라에서 마커까지의 레이와 마커 위치 그리기
        
        Args:
            img: 이미지
            detections: detect_and_estimate에서 반환된 det_all 리스트
            cams: 카메라 설정 딕셔너리
        """
        for d in detections:
            cam = d["cam"]
            if cam not in cams:
                continue
            
            cfg = cams[cam]
            
            # 카메라 위치
            cp = tuple(cfg['pos'].astype(int))
            
            # 마커 위치 (center_pos 대신 marker_pos가 있다면)
            if "marker_pos" in d:
                mp = tuple(d["marker_pos"].astype(int))
            else:
                mp = tuple(d["pos"].astype(int))
            
            # 색상 설정
            col = cfg.get("color", (180, 180, 180))
            
            # 레이 그리기
            cv2.line(img, cp, mp, col, 1, cv2.LINE_AA)
            
            # 마커 표시
            cv2.circle(img, mp, 4, (255, 255, 0), -1)
            cv2.putText(img, f"ID{d['marker_id']}", 
                       (mp[0] + 6, mp[1] - 6),
                       0, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
    
    def draw_help_text(self, img):
        """도움말 텍스트 그리기"""
        cv2.putText(
            img, 
            "L-Click: Add | R-Click: Remove | SPACE: Play/Pause | q: Quit | d: Debug", 
            (10, 30), 
            0, 0.5, (200, 200, 200), 1
        )

    def draw_action_command(self, img, action):
        """화면 좌측 하단에 주행 지시 표시"""
        cmd_map = {
            "FORWARD": ("FORWARD", (0, 255, 0)),
            "TURN LEFT": ("TURN LEFT", (255, 200, 0)),
            "TURN RIGHT": ("TURN RIGHT", (0, 255, 255)),
            "STOP": ("STOP", (0, 0, 255)),
            "BACKWARD": ("BACKWARD", (200, 0, 200))
        }
        
        text, color = cmd_map.get(action, ("WAITING", (150, 150, 150)))
        
        # 가독성을 위한 검은색 배경 박스
        cv2.rectangle(img, (20, self.map_h - 80), (320, self.map_h - 20), (0, 0, 0), -1)
        cv2.rectangle(img, (20, self.map_h - 80), (320, self.map_h - 20), (255, 255, 255), 1)
        
        # 명령 텍스트 출력
        cv2.putText(img, text, (35, self.map_h - 42), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    
    def draw_debug_info(self, img, per_cam_est, total_detections):
        """
        우측 상단에 디버그 정보 표시
        
        Args:
            img: 이미지
            per_cam_est: 카메라별 추정치
            total_detections: 전체 검출 수
        """
        y_offset = 50
        x_pos = self.map_w - 250
        
        # 배경 박스
        cv2.rectangle(img, (x_pos - 10, 30), (self.map_w - 10, y_offset + len(per_cam_est) * 25 + 20), 
                     (0, 0, 0), -1)
        cv2.rectangle(img, (x_pos - 10, 30), (self.map_w - 10, y_offset + len(per_cam_est) * 25 + 20), 
                     (100, 100, 100), 1)
        
        # 전체 검출 수
        cv2.putText(img, f"Total Detections: {total_detections}", 
                   (x_pos, y_offset), 0, 0.5, (255, 255, 255), 1)
        
        # 카메라별 정보
        for i, (cam_name, est) in enumerate(per_cam_est.items()):
            y = y_offset + (i + 1) * 25
            
            if est is None:
                text = f"{cam_name}: N/A"
                color = (100, 100, 100)
            else:
                _, cam_heading, weight = est
                angle_deg = (math.degrees(cam_heading) + 90.0) % 360.0
                text = f"{cam_name}: {angle_deg:.1f}deg (w:{weight:.2f})"
                color = (100, 255, 100)
            
            cv2.putText(img, text, (x_pos, y), 0, 0.45, color, 1)