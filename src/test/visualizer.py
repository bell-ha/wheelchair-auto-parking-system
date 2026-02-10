import cv2
import numpy as np
import math


class Visualizer:
    """맵 및 경로 시각화 모듈"""
    
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
    
    def draw_path(self, img, path, center, heading_angle):
        """경로 및 회전 정보 그리기"""
        if len(path) < 2:
            return
        
        cv2.polylines(img, [np.array(path, np.int32)], False, (0, 255, 255), 2)
        
        # 각도 정보
        pivot = center
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
    
    def draw_wheelchair(self, img, center, heading_angle):
        """휠체어 그리기"""
        w, l = (self.wc_w * self.map_scale) / 2, (self.wc_l * self.map_scale) / 2
        rot = np.array([
            [math.cos(heading_angle), -math.sin(heading_angle)],
            [math.sin(heading_angle), math.cos(heading_angle)]
        ])
        
        pts = np.dot([[-l, -w], [l, -w], [l, w], [-l, w]], rot.T) + center
        cv2.polylines(img, [pts.astype(np.int32)], True, (0, 255, 0), 2)
        cv2.line(img, tuple(pts[0].astype(int)), tuple(pts[3].astype(int)), (0, 0, 255), 3)
        cv2.arrowedLine(
            img,
            tuple(center.astype(int)),
            (int(center[0] + 45 * math.cos(heading_angle)),
             int(center[1] + 45 * math.sin(heading_angle))),
            (255, 255, 255),
            2
        )
    
    def draw_help_text(self, img):
        """도움말 텍스트 그리기"""
        cv2.putText(
            img, 
            "L-Click: Add Obstacle | R-Click: Remove", 
            (10, 30), 
            0, 0.5, (200, 200, 200), 1
        )
    
    def draw_action_command(self, img, action_text):
        """
        현재 주행 명령 표시
        
        Args:
            img: 이미지
            action_text: 명령 텍스트 (예: "FORWARD", "TURN LEFT" 등)
        """
        # 액션별 색상 설정
        color_map = {
            "FORWARD": (0, 255, 0),
            "TURN LEFT": (255, 200, 0),
            "TURN RIGHT": (255, 100, 0),
            "STOP": (0, 0, 255),
            "STOP (TRANSITION)": (150, 150, 255),
            "NO PATH": (100, 100, 100)
        }
        
        color = color_map.get(action_text, (200, 200, 200))
        
        # 큰 글씨로 중앙 상단에 표시
        text_size = cv2.getTextSize(action_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
        text_x = (self.map_w - text_size[0]) // 2
        text_y = 60
        
        # 배경 박스
        padding = 15
        cv2.rectangle(
            img,
            (text_x - padding, text_y - text_size[1] - padding),
            (text_x + text_size[0] + padding, text_y + padding),
            (30, 30, 30),
            -1
        )
        
        # 테두리
        cv2.rectangle(
            img,
            (text_x - padding, text_y - text_size[1] - padding),
            (text_x + text_size[0] + padding, text_y + padding),
            color,
            2
        )
        
        # 텍스트
        cv2.putText(
            img,
            action_text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            color,
            3
        )
    
    def draw_phase_guidance(self, img, center, heading_angle, current_phase, phase_controller, car_pos, target_y_px):
        """
        Phase별 맞춤 시각화
        
        Args:
            img: 이미지
            center: 휠체어 중심 위치 [x, y] (픽셀) - PhaseController와 동일한 값
            heading_angle: 현재 방향각 (라디안)
            current_phase: 현재 Phase 번호 (1~7)
            phase_controller: PhaseController 인스턴스
            car_pos: 차량 위치 (car_x, car_y)
            target_y_px: Phase 2 목표 Y 좌표 (픽셀)
        """
        import math
        
        center_pt = (int(center[0]), int(center[1]))
        current_yaw_deg = math.degrees(heading_angle)
        
        # Phase 정보 박스 (우측 상단)
        phase_info = {
            1: ("PHASE 1: ALIGNING", (255, 200, 100)),
            2: ("PHASE 2: APPROACHING", (100, 255, 200)),
            3: ("PHASE 3: ROTATING", (255, 150, 100)),
            4: ("PHASE 4: X-ALIGN", (200, 100, 255)),
            5: ("PHASE 5: FINAL ALIGN", (255, 100, 200)),
            6: ("PHASE 6: ENTERING", (100, 255, 100)),
            7: ("COMPLETE", (0, 255, 0))
        }
        
        if current_phase in phase_info:
            phase_name, phase_color = phase_info[current_phase]
            
            # Phase 박스 배경 (우측 상단)
            box_x, box_y = self.map_w - 250, 10
            box_w, box_h = 240, 60
            
            cv2.rectangle(img, (box_x, box_y), (box_x + box_w, box_y + box_h), (30, 30, 30), -1)
            cv2.rectangle(img, (box_x, box_y), (box_x + box_w, box_y + box_h), phase_color, 3)
            
            cv2.putText(img, phase_name, (box_x + 10, box_y + 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, phase_color, 2)
        
        # === Phase별 시각화 ===
        
        # PHASE 1: 각도 정렬 (-90.5 ~ -89.5도)
        if current_phase == 1:
            target_min = phase_controller.p1_min
            target_max = phase_controller.p1_max
            target_avg = (target_min + target_max) / 2
            
            # 목표 각도 범위 표시
            arc_radius = 100
            
            # 현재 각도 화살표 (흰색)
            curr_x = int(center[0] + arc_radius * math.cos(heading_angle))
            curr_y = int(center[1] + arc_radius * math.sin(heading_angle))
            cv2.arrowedLine(img, center_pt, (curr_x, curr_y), (255, 255, 255), 3, tipLength=0.3)
            
            # 목표 각도 범위 (녹색 호)
            target_rad = math.radians(target_avg)
            target_x = int(center[0] + arc_radius * math.cos(target_rad))
            target_y = int(center[1] + arc_radius * math.sin(target_rad))
            cv2.arrowedLine(img, center_pt, (target_x, target_y), (0, 255, 0), 3, tipLength=0.3)
            
            # 회전 방향 호
            angle_diff = target_avg - current_yaw_deg
            if abs(angle_diff) > 180:
                angle_diff = angle_diff - 360 if angle_diff > 0 else angle_diff + 360
            
            arc_color = (100, 255, 255)
            cv2.ellipse(img, center_pt, (arc_radius, arc_radius), 0, 
                       -current_yaw_deg, -target_avg, arc_color, 3)
            
            # 텍스트 정보
            info_y = 120
            cv2.putText(img, f"Current: {current_yaw_deg:.1f}deg", (10, info_y), 
                       0, 0.6, (255, 255, 255), 2)
            cv2.putText(img, f"Target: {target_avg:.1f}deg", (10, info_y + 30), 
                       0, 0.6, (0, 255, 0), 2)
            cv2.putText(img, f"To Rotate: {angle_diff:+.1f}deg", (10, info_y + 60), 
                       0, 0.6, (100, 255, 255), 2)
        
        # PHASE 2: Y축 주행 (목표 Y까지 직진)
        elif current_phase == 2:
            # Y축 목표선 그리기 (가로선)
            line_y = int(target_y_px)
            cv2.line(img, (0, line_y), (self.map_w, line_y), (0, 255, 255), 2)
            cv2.putText(img, "TARGET Y", (10, line_y - 10), 0, 0.5, (0, 255, 255), 2)
            
            # 현재 위치에서 목표까지 수직선
            cv2.line(img, center_pt, (int(center[0]), line_y), (100, 255, 100), 2, cv2.LINE_AA)
            cv2.arrowedLine(img, center_pt, (int(center[0]), int((center[1] + line_y) / 2)), 
                           (100, 255, 100), 2, tipLength=0.2)
            
            # 거리 정보
            dist_to_target = abs(center[1] - target_y_px)
            info_y = 120
            cv2.putText(img, f"Current Y: {center[1]:.1f}px", (10, info_y), 
                       0, 0.6, (255, 255, 255), 2)
            cv2.putText(img, f"Target Y: {target_y_px:.1f}px", (10, info_y + 30), 
                       0, 0.6, (0, 255, 255), 2)
            cv2.putText(img, f"Distance: {dist_to_target:.1f}px", (10, info_y + 60), 
                       0, 0.6, (100, 255, 100), 2)
        
        # PHASE 3: 회전 (135~140도)
        elif current_phase == 3:
            target_min = phase_controller.p3_min
            target_max = phase_controller.p3_max
            target_avg = (target_min + target_max) / 2
            
            arc_radius = 100
            
            # 현재 각도
            curr_x = int(center[0] + arc_radius * math.cos(heading_angle))
            curr_y = int(center[1] + arc_radius * math.sin(heading_angle))
            cv2.arrowedLine(img, center_pt, (curr_x, curr_y), (255, 255, 255), 3, tipLength=0.3)
            
            # 목표 각도
            target_rad = math.radians(target_avg)
            target_x = int(center[0] + arc_radius * math.cos(target_rad))
            target_y = int(center[1] + arc_radius * math.sin(target_rad))
            cv2.arrowedLine(img, center_pt, (target_x, target_y), (255, 150, 0), 3, tipLength=0.3)
            
            # 회전 호
            angle_diff = target_avg - current_yaw_deg
            if abs(angle_diff) > 180:
                angle_diff = angle_diff - 360 if angle_diff > 0 else angle_diff + 360
            
            cv2.ellipse(img, center_pt, (arc_radius, arc_radius), 0, 
                       -current_yaw_deg, -target_avg, (255, 200, 100), 3)
            
            info_y = 120
            cv2.putText(img, f"Current: {current_yaw_deg:.1f}deg", (10, info_y), 
                       0, 0.6, (255, 255, 255), 2)
            cv2.putText(img, f"Target: {target_avg:.1f}deg", (10, info_y + 30), 
                       0, 0.6, (255, 150, 0), 2)
            cv2.putText(img, f"To Rotate: {angle_diff:+.1f}deg", (10, info_y + 60), 
                       0, 0.6, (255, 200, 100), 2)
        
        # PHASE 4: X축 정렬
        elif current_phase == 4:
            # 차량 중심선
            car_cx = car_pos[0] + self.car_dim[0] / 2
            
            # X축 목표선 (세로선)
            cv2.line(img, (int(car_cx), 0), (int(car_cx), self.map_h), (200, 100, 255), 2)
            cv2.putText(img, "TARGET X", (int(car_cx) + 10, 30), 0, 0.5, (200, 100, 255), 2)
            
            # 현재 위치에서 목표까지 수평선
            cv2.line(img, center_pt, (int(car_cx), int(center[1])), (150, 100, 255), 2, cv2.LINE_AA)
            cv2.arrowedLine(img, center_pt, (int((center[0] + car_cx) / 2), int(center[1])), 
                           (150, 100, 255), 2, tipLength=0.2)
            
            # 거리 정보
            x_offset = abs(center[0] - car_cx)
            info_y = 120
            cv2.putText(img, f"Current X: {center[0]:.1f}px", (10, info_y), 
                       0, 0.6, (255, 255, 255), 2)
            cv2.putText(img, f"Target X: {car_cx:.1f}px", (10, info_y + 30), 
                       0, 0.6, (200, 100, 255), 2)
            cv2.putText(img, f"X Offset: {x_offset:.1f}px", (10, info_y + 60), 
                       0, 0.6, (150, 100, 255), 2)
        
        # PHASE 5: 최종 정렬 (90~91도)
        elif current_phase == 5:
            target_min = phase_controller.p5_min
            target_max = phase_controller.p5_max
            target_avg = (target_min + target_max) / 2
            
            arc_radius = 100
            
            # 현재 각도
            curr_x = int(center[0] + arc_radius * math.cos(heading_angle))
            curr_y = int(center[1] + arc_radius * math.sin(heading_angle))
            cv2.arrowedLine(img, center_pt, (curr_x, curr_y), (255, 255, 255), 3, tipLength=0.3)
            
            # 목표 각도
            target_rad = math.radians(target_avg)
            target_x = int(center[0] + arc_radius * math.cos(target_rad))
            target_y = int(center[1] + arc_radius * math.sin(target_rad))
            cv2.arrowedLine(img, center_pt, (target_x, target_y), (255, 100, 200), 3, tipLength=0.3)
            
            # 회전 호
            angle_diff = target_avg - current_yaw_deg
            if abs(angle_diff) > 180:
                angle_diff = angle_diff - 360 if angle_diff > 0 else angle_diff + 360
            
            cv2.ellipse(img, center_pt, (arc_radius, arc_radius), 0, 
                       -current_yaw_deg, -target_avg, (255, 150, 200), 3)
            
            info_y = 120
            cv2.putText(img, f"Current: {current_yaw_deg:.1f}deg", (10, info_y), 
                       0, 0.6, (255, 255, 255), 2)
            cv2.putText(img, f"Target: {target_avg:.1f}deg", (10, info_y + 30), 
                       0, 0.6, (255, 100, 200), 2)
            cv2.putText(img, f"To Rotate: {angle_diff:+.1f}deg", (10, info_y + 60), 
                       0, 0.6, (255, 150, 200), 2)
        
        # PHASE 6: 진입 (초음파 거리 표시)
        elif current_phase == 6:
            info_y = 120
            cv2.putText(img, "Entering parking spot...", (10, info_y), 
                       0, 0.7, (100, 255, 100), 2)
            
            # 진입 방향 화살표
            arrow_len = 80
            arrow_x = int(center[0] + arrow_len * math.cos(heading_angle))
            arrow_y = int(center[1] + arrow_len * math.sin(heading_angle))
            cv2.arrowedLine(img, center_pt, (arrow_x, arrow_y), (100, 255, 100), 4, tipLength=0.3)