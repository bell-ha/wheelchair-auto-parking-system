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
    
    def draw_parking_goal(self, img, goal):
        """주차 최종 목표 지점 그리기 (녹색)"""
        cv2.circle(img, (int(goal[0]), int(goal[1])), 12, (0, 255, 0), -1)
        cv2.putText(img, "PARK GOAL", (int(goal[0])-40, int(goal[1])-20), 0, 0.5, (0, 255, 0), 2)

    def draw_exit_goal(self, img, goal):
        """출차 최종 목표 지점 그리기 (주황색)"""
        cv2.circle(img, (int(goal[0]), int(goal[1])), 12, (255, 100, 0), -1)
        cv2.putText(img, "EXIT GOAL", (int(goal[0])-40, int(goal[1])-20), 0, 0.5, (255, 100, 0), 2)
    
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
    
    def draw_phase_guidance(self, img, center, heading_angle, current_phase, phase_controller, car_pos, goal_pos, exit_goal=None, phase_mode_text=""):
        import math
        center_pt = (int(center[0]), int(center[1]))
        current_yaw_deg = math.degrees(heading_angle)
        info_y_start = 120

        is_parking = phase_controller.is_parking
        mode_text = "PARKING MODE" if is_parking else "EXIT MODE"
        mode_color = (0, 255, 100) if is_parking else (255, 100, 100)
        cv2.putText(img, mode_text, (10, 50), 0, 0.7, mode_color, 2)

        # === 우측 상단 Phase 박스 시각화 (동적 텍스트 적용) ===
        # 텍스트가 길어질 수 있으므로 박스 너비를 300 정도로 넓게 잡습니다.
        bx, by, bw, bh = self.map_w - 310, 10, 300, 60
        
        # 단계에 따른 박스 색상 결정
        if current_phase >= 7:
            box_color = (0, 255, 0) # 완료 시 초록색
        else:
            box_color = (255, 200, 100) if is_parking else (100, 200, 255)

        # 배경 및 테두리 그리기
        cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (30, 30, 30), -1)
        cv2.rectangle(img, (bx, by), (bx + bw, by + bh), box_color, 3)

        # [핵심] phase_mode_text(control_mode)를 직접 출력
        cv2.putText(img, phase_mode_text, (bx + 15, by + 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2)

        # --- 내부 유틸리티 2: 공통 시각화 함수 정의 ---
        def draw_info_text(labels):
            for i, (txt, col) in enumerate(labels):
                cv2.putText(img, txt, (10, info_y_start + (i * 30)), 0, 0.6, col, 2)

        def draw_angle_guide(target_deg, arc_c, arrow_c):
            radius = 100
            for deg, color in [(current_yaw_deg, (255, 255, 255)), (target_deg, arrow_c)]:
                rad = math.radians(deg)
                tip = (int(center[0] + radius * math.cos(rad)), int(center[1] + radius * math.sin(rad)))
                cv2.arrowedLine(img, center_pt, tip, color, 3, tipLength=0.3)
            diff = target_deg - current_yaw_deg
            if abs(diff) > 180: diff = diff - 360 if diff > 0 else diff + 360
            cv2.ellipse(img, center_pt, (radius, radius), 0, -current_yaw_deg, -target_deg, arc_c, 3)
            draw_info_text([(f"Current: {current_yaw_deg:.1f}deg", (255, 255, 255)), 
                            (f"Target: {target_deg:.1f}deg", arrow_c), (f"To Rotate: {diff:+.1f}deg", arc_c)])

        def draw_y_guide(target_y, color):
            ly = int(target_y)
            cv2.line(img, (0, ly), (self.map_w, ly), (0, 255, 255), 2)
            cv2.line(img, center_pt, (int(center[0]), ly), color, 2, cv2.LINE_AA)
            draw_info_text([(f"Dist Y: {abs(center[1]-target_y):.1f}px", color)])

        def draw_x_guide(target_x, color):
            lx = int(target_x)
            cv2.line(img, (lx, 0), (lx, self.map_h), (200, 100, 255), 2)
            cv2.line(img, center_pt, (lx, int(center[1])), color, 2, cv2.LINE_AA)
            draw_info_text([(f"Offset X: {abs(center[0]-target_x):.1f}px", color)])

        # --- 메인 시각화 로직 ---

        if is_parking:
            # ================= PARKING 시나리오 =================
            if current_phase in [1, 3, 5]:
                # 주차 각도 정렬 (P1, P3, P5)
                target = (phase_controller.p1_min + phase_controller.p1_max)/2 if current_phase == 1 else \
                         (phase_controller.p3_min + phase_controller.p3_max)/2 if current_phase == 3 else \
                         (phase_controller.p5_min + phase_controller.p5_max)/2
                draw_angle_guide(target, (100, 255, 255), (0, 255, 0))
            elif current_phase == 2:
                draw_y_guide(goal_pos[1], (100, 255, 100)) # Y축 접근
            elif current_phase == 4:
                draw_x_guide(goal_pos[0], (150, 100, 255)) # X축 정렬
            elif current_phase == 6:
                draw_info_text([("Entering Spot...", (100, 255, 100))])
                tx, ty = int(center[0] + 80 * math.cos(heading_angle)), int(center[1] + 80 * math.sin(heading_angle))
                cv2.arrowedLine(img, center_pt, (tx, ty), (100, 255, 100), 4, tipLength=0.3)
        
        else:
            # ================= EXIT 시나리오 =================
            # 1. 출차 P1: Y축 후진 (주차 목표선 기준 가로선)
            if current_phase == 1:
                draw_y_guide(goal_pos[1], (255, 150, 150))
            
            # 2. 출차 P2 & P4: 각도 정렬 (주차와 반대 방향)
            elif current_phase in [2, 4]:
                # UnifiedPhaseController에서 부호 반전된 값을 p3(P2용), p1(P4용)으로 이미 가지고 있음
                target = (phase_controller.p3_min + phase_controller.p3_max)/2 if current_phase == 2 else \
                         (phase_controller.p1_min + phase_controller.p1_max)/2
                draw_angle_guide(target, (255, 200, 100), (255, 150, 0))

            # 3. 출차 P3: X축 이동 (요청하신 X축 정렬선)
            elif current_phase == 3:
                if exit_goal:
                    draw_x_guide(exit_goal[0], (200, 100, 255))
            
            # 4. 출차 P5: 최종 탈출
            elif current_phase == 5:
                draw_info_text([("Moving to Exit Goal...", (100, 255, 100))])
                if exit_goal:
                    eg = (int(exit_goal[0]), int(exit_goal[1]))
                    cv2.line(img, center_pt, eg, (255, 150, 150), 1, cv2.LINE_AA)
                    cv2.arrowedLine(img, center_pt, eg, (100, 255, 100), 2, tipLength=0.2)