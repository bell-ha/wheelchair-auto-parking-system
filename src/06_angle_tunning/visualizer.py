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

    def draw_car(self, img):
        """차량 및 차량 주변 확장 구역 그리기"""
        # 1. 기존 차량 그리기
        cv2.rectangle(
            img, 
            (int(self.car_x), int(self.car_y)), 
            (int(self.car_x + self.car_dim[0]), int(self.car_y + self.car_dim[1])), 
            (35, 35, 45), 
            -1
        )

        # 2. [추가] 차량 밖으로 600x720 사각형 그리기
        # 차량의 중심을 기준으로 그릴지, 좌측 상단을 기준으로 그릴지에 따라 좌표가 달라집니다.
        # 여기서는 차량 구역을 포함하도록 '확장 영역' 개념으로 중심 기준 계산을 적용해 보겠습니다.
        
        ext_w, ext_h = 600, 720
        # 차량의 중심점 계산
        car_center_x = self.car_x + self.car_dim[0] / 2
        car_center_y = self.car_y + self.car_dim[1] / 2
        
        # 600x720 사각형의 좌상단(x1, y1) 및 우하단(x2, y2) 좌표 계산
        x1 = int(car_center_x - ext_w / 2)
        y1 = int(car_center_y - ext_h / 2)
        x2 = int(car_center_x + ext_w / 2)
        y2 = int(car_center_y + ext_h / 2)
        
        # 사각형 그리기 (연한 회색 점선 느낌의 실선)
        cv2.rectangle(img, (x1, y1), (x2, y2), (60, 60, 60), 2)
        
        # 영역 이름 텍스트 (선택 사항)
        cv2.putText(img, "Boundary (600x720)", (x1, y1 - 10), 
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
    
    def draw_angle_info(self, img, marker_pos, heading_angle):
        """위를 0도로 기준 잡은 현재 각도 표시 (0~360도 체계)"""
        # 1. 위(북향)를 0도로 설정 (시계 방향으로 증가)
        # 라디안을 도로 변환 후 90도 보정
        raw_deg = math.degrees(heading_angle) + 90
        
        # 2. 0~360도 범위로 변환 (모듈러 연산 활용)
        # 음수 각도를 양수로 만들고 360으로 나눈 나머지를 취함
        display_angle = raw_deg % 360
        
        # 3. 텍스트 출력
        cv2.putText(
            img, 
            f"Angle(N=0): {display_angle:.1f}deg", 
            (int(marker_pos[0]) + 50, int(marker_pos[1]) - 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1
        )

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
    
    def draw_wheelchair(self, img, marker_pos, heading_angle):
        """휠체어 그리기"""
        center = marker_pos + np.array([
            (self.wc_l/2) * self.map_scale * math.cos(heading_angle), 
            (self.wc_l/2) * self.map_scale * math.sin(heading_angle)
        ])
        
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
            tuple(marker_pos.astype(int)), 
            (int(marker_pos[0] + 45 * math.cos(heading_angle)), 
             int(marker_pos[1] + 45 * math.sin(heading_angle))), 
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