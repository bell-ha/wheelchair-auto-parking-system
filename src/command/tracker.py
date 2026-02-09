
import numpy as np
import math


class PathTracker:
    """경로 추적 및 관리 모듈"""
    
    def __init__(self, wc_l, map_scale):
        self.wc_l = wc_l
        self.map_scale = map_scale
        self.path = []
        self.planner = None  # 외부에서 설정
        self.obstacle_checker = None  # 외부에서 설정
        self.last_action = "STOP"
        self.stop_count = 0
        self.min_stop_frames = 5  # 명령 전환 전 정지할 프레임 수 (약 0.15~0.2초)
    
    def set_planner(self, planner):
        """경로 계획기 설정"""
        self.planner = planner
    
    def set_obstacle_checker(self, checker):
        """장애물 검사 함수 설정"""
        self.obstacle_checker = checker
    
    def clear_path(self):
        """경로 초기화"""
        self.path = []
    
    def update_path(self, marker_pos, heading_angle, goal_pos):
        """경로 업데이트 및 재계획"""
        center = marker_pos + np.array([
            (self.wc_l/2) * self.map_scale * math.cos(heading_angle), 
            (self.wc_l/2) * self.map_scale * math.sin(heading_angle)
        ])
        
        need_replan = False
        
        if not self.path or len(self.path) < 2:
            need_replan = True
        else:
            # 1. 장애물 감지
            for i in range(len(self.path)-1):
                p1, p2 = np.array(self.path[i]), np.array(self.path[i+1])
                for t in [0.3, 0.6, 0.9]:
                    check_pt = p1 * (1-t) + p2 * t
                    if self.obstacle_checker and self.obstacle_checker(check_pt[0], check_pt[1]):
                        need_replan = True
                        break
                if need_replan:
                    break
            
            # 2. 경로 이탈 판단
            min_d = float('inf')
            for i in range(len(self.path)-1):
                p1, p2 = np.array(self.path[i]), np.array(self.path[i+1])
                line_vec = p2 - p1
                p_vec = center - p1
                line_len = np.sum(line_vec**2)
                if line_len == 0:
                    d = math.dist(center, p1)
                else:
                    t = max(0, min(1, np.dot(p_vec, line_vec) / line_len))
                    projection = p1 + t * line_vec
                    d = math.dist(center, projection)
                min_d = min(min_d, d)
            
            if min_d > 70:
                need_replan = True

        if need_replan:
            if self.planner:
                new_path = self.planner.astar(center, goal_pos)
                self.path = new_path
                print("🔄 경로 재계획 실행")
        else:
            # 3. 웨이포인트 통과 판단
            if len(self.path) > 1:
                p1 = np.array(self.path[0])
                p2 = np.array(self.path[1])
                
                v_path = p2 - p1
                v_wc = center - p1
                
                dist_to_p1 = math.dist(center, p1)
                dot_product = np.dot(v_path, v_wc)
                
                if dist_to_p1 < 25 or dot_product > 0:
                    if len(self.path) > 2:
                        self.path.pop(0)
    
    def get_path(self):
        """현재 경로 반환"""
        return self.path
    
    # tracker.py 또는 CompactTracker 클래스 내부에 추가
    def compute_action(self, center, current_yaw, look_ahead_dist=40.0):
        # 1. 원시 타겟 방향 계산 (기존 로직)
        target_pt = self.path[-1]
        for p in self.path:
            if math.dist(center, p) >= look_ahead_dist:
                target_pt = p
                break

        dx, dy = target_pt[0] - center[0], target_pt[1] - center[1]
        target_yaw = math.atan2(dy, dx)
        yaw_error = math.atan2(math.sin(target_yaw - current_yaw), 
                               math.cos(target_yaw - current_yaw))

        # 2. 이번 프레임에서 내리고 싶은 '이상적인' 명령 판별
        dead_zone = math.radians(25)
        if abs(yaw_error) < dead_zone:
            ideal_action = "FORWARD"
        elif yaw_error > 0:
            ideal_action = "TURN LEFT"
        else:
            ideal_action = "TURN RIGHT"

        # 3. [핵심] 명령 전환 시 강제 정지 로직
        # 이전 명령과 지금 내리려는 명령이 다를 때
        if ideal_action != self.last_action and self.last_action != "STOP":
            self.stop_count = self.min_stop_frames
            self.last_action = "STOP"
            return 0.0, 0.0, "STOP"

        # 정지 카운트가 남아있는 경우 (정지 유지)
        if self.stop_count > 0:
            self.stop_count -= 1
            return 0.0, 0.0, "STOP (TRANSITION)"

        # 4. 최종 명령 확정
        self.last_action = ideal_action
        if ideal_action == "FORWARD":
            return 0.2, 0.0, "FORWARD"
        elif ideal_action == "TURN LEFT":
            return 0.0, 0.15, "TURN LEFT"
        elif ideal_action == "TURN RIGHT":
            return 0.0, -0.15, "TURN RIGHT"
        
        return 0.0, 0.0, "STOP"