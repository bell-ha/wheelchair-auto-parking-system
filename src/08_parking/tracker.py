import numpy as np
import math

class PathTracker:
    def __init__(self, wc_l, map_scale):
        self.wc_l = wc_l
        self.map_scale = map_scale
        self.path = []
        self.planner = None
        self.obstacle_checker = None
        self.phase_controller = None
        self.use_phase_mode = False
        
        self.last_action = "STOP"
        self.wait_counter = 0
        self.min_wait_frames = 15
        
        # 180° 회전 및 상태 관리
        self.rotation_mode = False
        self.rotation_direction = None
        self.rotation_target_marker = 1

    def set_planner(self, planner):
        """경로 계획기 설정"""
        self.planner = planner
    
    def set_obstacle_checker(self, checker):
        """장애물 검사 함수 설정"""
        self.obstacle_checker = checker
    
    def set_phase_controller(self, phase_controller):
        """Phase Controller 설정"""
        self.phase_controller = phase_controller
    
    def clear_path(self):
        """경로 초기화"""
        self.path = []
    
    def check_obstacles_in_corridor(self, center, goal_pos, corridor_width=100):
        """
        현재 위치에서 목표까지 직선 경로상의 장애물 체크
        
        Args:
            center: 현재 위치 [x, y]
            goal_pos: 목표 위치 [x, y]
            corridor_width: 경로 폭 (픽셀)
        
        Returns:
            bool: 장애물이 있으면 True
        """
        if not self.obstacle_checker:
            return False
        
        # 시작점과 목표점 사이를 샘플링하여 장애물 체크
        num_samples = 50
        for i in range(num_samples + 1):
            t = i / num_samples
            check_pt = center * (1 - t) + np.array(goal_pos) * t
            
            # 경로 주변 폭도 체크 (좌우로 corridor_width/2 만큼)
            for offset in range(-corridor_width//2, corridor_width//2, 10):
                # 경로에 수직인 방향으로 오프셋 적용
                dx = goal_pos[0] - center[0]
                dy = goal_pos[1] - center[1]
                length = math.sqrt(dx**2 + dy**2)
                if length > 0:
                    perp_x = -dy / length * offset
                    perp_y = dx / length * offset
                    test_pt = check_pt + np.array([perp_x, perp_y])
                    
                    if self.obstacle_checker(test_pt[0], test_pt[1]):
                        return True
        
        return False

    def get_path(self):
        return self.path

    def update_path(self, center, heading_angle, goal_pos, sonar_dist_cm=999.0):
        """경로 업데이트 및 재계획 (회전 시 재계획 억제 로직 추가)"""
        if self.use_phase_mode or center is None:
            return

        # [수정] 현재 회전 중이라면 경로 재계획을 수행하지 않음
        # 회전할 때 마커 좌표가 움직여서 생기는 가짜 '경로 이탈'을 방지합니다.
        if self.rotation_mode or "TURN" in self.last_action:
            return 

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
                if need_replan: break
            
            # 2. 경로 이탈 판단
            if not need_replan:
                min_d = float('inf')
                for i in range(len(self.path)-1):
                    p1, p2 = np.array(self.path[i]), np.array(self.path[i+1])
                    line_vec = p2 - p1
                    p_vec = center - p1
                    line_len = np.sum(line_vec**2)
                    d = math.dist(center, p1) if line_len == 0 else math.dist(center, p1 + max(0, min(1, np.dot(p_vec, line_vec) / line_len)) * line_vec)
                    min_d = min(min_d, d)
                
                # [수정] 이탈 허용 범위를 50에서 70으로 소폭 완화하여 노이즈에 대응
                if min_d > 70: 
                    need_replan = True

        if need_replan and self.planner:
            self.path = self.planner.astar(center, goal_pos)
            print("🔄 A* 경로 재계획 실행")
        elif self.path:
            # 3. 웨이포인트 통과 판단
            if len(self.path) > 1:
                p1, p2 = np.array(self.path[0]), np.array(self.path[1])
                # 통과 판정 거리를 늘려 더 부드럽게 다음 점으로 넘어가게 함
                if math.dist(center, p1) < 35 or np.dot(p2 - p1, center - p1) > 0:
                    if len(self.path) > 2: self.path.pop(0)

    def _trigger_wait(self, frames=None):
        self.wait_counter = frames if frames is not None else self.min_wait_frames
        self.last_action = "STOP"

    def compute_action(self, center, current_yaw, marker_id=None, look_ahead_dist=40.0):
        """제어 명령 계산"""
        if self.wait_counter > 0:
            self.wait_counter -= 1
            return 0.0, 0.0, "STOP"

        # 1. 180° 회전 모드 (관성 유지)
        if self.rotation_mode:
            if marker_id == self.rotation_target_marker:
                self.rotation_mode = False
            else:
                w_speed = 0.18 if self.rotation_direction == "LEFT" else -0.18
                return 0.0, w_speed, f"TURN {self.rotation_direction} (ROTATION)"

        if center is None or not self.path:
            return 0.0, 0.0, "IDLE"

        # 2. 타겟 선정
        target_pt = self.path[-1]
        for p in self.path:
            if math.dist(center, p) >= look_ahead_dist:
                target_pt = p
                break

        dx, dy = target_pt[0] - center[0], target_pt[1] - center[1]
        target_yaw = math.atan2(dy, dx)
        yaw_error = math.atan2(math.sin(target_yaw - current_yaw), 
                               math.cos(target_yaw - current_yaw))

        # 3. 회전 모드 진입 판정
        if marker_id == 0 and abs(yaw_error) > math.radians(90):
            self.rotation_mode = True
            self.rotation_direction = "LEFT" if yaw_error > 0 else "RIGHT"
            self._trigger_wait(5)
            return 0.0, 0.0, "WAIT ROTATION"

        # 4. 액션 판별
        dead_zone = math.radians(15)
        if abs(yaw_error) < dead_zone:
            ideal_action = "FORWARD"
        else:
            ideal_action = "TURN LEFT" if yaw_error > 0 else "TURN RIGHT"

        # 5. 액션 전환 시 대기
        if ideal_action != self.last_action:
            # [수정] 회전 방향 사이(LEFT <-> RIGHT)의 떨림 시에는 대기 생략
            if not ("TURN" in ideal_action and "TURN" in self.last_action):
                if self.last_action != "STOP":
                    self._trigger_wait()
                    return 0.0, 0.0, "STOP"

        self.last_action = ideal_action
        actions = {"FORWARD": (0.2, 0.0), "TURN LEFT": (0.0, 0.18), "TURN RIGHT": (0.0, -0.18)}
        v, w = actions.get(ideal_action, (0.0, 0.0))
        return v, w, ideal_action
