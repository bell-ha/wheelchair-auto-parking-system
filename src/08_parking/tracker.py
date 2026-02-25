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
        self.phase_controller = None  # Phase Controller
        self.use_phase_mode = True  # True: Phase 모드, False: A* 모드
        self.obstacle_detected = False  # 장애물 감지 플래그
        
        # 액션 추적 변수
        self.last_action = "STOP"
        self.wait_counter = 0
        self.min_wait_frames = 15  # 명령 전환 전 정지할 프레임 수 (약 0.15~0.2초)
    
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
    
    def update_path(self, center, heading_angle, goal_pos, sonar_dist_cm=999.0):
        """
        경로 업데이트 및 재계획
        
        Args:
            center: 휠체어 중심 위치 [x, y]
            heading_angle: 현재 방향각 (라디안)
            goal_pos: 목표 위치 [x, y]
            sonar_dist_cm: 초음파 거리 (cm)
        """
        # === Phase 모드 (수동 전환 방식) ===
        if self.use_phase_mode:
            # Phase Controller가 있으면 사용
            if self.phase_controller:
                # Phase Controller는 속도 명령을 반환하므로 경로는 비움
                self.path = []
            else:
                # Phase Controller 없으면 단순 직선 경로
                self.path = [center.tolist(), goal_pos]
            return
        
        # === A* 모드 (장애물 있을 때) ===
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
            
            if min_d > 50:
                need_replan = True

        if need_replan:
            if self.planner:
                new_path = self.planner.astar(center, goal_pos)
                self.path = new_path
                print("🔄 A* 경로 재계획 실행")
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
    
    def _trigger_wait(self, frames=None):
        """명령 전환 시 대기를 발생시키는 내부 함수 (PhaseController의 방식)"""
        self.wait_counter = frames if frames is not None else self.min_wait_frames
        self.last_action = "STOP"

    def compute_action(self, center, current_yaw, look_ahead_dist=40.0):
        """제어 명령 계산 (후진 제거 버전)"""
        # 1. 대기 카운터가 동작 중이면 즉시 STOP 반환
        if self.wait_counter > 0:
            self.wait_counter -= 1
            return 0.0, 0.0, "STOP"

        # A* 모드가 아니거나 경로가 없으면 정지
        if self.use_phase_mode or not self.path or len(self.path) < 1:
            return 0.0, 0.0, "NO PATH"
        
        # 2. 타겟 방향 선정 (Look-ahead 방식 적용)
        target_pt = self.path[-1]
        for p in self.path:
            if math.dist(center, p) >= look_ahead_dist:
                target_pt = p
                break

        dx, dy = target_pt[0] - center[0], target_pt[1] - center[1]
        target_yaw = math.atan2(dy, dx)
        yaw_error = math.atan2(math.sin(target_yaw - current_yaw), 
                               math.cos(target_yaw - current_yaw))

        # 3. 액션 판별 (BACKWARD 로직 제거)
        # 직진이 가능한 허용 각도
        dead_zone = math.radians(15) 
        
        # [수정] 후진 대신 방향이 많이 틀어지면 회전만 수행
        if abs(yaw_error) < dead_zone: 
            ideal_action = "FORWARD"
        elif yaw_error > 0: 
            ideal_action = "TURN LEFT"
        else: 
            ideal_action = "TURN RIGHT"

        # 4. 액션 전환 시 대기 처리 (Trigger Wait)
        if ideal_action != self.last_action:
            if self.last_action != "STOP":
                self._trigger_wait()
                return 0.0, 0.0, "STOP"

        # 5. 최종 명령 확정 및 상태 저장
        self.last_action = ideal_action
        
        # 액션 매핑 (BACKWARD 항목 삭제)
        actions = {
            "FORWARD": (0.2, 0.0),
            "TURN LEFT": (0.0, 0.18),  # 제자리 회전 속도를 약간 높임
            "TURN RIGHT": (0.0, -0.18)
        }
        
        v, w = actions.get(ideal_action, (0.0, 0.0))
        return v, w, ideal_action
