import numpy as np
import math

class PathTracker:
    """경로 추적 및 관리 모듈 (Phase 연동 유지 + 180° 회전 관성 모드)"""
    
    def __init__(self, wc_l, map_scale):
        self.wc_l = wc_l
        self.map_scale = map_scale
        self.path = []
        self.planner = None
        self.obstacle_checker = None
        
        # Phase Controller 연동 (오류 방지용 유지)
        self.phase_controller = None
        self.use_phase_mode = False  # 기본은 A* 모드
        
        # 액션 추적 및 대기 변수
        self.last_action = "STOP"
        self.wait_counter = 0
        self.min_wait_frames = 15
        
        # 180° 회전 모드 (사각지대 돌파용)
        self.rotation_mode = False
        self.rotation_direction = None
        self.rotation_target_marker = 1  # 0번(전방) 포착 중일 때 1번(후방)을 찾으러 돎
    
    def set_planner(self, planner):
        self.planner = planner
    
    def set_obstacle_checker(self, checker):
        self.obstacle_checker = checker
    
    def set_phase_controller(self, phase_controller):
        """Phase Controller 설정 (의존성 유지)"""
        self.phase_controller = phase_controller
    
    def clear_path(self):
        self.path = []

    def update_path(self, center, heading_angle, goal_pos, sonar_dist_cm=999.0):
        """경로 업데이트 (Phase 모드 분기 유지)"""
        if self.use_phase_mode:
            if self.phase_controller:
                self.path = [] # Phase 모드 시 경로는 비움
            else:
                self.path = [center.tolist(), goal_pos]
            return

        if center is None: return

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
                if min_d > 50: need_replan = True

        if need_replan and self.planner:
            self.path = self.planner.astar(center, goal_pos)
            print("🔄 A* 경로 재계획 실행")
        elif self.path:
            # 3. 웨이포인트 통과 판단
            if len(self.path) > 1:
                p1, p2 = np.array(self.path[0]), np.array(self.path[1])
                if math.dist(center, p1) < 25 or np.dot(p2 - p1, center - p1) > 0:
                    if len(self.path) > 2: self.path.pop(0)

    def _trigger_wait(self, frames=None):
        self.wait_counter = frames if frames is not None else self.min_wait_frames
        self.last_action = "STOP"

    def compute_action(self, center, current_yaw, marker_id=None, look_ahead_dist=40.0):
        """제어 명령 계산 (Rotation Mode + Phase Mode 지원)"""
        # 0. 대기 카운터 처리
        if self.wait_counter > 0:
            self.wait_counter -= 1
            return 0.0, 0.0, "STOP"

        # 1. Phase 모드일 경우 처리 (기존 로직 유지)
        if self.use_phase_mode:
            # 여기서는 외부(main)에서 phase_controller.step()을 호출할 것이므로 
            # Tracker는 기본적으로 관여하지 않거나 NO PATH 리턴
            return 0.0, 0.0, "PHASE MODE"

        # 2. 180° 회전 모드 (Rotation Mode) 우선 처리
        if self.rotation_mode:
            if marker_id == self.rotation_target_marker:
                print(f"✅ 타겟 마커 {self.rotation_target_marker}번 포착 → 회전 모드 종료")
                self.rotation_mode = False
            else:
                ideal_action = f"TURN {self.rotation_direction}"
                w_speed = 0.18 if self.rotation_direction == "LEFT" else -0.18
                if ideal_action == self.last_action:
                    return 0.0, w_speed, f"{ideal_action} (ROTATION MODE)"
                else:
                    if self.last_action != "STOP":
                        self._trigger_wait()
                        return 0.0, 0.0, "STOP"
                    self.last_action = ideal_action
                    return 0.0, w_speed, ideal_action

        # 3. 일반 A* 주행 모드
        if center is None or not self.path or len(self.path) < 1:
            return 0.0, 0.0, "NO PATH"

        # 타겟 선정
        target_pt = self.path[-1]
        for p in self.path:
            if math.dist(center, p) >= look_ahead_dist:
                target_pt = p
                break

        dx, dy = target_pt[0] - center[0], target_pt[1] - center[1]
        target_yaw = math.atan2(dy, dx)
        yaw_error = math.atan2(math.sin(target_yaw - current_yaw), 
                               math.cos(target_yaw - current_yaw))

        # 4. 회전 모드 진입 조건 판단
        # 0번(전방) 마커를 보고 있는데 가야 할 곳이 뒤쪽(90도 이상)일 때
        if marker_id == 0 and abs(yaw_error) > math.radians(90):
            print(f"🔄 180° 회전 모드 시작 (0번 → 1번 탐색)")
            self.rotation_mode = True
            self.rotation_direction = "LEFT" if yaw_error > 0 else "RIGHT"
            self.rotation_target_marker = 1
            if self.last_action != "STOP":
                self._trigger_wait()
                return 0.0, 0.0, "STOP"
            return 0.0, 0.0, "WAIT ROTATION"

        # 5. 일반 액션 (FORWARD / TURN)
        dead_zone = math.radians(15)
        if abs(yaw_error) < dead_zone:
            ideal_action = "FORWARD"
        elif yaw_error > 0:
            ideal_action = "TURN LEFT"
        else:
            ideal_action = "TURN RIGHT"

        if ideal_action != self.last_action:
            if self.last_action != "STOP":
                self._trigger_wait()
                return 0.0, 0.0, "STOP"

        self.last_action = ideal_action
        actions = {"FORWARD": (0.2, 0.0), "TURN LEFT": (0.0, 0.18), "TURN RIGHT": (0.0, -0.18)}
        v, w = actions.get(ideal_action, (0.0, 0.0))
        return v, w, ideal_action
