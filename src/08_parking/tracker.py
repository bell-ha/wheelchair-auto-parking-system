import numpy as np
import math

class PathTracker:
    """경로 추적 및 관리 모듈 (오류 수정 및 180° 회전 모드 포함)"""
    
    def __init__(self, wc_l, map_scale):
        self.wc_l = wc_l
        self.map_scale = map_scale
        self.path = []
        self.planner = None
        self.obstacle_checker = None
        
        # 외부 인터페이스 유지를 위한 변수들
        self.phase_controller = None
        self.use_phase_mode = False
        
        # 액션 및 상태 관리
        self.last_action = "STOP"
        self.wait_counter = 0
        self.min_wait_frames = 15
        
        # 180° 회전 모드 관련
        self.rotation_mode = False
        self.rotation_direction = None
        self.rotation_target_marker = 1  # 0번(전방) 포착 시 1번(후방)을 찾으러 회전

    def set_planner(self, planner):
        self.planner = planner

    def set_obstacle_checker(self, checker):
        self.obstacle_checker = checker

    def set_phase_controller(self, phase_controller):
        """Phase Controller 설정 (main.py 의존성 유지)"""
        self.phase_controller = phase_controller

    def clear_path(self):
        """경로 초기화"""
        self.path = []

    def get_path(self):
        """현재 경로 반환 (main.py 시각화 오류 해결용)"""
        return self.path

    def update_path(self, center, heading_angle, goal_pos, sonar_dist_cm=999.0):
        """경로 업데이트 및 재계획"""
        if self.use_phase_mode:
            if self.phase_controller:
                self.path = []
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
            
            # 2. 경로 이탈 판단 (50px)
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
        """제어 명령 계산 (180° 진동 오류 및 마커 타입 오류 수정)"""
        if self.wait_counter > 0:
            self.wait_counter -= 1
            return 0.0, 0.0, "STOP"

        if self.use_phase_mode:
            return 0.0, 0.0, "PHASE MODE"

        # [수정 1] 마커 ID 안전하게 추출 (배열 형태로 들어올 경우 대비)
        m_id = None
        if marker_id is not None:
            m_id = int(marker_id[0]) if isinstance(marker_id, (list, np.ndarray)) else int(marker_id)

        # 1. 180° 회전 모드 (Rotation Mode)
        if self.rotation_mode:
            if m_id == self.rotation_target_marker:
                print(f"✅ 마커 {self.rotation_target_marker}번 포착 → 회전 모드 종료")
                self.rotation_mode = False
            else:
                ideal_action = f"TURN {self.rotation_direction}"
                w_speed = 0.18 if self.rotation_direction == "LEFT" else -0.18
                if ideal_action == self.last_action:
                    return 0.0, w_speed, f"{ideal_action} (ROTATION)"
                else:
                    if self.last_action != "STOP":
                        self._trigger_wait()
                        return 0.0, 0.0, "STOP"
                    self.last_action = ideal_action
                    return 0.0, w_speed, ideal_action

        # 2. 일반 주행 모드
        if center is None or not self.path or len(self.path) < 1:
            return 0.0, 0.0, "NO PATH"

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
        # [수정 2] 안전하게 변환된 m_id 사용
        if m_id == 0 and abs(yaw_error) > math.radians(90):
            print(f"🔄 180° 회전 모드 시작 (0번 → 1번 탐색)")
            self.rotation_mode = True
            self.rotation_direction = "LEFT" if yaw_error > 0 else "RIGHT"
            self.rotation_target_marker = 1
            if self.last_action != "STOP":
                self._trigger_wait()
                return 0.0, 0.0, "STOP"
            
            # [수정 3] WAIT 상태 돌입 시 last_action을 업데이트하여 상태 꼬임 방지
            self.last_action = "STOP"
            return 0.0, 0.0, "WAIT ROTATION"

        # 4. 일반 액션 판별
        dead_zone = math.radians(15)
        if abs(yaw_error) < dead_zone:
            ideal_action = "FORWARD"
        else:
            # [수정 4] 180도 부근에서 좌우 진동(Oscillation) 방지용 방향 고정 로직
            if abs(yaw_error) > math.radians(160):
                ideal_action = "TURN LEFT" if yaw_error > 0 else "TURN RIGHT"
                # 이전 액션이 회전이었다면, 각도가 반전되더라도 이전 회전 방향을 강제 유지
                if self.last_action in ["TURN LEFT", "TURN RIGHT"]:
                    ideal_action = self.last_action
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
