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
        
        # 액션 추적 변수
        self.last_action = "STOP"
        self.wait_counter = 0
        self.min_wait_frames = 15  # 명령 전환 전 정지할 프레임 수
        
        # 180° 회전 모드 상태
        self.rotation_mode = False  # 180° 회전 중인지 여부
        self.rotation_direction = None  # "LEFT" 또는 "RIGHT"
        self.rotation_target_marker = None  # 찾아야 할 마커 ID (1번)
    
    def set_planner(self, planner):
        """경로 계획기 설정"""
        self.planner = planner
    
    def set_obstacle_checker(self, checker):
        """장애물 검사 함수 설정"""
        self.obstacle_checker = checker
    
    def clear_path(self):
        """경로 초기화"""
        self.path = []
    
    def update_path(self, center, heading_angle, goal_pos, sonar_dist_cm=999.0):
        """
        경로 업데이트 및 재계획 (원본 스타일 복원)
        
        Args:
            center: 휠체어 중심 위치 [x, y]
            heading_angle: 현재 방향각 (라디안)
            goal_pos: 목표 위치 [x, y]
            sonar_dist_cm: 초음파 거리 (cm)
        """
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

    def compute_action(self, center, current_yaw, marker_id=1, look_ahead_dist=40.0):
        """
        제어 명령 계산 (180° 회전 모드 포함)
        
        Args:
            center: 휠체어 중심 위치 (마커 기준 계산된 값, None이면 마커 미감지)
            current_yaw: 현재 방향각 (라디안)
            marker_id: 감지된 마커 ID (0=전방, 1=후방, None=미감지)
            look_ahead_dist: Look-ahead 거리
        """
        # 1. 대기 카운터가 동작 중이면 즉시 STOP 반환
        if self.wait_counter > 0:
            self.wait_counter -= 1
            return 0.0, 0.0, "STOP"

        # ──────────────────────────────────────────────────────────
        # 180° 회전 모드가 활성화되어 있으면 마커 없어도 계속 회전
        # ──────────────────────────────────────────────────────────
        if self.rotation_mode:
            # 목표 마커(1번)가 보이면 회전 모드 종료
            if marker_id == self.rotation_target_marker:
                print(f"✅ 마커 {self.rotation_target_marker}번 포착 → 회전 모드 종료")
                self.rotation_mode = False
                self.rotation_direction = None
                self.rotation_target_marker = None
                # 정상 경로 추적으로 전환
            else:
                # 목표 마커가 아직 안 보이면 계속 같은 방향으로 회전
                if self.rotation_direction == "LEFT":
                    ideal_action = "TURN LEFT"
                else:
                    ideal_action = "TURN RIGHT"
                
                # 같은 방향이면 대기 없이 계속 회전
                if ideal_action == self.last_action:
                    self.last_action = ideal_action
                    actions = {
                        "TURN LEFT": (0.0, 0.18),
                        "TURN RIGHT": (0.0, -0.18)
                    }
                    v, w = actions.get(ideal_action, (0.0, 0.0))
                    return v, w, f"{ideal_action} (ROTATION MODE)"
                else:
                    # 방향 전환은 대기 필요
                    if self.last_action != "STOP":
                        self._trigger_wait()
                        return 0.0, 0.0, "STOP"
                    self.last_action = ideal_action
                    actions = {
                        "TURN LEFT": (0.0, 0.18),
                        "TURN RIGHT": (0.0, -0.18)
                    }
                    v, w = actions.get(ideal_action, (0.0, 0.0))
                    return v, w, f"{ideal_action} (ROTATION MODE)"

        # ──────────────────────────────────────────────────────────
        # 일반 모드: 경로가 없거나 마커가 없으면 정지
        # ──────────────────────────────────────────────────────────
        if center is None or not self.path or len(self.path) < 1:
            return 0.0, 0.0, "NO PATH"
        
        # 2. 타겟 방향 선정 (Look-ahead 방식 적용)
        target_pt = self.path[-1]
        for p in self.path:
            if math.dist(center, p) >= look_ahead_dist:
                target_pt = p
                break

        dx, dy = target_pt[0] - center[0], target_pt[1] - center[1]
        target_yaw = math.atan2(dy, dx)
        
        # ── 마커 ID 기반 yaw 보정 ──────────────────────────────
        # 마커 0번(전방)이 보이면 실제 heading은 current_yaw + 180°
        if marker_id == 0:
            actual_yaw = current_yaw + math.pi
        else:
            actual_yaw = current_yaw
        
        yaw_error = math.atan2(math.sin(target_yaw - actual_yaw), 
                               math.cos(target_yaw - actual_yaw))

        # 3. 180° 회전 모드 판단
        # 마커 0번이 보이고 목표가 반대편(±90° 초과)에 있으면 회전 모드 진입
        need_180_turn = (marker_id == 0) and (abs(yaw_error) > math.radians(90))
        
        if need_180_turn:
            # 180° 회전 모드 진입
            print(f"🔄 180° 회전 모드 진입 (마커 0번 → 마커 1번 찾기)")
            self.rotation_mode = True
            self.rotation_target_marker = 1
            self.rotation_direction = "LEFT" if yaw_error > 0 else "RIGHT"
            
            ideal_action = f"TURN {self.rotation_direction}"
        else:
            # 4. 일반 액션 판별
            dead_zone = math.radians(15)  # 직진 허용 각도
            
            if abs(yaw_error) < dead_zone:
                ideal_action = "FORWARD"
            elif yaw_error > 0:
                ideal_action = "TURN LEFT"
            else:
                ideal_action = "TURN RIGHT"

        # 5. 액션 전환 시 대기 처리
        if ideal_action != self.last_action:
            if self.last_action != "STOP":
                self._trigger_wait()
                return 0.0, 0.0, "STOP"

        # 6. 최종 명령 확정 및 상태 저장
        self.last_action = ideal_action
        
        # 액션 매핑 (3가지만)
        actions = {
            "FORWARD": (0.2, 0.0),
            "TURN LEFT": (0.0, 0.18),
            "TURN RIGHT": (0.0, -0.18)
        }
        
        v, w = actions.get(ideal_action, (0.0, 0.0))
        return v, w, ideal_action
