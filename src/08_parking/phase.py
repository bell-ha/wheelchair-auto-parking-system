import numpy as np
import math

class UnifiedPhaseController:
    def __init__(self, goal_pos, exit_goal, car_dim):
        self.goal_pos = goal_pos
        self.exit_goal = exit_goal
        self.car_dim = car_dim
        
        # --- [주차 전용] 각도 파라미터 ---
        self.park_angle_p1 = (91.0, 92.0)   # 정렬
        self.park_angle_p3 = (-50.0, -40.0) # 회전 진입
        self.park_angle_p5 = (-95.0, -92.0) # 최종 정렬
        
        # --- [출차 전용] 각도 파라미터 (공통 사용 X, 독립 선언) ---
        self.exit_angle_p2 = (-150.0, -140.0)   # 출차 초기 회전
        self.exit_angle_p4 = (-92.0, -89.0) # 출차 최종 정렬
        
        self.y_tolerance = 5.0
        self.x_tolerance = 1.0
        
        # 명령어
        self.CMD_STOP = "STOP"
        self.CMD_FORWARD = "FORWARD"
        self.CMD_BACKWARD = "BACKWARD"
        self.CMD_LEFT = "TURN LEFT"
        self.CMD_RIGHT = "TURN RIGHT"
        
        self.is_parking = True
        self.reset()

    def reset(self):
        """상태 초기화 - 주차/출차 변수 독립 관리"""
        self.wait_counter = 0
        self.park_step = 1  # 주차 현재 단계 (1~6)
        self.exit_step = 1  # 출차 현재 단계 (1~5)
        self.exit_complete = False
        self.control_mode = "INIT"
        self._update_visualizer_params()

    def set_mode(self, is_parking):
        """모드 전환 시 리셋 필수"""
        self.is_parking = is_parking
        self.reset()
        print(f"🔄 모드 변경: {'주차(PARKING)' if is_parking else '출차(EXIT)'}")

    def _update_visualizer_params(self):
        """Visualizer가 참조하는 p1~p5 값을 현재 모드에 맞춰 즉시 갱신"""
        if self.is_parking:
            self.p1_min, self.p1_max = self.park_angle_p1
            self.p3_min, self.p3_max = self.park_angle_p3
            self.p5_min, self.p5_max = self.park_angle_p5
        else:
            self.p3_min, self.p3_max = -self.exit_angle_p2[1], -self.exit_angle_p2[0]
            self.p1_min, self.p1_max = -self.exit_angle_p4[1], -self.exit_angle_p4[0]
            self.p5_min, self.p5_max = 0.0, 0.0

    def _sync_visualizer_values(self):
        """Visualizer가 참조하는 변수를 현재 모드와 단계에 맞게 강제 동기화"""
        if self.is_parking:
            # 주차 모드일 때 시각화 타겟
            self.p1_min, self.p1_max = self.park_angle_p1
            self.p3_min, self.p3_max = self.park_angle_p3
            self.p5_min, self.p5_max = self.park_angle_p5
        else:
            # 출차 모드일 때 시각화 타겟 (출차 전용 변수 직접 매핑)
            # p3_min/max는 시각화 모듈이 Phase 2에서 참조하도록 설계됨
            self.p3_min, self.p3_max = self.exit_angle_p2 
            # p1_min/max는 시각화 모듈이 Phase 4에서 참조하도록 설계됨
            self.p1_min, self.p1_max = self.exit_angle_p4
            self.p5_min, self.p5_max = 0.0, 0.0

    def check_phase_completion(self, rel_pos, yaw_deg, marker_id, cam_side):
        if self.wait_counter > 0: return

        if self.is_parking:
            # --- 주차 완료 조건 체크 ---
            if self.park_step == 1 and cam_side == 'left' and marker_id == 1:
                if self.park_angle_p1[0] <= yaw_deg <= self.park_angle_p1[1]:
                    self._advance_park(2)
            elif self.park_step == 3 and cam_side == 'back':
                if self.park_angle_p3[0] <= yaw_deg <= self.park_angle_p3[1]:
                    self._advance_park(4)
            elif self.park_step == 5 and cam_side == 'back' and marker_id == 0:
                if self.park_angle_p5[0] <= yaw_deg <= self.park_angle_p5[1]:
                    self._advance_park(6)
        else:
            # --- 출차 완료 조건 체크 (출차 전용 변수 사용) ---
            if self.exit_step == 2:
                target_min, target_max = self.exit_angle_p2
                print(f"DEBUG: Yaw={yaw_deg:.2f}, Range=[{target_min}, {target_max}]")
                print(f"DEBUG: Lower check={yaw_deg >= target_min}, Upper check={yaw_deg <= target_max}")
                if target_min <= yaw_deg <= target_max:
                    self._advance_exit(3)
            elif self.exit_step == 4 and cam_side == 'left':
                if self.exit_angle_p4[0] <= yaw_deg <= self.exit_angle_p4[1]:
                    self._advance_exit(5)

    def _advance_park(self, next_s):
        self.park_step = next_s
        self.wait_counter = 30
        self._sync_visualizer_values()

    def _advance_exit(self, next_s):
        self.exit_step = next_s
        self.wait_counter = 30
        self._sync_visualizer_values()

    def compute_control(self, rel_pos, sonar_dist=999.0):
        if self.wait_counter > 0:
            self.wait_counter -= 1
            return self.CMD_STOP, f"STABILIZING ({self.wait_counter})"
        
        return self._parking_control_logic(rel_pos) if self.is_parking else self._exit_control_logic(rel_pos)

    def _parking_control_logic(self, rel_pos):
        """오직 self.park_step 변수만 사용"""
        action = self.CMD_STOP
        if self.park_step == 1:
            action = self.CMD_LEFT
            self.control_mode = "PARK P1: ALIGNING"
        elif self.park_step == 2:
            if rel_pos:
                dist = abs(rel_pos[1] - self.goal_pos[1])
                if dist <= self.y_tolerance: self._advance_park(3)
                else: action = self.CMD_FORWARD if rel_pos[1] < self.goal_pos[1] else self.CMD_BACKWARD
            self.control_mode = "PARK P2: Y-MOVE"
        elif self.park_step == 3:
            action = self.CMD_LEFT
            self.control_mode = "PARK P3: ROTATING"
        elif self.park_step == 4:
            if rel_pos:
                x_err = rel_pos[0] - 500 - self.goal_pos[0]
                if abs(x_err) <= self.x_tolerance: self._advance_park(5)
                else: action = self.CMD_BACKWARD if x_err > 0 else self.CMD_FORWARD
            self.control_mode = "PARK P4: X-ALIGN"
        elif self.park_step == 5:
            action = self.CMD_LEFT
            self.control_mode = "PARK P5: FINAL ALIGN"
        elif self.park_step == 6:
            if rel_pos:
                dist = abs(rel_pos[1] - self.goal_pos[1])
                if dist <= self.y_tolerance: self.control_mode = "PARKING DONE"
                else: action = self.CMD_BACKWARD if rel_pos[1] > self.goal_pos[1] else self.CMD_FORWARD
            self.control_mode = "PARK P6: FINAL BACK"
        return action, self.control_mode

    def _exit_control_logic(self, rel_pos):
        """오직 self.exit_step 변수만 사용"""
        action = self.CMD_STOP
        if self.exit_step == 1:
            if rel_pos:
                if rel_pos[1] >= self.goal_pos[1]: self._advance_exit(2)
                else: action = self.CMD_BACKWARD
            self.control_mode = "EXIT P1: REVERSING"
        elif self.exit_step == 2:
            action = self.CMD_LEFT
            self.control_mode = "EXIT P2: ROTATING"
        elif self.exit_step == 3:
            if rel_pos:
                if rel_pos[0] <= self.exit_goal[0]: self._advance_exit(4)
                else: action = self.CMD_FORWARD
            self.control_mode = "EXIT P3: X-MOVE"
        elif self.exit_step == 4:
            action = self.CMD_RIGHT
            self.control_mode = "EXIT P4: FINAL ROTATION"
        elif self.exit_step == 5:
            if rel_pos:
                if abs(rel_pos[1] - self.exit_goal[1]) <= self.y_tolerance:
                    self.exit_complete = True
                    self.control_mode = "EXIT SUCCESS"
                else: action = self.CMD_FORWARD
            self.control_mode = "EXIT P5: EXITING"
        return action, self.control_mode

    def get_current_phase(self):
        return self.park_step if self.is_parking else self.exit_step

    def get_goal_pos(self):
        return self.goal_pos if self.is_parking else self.exit_goal
    
    def update_marker_visibility(self, visible):
        """후방 마커 가시성 (주차 전용)"""
        if self.is_parking:
            self.marker1_visible_back = visible