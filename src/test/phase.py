import numpy as np
import math


class PhaseController:
    """단계별 주차 제어 모듈 (ROS2 parking 로직 이식)"""
    
    def __init__(self, goal_pos, car_dim):
        self.goal_pos = goal_pos  # [x, y] 목표 위치
        self.car_dim = car_dim    # [width, length] 차량 크기
        
        # === 단계별 목표 각도 설정 ===
        # Phase 1: -85 ~ -91도 (Left Cam 기준 정렬)
        self.p1_min, self.p1_max = 91.0, 92.0
        # Phase 3: 145 ~ 150도 (Back Cam 기준 회전)
        self.p3_min, self.p3_max = -50.0, -40.0
        # Phase 5: 90 ~ 95도 (Back Cam 기준 최종 정렬)
        self.p5_min, self.p5_max = -95.0, -92.0
        
        # 고정 명령 상수 정의
        self.CMD_STOP = "STOP"
        self.CMD_FORWARD = "FORWARD"
        self.CMD_BACKWARD = "BACKWARD"
        self.CMD_LEFT = "TURN LEFT"
        self.CMD_RIGHT = "TURN RIGHT"
        # 현재 (픽셀 통일)
        self.goal_tolerance = 5.0      # 픽셀
        self.x_align_tolerance = 5.0   # 픽셀
        self.sonar_threshold = 40.0     # 픽셀
        # 상태 플래그
        self.alignment_complete = False
        self.goal_reached = False
        self.phase3_rotating = False
        self.phase3_stopped = False
        self.phase4_complete = False
        self.phase5_complete = False
        self.phase6_complete = False
        self.marker1_visible_back = False
        self.wait_counter = 0  # 단계 전환 시 대기 프레임 수
        
        self.control_mode = "PHASE1_ALIGNING"
    
    def reset(self):
        """상태 초기화"""
        self.alignment_complete = False
        self.goal_reached = False
        self.phase3_rotating = False
        self.phase3_stopped = False
        self.phase4_complete = False
        self.phase5_complete = False
        self.phase6_complete = False
        self.marker1_visible_back = False
        self.control_mode = "PHASE1_ALIGNING"
    
    def update_marker_visibility(self, visible):
        """후방 카메라 마커 1번 가시성 업데이트"""
        self.marker1_visible_back = visible
    
    def check_phase_completion(self, rel_pos, yaw_deg, marker_id, cam_side):
        # [PHASE 1]
        if not self.alignment_complete and cam_side == 'left' and marker_id == 1:
            if self.p1_min <= yaw_deg <= self.p1_max:
                self.alignment_complete = True
                self.wait_counter = 30 # 추가
                print(f"🎯 Phase 1 완료: 1초 정지")
        
        # [PHASE 3 & 5] Back Cam 기반 정렬
        elif self.goal_reached and cam_side == 'back':
            if not self.phase3_stopped:
                if not self.phase3_rotating:
                    self.phase3_rotating = True
                    print(f"🔄 PHASE 3: 마커 1번 포착! 회전을 시작합니다.")
                elif self.phase3_rotating:
                    if self.p3_min <= yaw_deg <= self.p3_max:
                        self.phase3_rotating = False
                        self.phase3_stopped = True
                        self.wait_counter = 30 # 추가
                        print(f"🎯 PHASE 3 완료: 1초 정지")
            
            elif self.phase4_complete and not self.phase5_complete:
                if marker_id == 0:
                    if self.p5_min <= yaw_deg <= self.p5_max:
                        self.phase5_complete = True
                        self.wait_counter = 30 # 추가
                        print(f"🎯 PHASE 5 완료: 1초 정지")
    
    def compute_control(self, rel_pos, sonar_dist_cm):
        """고정 명령(Action) 기반 제어"""
        
        # 1초 대기 로직 (고정 STOP 명령 발행)
        if self.wait_counter > 0:
            self.wait_counter -= 1
            return self.CMD_STOP, f"STABILIZING... ({self.wait_counter})"

        # 현재 루프의 명령 초기값은 STOP
        action = self.CMD_STOP
        
        # PHASE 1: 정렬 (제자리 좌회전 가정)
        if not self.alignment_complete:
            action = self.CMD_LEFT
            self.control_mode = "PHASE 1: ALIGNING"
        
        # PHASE 2: Y축 주행
        elif not self.goal_reached:
            if rel_pos:
                target_y = self.goal_pos[1]
                current_y = rel_pos[1]
                y_diff = abs(current_y - target_y)
                
                if y_diff <= self.goal_tolerance:
                    self.goal_reached = True
                    self.wait_counter = 30
                    action = self.CMD_STOP
                    print(f"🎯 PHASE 2 완료")
                else:
                    if current_y < target_y:
                        action = self.CMD_FORWARD
                        self.control_mode = "PHASE 2: MOVING TO TARGET"
                    else:
                        action = self.CMD_BACKWARD # 오버런 시 후진
                        self.control_mode = "PHASE 2: SEARCHING"
            else:
                action = self.CMD_STOP

        # PHASE 3: 회전
        elif not self.phase3_stopped:
            if self.phase3_rotating:
                action = self.CMD_LEFT # 혹은 CMD_RIGHT (상황에 따라)
            else:
                action = self.CMD_STOP
            self.control_mode = "PHASE 3: ROTATING"
        
        # PHASE 4: X축 정렬
        elif not self.phase4_complete:
            if rel_pos:
                car_cx = self.goal_pos[0]
                current_x = rel_pos[0] - 500
                x_diff = abs(current_x - car_cx)

                if x_diff <= self.x_align_tolerance:
                    self.phase4_complete = True
                    self.wait_counter = 30
                    action = self.CMD_STOP
                else:
                    action = self.CMD_BACKWARD if current_x > 0 else self.CMD_FORWARD
                self.control_mode = f"PHASE 4: X-ALIGN"
        # PHASE 5: 최종 정밀 정렬
        elif not self.phase5_complete:
            action = self.CMD_LEFT # 혹은 CMD_RIGHT (상황에 따라)
            self.control_mode = "PHASE 5: FINAL ALIGN"
                # PHASE 6: 후진 정렬
        elif not self.phase6_complete:
            if rel_pos:
                target_y = self.goal_pos[1]
                current_y = rel_pos[1]
                y_diff = abs(current_y - target_y)
                
                if y_diff <= self.goal_tolerance:
                    self.goal_reached = True
                    self.wait_counter = 30
                    action = self.CMD_STOP
                    print(f"🎯 PHASE 6 완료")
                else:
                    if current_y < target_y:
                        action = self.CMD_BACKWARD # 오버런 시 후진
                        self.control_mode = "PHASE 6: BACK"
                    else:
                        action = self.CMD_FORWARD # 언더런 시 전진
                        self.control_mode = "PHASE 6: FORWARD"
        
        else:
            action = self.CMD_STOP
            self.control_mode = "ALL COMPLETE"
        
        # (명령어, 상태 메시지) 형태로 반환
        return action, self.control_mode
    
    def is_complete(self):
        """전체 주차 완료 여부"""
        return self.phase6_complete
    
    def get_current_phase(self):
        """현재 단계 반환 (1~6)"""
        if not self.alignment_complete:
            return 1
        elif not self.goal_reached:
            return 2
        elif not self.phase3_stopped:
            return 3
        elif not self.phase4_complete:
            return 4
        elif not self.phase5_complete:
            return 5
        elif not self.phase6_complete:
            return 6
        else:
            return 7  # 완료
