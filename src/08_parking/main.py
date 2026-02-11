import cv2
import numpy as np
import math
import json
import socket
import time

from planner import PathPlanner
from tracker import PathTracker
from localization import PoseEstimator
from visualizer import Visualizer
from phase import UnifiedPhaseController  # 통합 컨트롤러
from control import SERVER_IP, SERVER_PORT, SEND_HZ, FWD_MM_S, REV_MM_S, YAW_MRAD_S


class UdpCommandSender:
    def __init__(self, server_ip, server_port, send_hz):
        self.addr = (server_ip, server_port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.dt = 1.0 / float(send_hz)
        self.last_send = 0.0
        self.last_cmd = (0, 0)

    def _send_payload(self, payload):
        self.sock.sendto(json.dumps(payload).encode("utf-8"), self.addr)

    def send_stop(self, force=False):
        if force:
            self._send_payload({"stop": True})
        self.send_cmd(0, 0)

    def send_cmd(self, v_mm_s, w_mrad_s):
        now = time.time()
        if (now - self.last_send) < self.dt:
            return
        self._send_payload({"v": v_mm_s, "w": w_mrad_s})
        self.last_cmd = (v_mm_s, w_mrad_s)
        self.last_send = now

    def send_action(self, action_cmd):
        if action_cmd in ("STOP", "NO PATH", "STOP (TRANSITION)"):
            self.send_cmd(0, 0)
        elif action_cmd == "FORWARD":
            self.send_cmd(FWD_MM_S, 0)
        elif action_cmd == "BACKWARD":
            self.send_cmd(REV_MM_S, 0)
        elif action_cmd == "TURN LEFT":
            self.send_cmd(0, YAW_MRAD_S)
        elif action_cmd == "TURN RIGHT":
            self.send_cmd(0, -YAW_MRAD_S)
        else:
            self.send_cmd(0, 0)


# 카메라 캘리브레이션 파라미터
K = np.array([[601.71923257, 0.0, 630.47700714],
              [0.0, 601.34529853, 367.21223657],
              [0.0, 0.0, 1.0]], dtype=np.float32)

D = np.array([-0.18495647, 0.02541005, -0.01068433, 0.00321714], dtype=np.float32)


class CompactTracker:
    def __init__(self):
        # 맵 설정
        self.map_w, self.map_h = 1500, 1500
        self.grid_w, self.grid_h = 600, 720
        self.off_x, self.off_y = 200, 150
        self.map_scale = 1.0
        self.wc_w, self.wc_l = 55.0, 66.0
        
        # 마커 및 카메라
        self.marker_size_m, self.marker_h_cm = 0.25, 70.0
        self.car_zone = ((200 + self.off_x, 180 + self.off_y),
                         (400 + self.off_x, 540 + self.off_y))
        self.ramp_zone = ((200 + self.off_x, 540 + self.off_y),
                         (400 + self.off_x, 720 + self.off_y))
        car_cx = (self.car_zone[0][0] + self.car_zone[1][0]) / 2
        car_cy = (self.car_zone[0][1] + self.car_zone[1][1]) / 2

        self.cams = {
            "cam0": {
                "pos_px": np.array([301.4 + self.off_x, 540.0 + self.off_y], dtype=np.float32),
                "h_cm": 105.5,
                "map_angle_deg": 90.0,
                "sens": 1.6,
                "install_angle": 0.0,
                "install_offset": 0.0,
                "yaw_trim_deg": 3.0,
                "dist_gain": 0.90,
                "map_scale": self.map_scale,
                "color": (100, 120, 255)
            },
            "cam1": {
                "pos_px": np.array([200.0 + self.off_x, 270.0 + self.off_y], dtype=np.float32),
                "h_cm": 110.0,
                "map_angle_deg": 157.0,
                "sens": 1.6,
                "install_angle": 113.0,
                "install_offset": 50.84,
                "yaw_trim_deg": 8.0,
                "dist_gain": 0.90,
                "map_scale": self.map_scale,
                "color": (255, 120, 100)
            }
        }
        self.dist_gain, self.angle_gain, self.alpha = 0.90, 1.56, 0.30
        
        # 차량
        self.car_dim = [self.car_zone[1][0] - self.car_zone[0][0],
                self.car_zone[1][1] - self.car_zone[0][1]]
        self.car_x, self.car_y = self.car_zone[0]
        car_rear_y = self.car_y + self.car_dim[1] + 150

        # 윈도우 설정
        self.win_name = "Compact Tracker"
        cv2.namedWindow(self.win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.win_name, 600, 600)
        
        # 시나리오 목표 위치 (픽셀 단위) - 주차/출차/A* 모두 공유
        self.parking_goal = (car_cx, car_rear_y + 100)
        self.exit_goal = (car_cx - 200, car_cy)
        
        # 동적 장애물
        self.dynamic_obstacles = []
        # 초음파 거리
        self.sonar_dist_cm = 999.0
        
        self.phase_controller = UnifiedPhaseController(self.parking_goal, self.exit_goal, self.car_dim)
        
        # 모듈 초기화
        self.planner = PathPlanner(self.map_w, self.map_h, self.wc_w, self.map_scale)
        self.planner.set_obstacle_checker(self.is_obstacle)
        
        self.tracker = PathTracker(self.wc_l, self.map_scale)
        self.tracker.set_planner(self.planner)
        self.tracker.set_obstacle_checker(self.is_obstacle)
        self.tracker.set_phase_controller(self.phase_controller)
        
        
        self.estimator = PoseEstimator(
            K, D, self.cams, self.marker_size_m, self.marker_h_cm,
            self.dist_gain, self.alpha
        )
        
        self.visualizer = Visualizer(self.map_w, self.map_h, self.car_dim, 
                                     (self.car_x, self.car_y), self.wc_w, self.wc_l, self.map_scale, self.ramp_zone)

        # 명령 전송기
        self.cmd_sender = UdpCommandSender(SERVER_IP, SERVER_PORT, SEND_HZ)
        
        # 영상
        self.cap0 = cv2.VideoCapture(0)
        self.cap1 = cv2.VideoCapture(1)
        self.total_frames = int(min(self.cap0.get(cv2.CAP_PROP_FRAME_COUNT), 
                                   self.cap1.get(cv2.CAP_PROP_FRAME_COUNT)))
        cv2.setMouseCallback(self.win_name, self.mouse_callback)
        cv2.createTrackbar("Frame", self.win_name, 0, self.total_frames-1, self.on_frame)
        cv2.createTrackbar("Mode", self.win_name, 0, 3, self.on_mode_toggle) # 0~3으로 확장
    
    def mouse_callback(self, event, x, y, flags, param):
        """마우스 클릭으로 장애물 추가/제거"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dynamic_obstacles.append((x, y, 30))
            print(f"➕ 장애물 추가: ({x}, {y})")
            if self.estimator.is_initialized:
                self.update_path()
        elif event == cv2.EVENT_RBUTTONDOWN:
            for i, (ox, oy, r) in enumerate(self.dynamic_obstacles):
                if math.sqrt((ox-x)**2 + (oy-y)**2) < r:
                    self.dynamic_obstacles.pop(i)
                    print(f"➖ 장애물 제거: ({ox}, {oy})")
                    if self.estimator.is_initialized:
                        self.update_path()
                    break

    def on_frame(self, v):
        self.cap0.set(cv2.CAP_PROP_POS_FRAMES, v)
        self.cap1.set(cv2.CAP_PROP_POS_FRAMES, v)
        _, self.f0 = self.cap0.read()
        _, self.f1 = self.cap1.read()

    def on_mode_toggle(self, v):
        if v == 0: # Parking Phase
            self.phase_controller.set_mode(is_parking=True)
            self.tracker.use_phase_mode = True
        elif v == 1: # Exit Phase
            self.phase_controller.set_mode(is_parking=False)
            self.tracker.use_phase_mode = True
        elif v == 2: # Parking A*
            self.phase_controller.set_mode(is_parking=True)
            self.tracker.use_phase_mode = False
        elif v == 3: # Exit A*
            self.phase_controller.set_mode(is_parking=False)
            self.tracker.use_phase_mode = False
        
        self.tracker.clear_path()
        print(f"Mode Changed to: {v}")
    
    def is_obstacle(self, px, py):
        """장애물 검사"""
        safe_margin = (self.wc_w * self.map_scale / 2) + 15
        
        # 1. 차량 장애물
        if (self.car_x - safe_margin) <= px <= (self.car_x + self.car_dim[0] + safe_margin) and \
           (self.car_y - safe_margin) <= py <= (self.car_y + self.car_dim[1] + safe_margin):
            return True
        
        # 반대로 경사로 자체가 휠체어가 올라가면 안 되는 곳이라면 아래와 같이 추가합니다.
        r_pt1, r_pt2 = self.ramp_zone[0], self.ramp_zone[1]
        if (r_pt1[0] - safe_margin) <= px <= (r_pt2[0] + safe_margin) and \
           (r_pt1[1] - safe_margin) <= py <= (r_pt2[1] + safe_margin):
            return True
            
        # 2. 동적 장애물
        for ox, oy, r in self.dynamic_obstacles:
            dist = math.sqrt((px - ox)**2 + (py - oy)**2)
            if dist < (r + safe_margin):
                return True
        return False
    
    def update_phase_from_detection(self, marker_id, yaw_deg, cam_side):
        """마커 감지 정보로 통합 Phase Controller 업데이트 (안전 장치 추가)"""
        if not self.tracker.use_phase_mode:
            return
        
        # 카메라 이름 매핑 (None일 경우 처리)
        cam_name = None
        if cam_side == 'cam0': cam_name = 'back'
        elif cam_side == 'cam1': cam_name = 'left'
        
        # 마커 좌표가 있을 때와 없을 때 모두 컨트롤러에 신호를 줌
        # (출차 시에는 마커 위치보다 yaw_deg 정렬이 중요하기 때문)
        rel_pos = None
        if self.estimator.marker_pos is not None:
            rel_pos = [self.estimator.marker_pos[0], self.estimator.marker_pos[1]]
        
        # 컨트롤러 호출
        self.phase_controller.check_phase_completion(
            rel_pos, yaw_deg, marker_id, cam_name
        )

    def update_path(self):
        """경로 업데이트 (A* goal은 phase_controller와 공유)"""
        if not self.estimator.is_initialized:
            return
        
        # A* 모드에서도 phase_controller의 goal 사용
        astar_goal = self.phase_controller.get_goal_pos()
        
        self.tracker.update_path(
            self.estimator.marker_pos,
            self.estimator.heading_angle,
            astar_goal,  # phase_controller와 동일한 goal
            self.sonar_dist_cm
        )
    
    def send(self, action_cmd):
        """명령어 전송"""
        print(f"➡️ 명령어 전송: {action_cmd}")
        self.cmd_sender.send_action(action_cmd)

    def run(self):
        """메인 루프"""
        play = True
        
        while True:
            if play:
                ret0, self.f0 = self.cap0.read()
                ret1, self.f1 = self.cap1.read()
                if not ret0 or not ret1:
                    # 영상 끝까지 도달 시 첫 프레임으로 되돌리기 (선택 사항)
                    self.cap0.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
            
            # 1. 맵 생성 및 차량/장애물 기본 시각화
            img = self.visualizer.create_map()
            self.visualizer.draw_car(img)
            self.visualizer.draw_ramp(img)
            self.visualizer.draw_obstacles(img, self.dynamic_obstacles)
            
            # 2. 현재 모드에 따른 목표 지점 표시 (A* 목표 포함)
            if self.phase_controller.is_parking:
                # 주차 모드일 때 녹색 목표 표시
                self.visualizer.draw_parking_goal(img, self.parking_goal)
            else:
                # 출차 모드일 때 주황색 목표 표시
                self.visualizer.draw_exit_goal(img, self.exit_goal)
            
            # 모니터 프레임 복사
            mon0 = self.f0.copy() if self.f0 is not None else np.zeros((360,640,3), np.uint8)
            mon1 = self.f1.copy() if self.f1 is not None else np.zeros((360,640,3), np.uint8)

            # 3. 포즈 추정 및 휠체어 상태 업데이트
            frames = {'cam0': self.f0, 'cam1': self.f1}
            marker_pos, heading_angle, is_init = self.estimator.detect_and_estimate(frames)
            
            current_action = "NO PATH"
            
            # main.py의 run() 메서드 루프 내부
            if is_init:
                yaw_deg = math.degrees(heading_angle)

                # [핵심 수정] 모드에 따라 업데이트 로직을 완전히 분리
                if self.phase_controller.is_parking:
                    # 주차 전용: 후방 마커 가시성 체크
                    back_cam_list = self.estimator.last_det_by_cam.get('cam0', [])
                    self.phase_controller.update_marker_visibility(len(back_cam_list) > 0)
                    
                    # 주차 전용: 마커 1(진입), 0(최종) 업데이트
                    self.update_phase_from_detection(1, yaw_deg, 'cam1')
                    self.update_phase_from_detection(0, yaw_deg, 'cam0')
                else:
                    # 출차 전용: 주차용 마커 ID 대신 현재 감지된 정보를 유연하게 전달
                    # 출차 시에는 특정 마커 ID에 묶이지 않도록 None 혹은 범용 업데이트 호출
                    self.update_phase_from_detection(None, yaw_deg, None)
                
                center = self.estimator.get_center_position()
                self.update_path() # 내부적으로 is_parking에 따라 목표 분리
                
                # --- 제어 명령 및 시각화 가이드 계산 ---
                if self.tracker.use_phase_mode:
                    # Phase 모드 제어
                    rel_pos_pixels = [marker_pos[0], marker_pos[1]]
                    current_action, phase_mode_text = self.phase_controller.compute_control(
                        rel_pos_pixels, self.sonar_dist_cm
                    )
                    self.send(current_action)
                    
                    # 구체적인 Phase 가이드 시각화 (주차/출차 통합 가이드 호출)
                    # exit_goal 인자를 추가하여 출차 시에도 상세 가이드 표시 가능
                    self.visualizer.draw_phase_guidance(
                        img, marker_pos, heading_angle, 
                        self.phase_controller.get_current_phase(),
                        self.phase_controller, 
                        (self.car_x, self.car_y), 
                        self.parking_goal,
                        exit_goal=self.exit_goal,
                        phase_mode_text=phase_mode_text  # <-- 추가
                    )
                else:
                    # A* 모드 제어 (현재 선택된 시나리오 목표를 추종)
                    linear_vel, angular_vel, current_action = self.tracker.compute_action(
                        center, heading_angle
                    )
                    self.send(current_action)
                    
                    # A* 모드 알림 시각화
                    mode_name = "A* (PARKING)" if self.phase_controller.is_parking else "A* (EXITING)"
                    cv2.putText(img, mode_name, (10, 50), 0, 0.7, (0, 100, 255), 2)
                
                # 4. 경로 및 휠체어 렌더링
                path = self.tracker.get_path()
                # A* 모드 혹은 Phase 모드에서 경로가 존재할 때만 드로잉
                if len(path) >= 2:
                    self.visualizer.draw_path(img, path, center, heading_angle)

                self.visualizer.draw_wheelchair(img, center, heading_angle)
                
                # 현재 실행 중인 최종 액션(FORWARD, STOP 등) 표시
                self.visualizer.draw_action_command(img, current_action)
            
            # 도움말 및 상태표시
            self.visualizer.draw_help_text(img)
            cv2.putText(img, "Mode: 0=Parking, 1=Exit, 2=A* | 'r'=Reset | 'c'=Clear", 
                       (10, self.map_h - 10), 0, 0.4, (150, 150, 150), 1)
            
            # 5. 화면 출력 처리 (리사이즈)
            display_scale = 0.6
            img_resized = cv2.resize(img, (int(self.map_w * display_scale), int(self.map_h * display_scale)))
            mon_resized = np.hstack([
                cv2.resize(mon1, (400, 225)), 
                cv2.resize(mon0, (400, 225))
            ])
            
            cv2.imshow(self.win_name, img_resized)
            cv2.imshow("Monitor", mon_resized)
            
            # 키 입력 처리
            key = cv2.waitKey(30) & 0xFF
            if key == ord(' '):
                play = not play
            elif key == ord('q'):
                self.cmd_sender.send_stop(force=True)
                break
            elif key == ord('c'):
                self.dynamic_obstacles.clear()
                self.update_path()
            elif key == ord('r'):
                self.phase_controller.reset()
                self.tracker.clear_path()
                self.tracker.last_action = "STOP"
                self.tracker.wait_counter = 0
                print("🔄 Controller 리셋")
        
        self.cap0.release()
        self.cap1.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    CompactTracker().run()
