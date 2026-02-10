import cv2
import numpy as np
import math

from planner import PathPlanner
from tracker import PathTracker
from localization import PoseEstimator
from visualizer import Visualizer
from phase import PhaseController


# 카메라 캘리브레이션 파라미터
K = np.array([[601.71923257, 0.0, 630.47700714],
              [0.0, 601.34529853, 367.21223657],
              [0.0, 0.0, 1.0]], dtype=np.float32)

D = np.array([-0.18495647, 0.02541005, -0.01068433, 0.00321714], dtype=np.float32)


class CompactTracker:
    def __init__(self):
        # 맵 설정
        self.map_w, self.map_h = 1000, 1000
        self.grid_w, self.grid_h = 600, 720
        self.off_x, self.off_y = 200, 150
        self.map_scale = 0.5
        self.wc_w, self.wc_l = 57.0, 100.0
        
        # 마커 및 카메라
        self.marker_size, self.marker_h = 25.0, 72.0
        car_cx, car_cy = self.off_x + self.grid_w/2, self.off_y + self.grid_h/2
        self.cams = {
            'cam1': {'pos': np.array([car_cx-140, car_cy-135]), 'h': 110, 'focal': 950, 'map_angle': 157, 'yaw': 1, 'fov': 45, 'color': (255,120,100), 'map_scale': self.map_scale},
            'cam0': {'pos': np.array([car_cx+1.4, car_cy+170]), 'h': 105, 'focal': 950, 'map_angle': 90, 'yaw': 1, 'fov': 45, 'color': (100,120,255), 'map_scale': self.map_scale}
        }
        self.dist_gain, self.angle_gain, self.alpha = 2.0, 1.56, 0.75
        
        # 차량
        self.car_dim = [200, 360]
        self.car_x, self.car_y = car_cx - self.car_dim[0]/2, car_cy - self.car_dim[1]/1.6
        car_rear_y = self.car_y + self.car_dim[1] + 150

        # CompactTracker.__init__ 내부
        self.win_name = "Compact Tracker"
        cv2.namedWindow(self.win_name, cv2.WINDOW_NORMAL) # WINDOW_NORMAL로 변경
        cv2.resizeWindow(self.win_name, 600, 600)        # 초기 창 크기를 600x600으로 설정
        
        # 시나리오 목표 위치 (픽셀 단위)
        self.parking_goal = (car_cx, car_rear_y - 30)  # 최종 주차 위치
        
        # 동적 장애물
        self.dynamic_obstacles = []
        
        # 초음파 거리 (cm)
        self.sonar_dist_cm = 999.0
        
        # === Phase Controller 초기화 ===
        # 픽셀 좌표를 미터로 변환 (ROS2 코드와 매칭)
          # Phase 2 목표 Y 위치
        # 현재: 픽셀 직접 사용
        target_y_offset = 200  # 차량 후방으로부터 200픽셀
        self.target_y = self.car_y + self.car_dim[1] + target_y_offset
        goal_pos_meters = [0.0, self.target_y]
        
        self.phase_controller = PhaseController(goal_pos_meters, self.car_dim)
        
        # 모듈 초기화
        self.planner = PathPlanner(self.map_w, self.map_h, self.wc_w, self.map_scale)
        self.planner.set_obstacle_checker(self.is_obstacle)
        
        self.tracker = PathTracker(self.wc_l, self.map_scale)
        self.tracker.set_planner(self.planner)
        self.tracker.set_obstacle_checker(self.is_obstacle)
        self.tracker.set_phase_controller(self.phase_controller)  # Phase Controller 연결
        
        self.estimator = PoseEstimator(K, D, self.cams, self.marker_size, self.marker_h, 
                                       self.dist_gain, self.alpha)
        
        self.visualizer = Visualizer(self.map_w, self.map_h, self.car_dim, 
                                     (self.car_x, self.car_y), self.wc_w, self.wc_l, self.map_scale)
        
        # 영상
        self.cap0 = cv2.VideoCapture('rear_1.mp4')
        self.cap1 = cv2.VideoCapture('left_1.mp4')
        self.total_frames = int(min(self.cap0.get(cv2.CAP_PROP_FRAME_COUNT), 
                                   self.cap1.get(cv2.CAP_PROP_FRAME_COUNT)))
        
        self.win_name = "Compact Tracker"
        cv2.namedWindow(self.win_name)
        cv2.setMouseCallback(self.win_name, self.mouse_callback)
        cv2.createTrackbar("Frame", self.win_name, 278, self.total_frames-1, self.on_frame)
        cv2.createTrackbar("Mode", self.win_name, 0, 1, self.on_mode_toggle)  # 0=Phase(default), 1=A*
        self.on_frame(278)
    
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
        """모드 토글 - 0=Phase(default), 1=A*"""
        if v == 0:
            # Phase 모드로 전환
            self.tracker.use_phase_mode = True
            self.phase_controller.reset()
            self.tracker.clear_path()
            print("✅ Phase 모드로 전환")
        else:
            # A* 모드로 전환
            self.tracker.use_phase_mode = False
            self.tracker.clear_path()
            print("🔄 A* 모드로 전환")
    
    def is_obstacle(self, px, py):
        """장애물 검사"""
        safe_margin = (self.wc_w * self.map_scale / 2) + 15
        
        # 1. 차량 장애물
        if (self.car_x - safe_margin) <= px <= (self.car_x + self.car_dim[0] + safe_margin) and \
           (self.car_y - safe_margin) <= py <= (self.car_y + self.car_dim[1] + safe_margin):
            return True
            
        # 2. 동적 장애물
        for ox, oy, r in self.dynamic_obstacles:
            dist = math.sqrt((px - ox)**2 + (py - oy)**2)
            if dist < (r + safe_margin):
                return True
        return False
    
    def pixel_to_meters(self, px, py):
        """픽셀 좌표를 미터 좌표로 변환 (차량 중심 기준)"""
        # 차량 중심점 계산
        car_cx = self.car_x + self.car_dim[0] / 2
        car_cy = self.car_y + self.car_dim[1] / 2
        
        # 상대 좌표 (픽셀)
        dx_px = px - car_cx
        dy_px = py - car_cy
        
        # 미터로 변환 (map_scale 고려)
        # 실제 거리 = 픽셀 거리 / map_scale / 스케일 팩터
        # 예: 1미터 = 50픽셀이라고 가정
        PIXEL_PER_METER = 50.0 / self.map_scale
        
        x_m = dx_px / PIXEL_PER_METER
        y_m = -dy_px / PIXEL_PER_METER  # Y축 반전 (이미지 좌표계 → 실제 좌표계)
        
        return [x_m, y_m]
    
    def update_phase_from_detection(self, marker_id, yaw_deg, cam_side):
        """
        마커 감지 정보로 Phase Controller 업데이트 (픽셀 직접 사용)
        """
        if not self.tracker.use_phase_mode:
            return
        
        # 카메라 매핑 (cam0=back, cam1=left)
        cam_name = 'back' if cam_side == 'cam0' else 'left'
        
        # [수정] 미터 변환 없이 픽셀 좌표를 직접 전달
        if self.estimator.marker_pos is not None:
            # 변수 이름(rel_pos_meters)은 유지하되 데이터는 픽셀 값
            rel_pos_meters = [self.estimator.marker_pos[0], self.estimator.marker_pos[1]]
            
            # Phase 완료 조건 체크
            self.phase_controller.check_phase_completion(
                rel_pos_meters, yaw_deg, marker_id, cam_name
            )

    def update_path(self):
        """경로 업데이트 (픽셀 직접 사용)"""
        if not self.estimator.is_initialized:
            return
        
        # [수정] 미터 변환 로직 제거, 픽셀 좌표 사용
        rel_pos_meters = None
        if self.estimator.marker_pos is not None:
            rel_pos_meters = [self.estimator.marker_pos[0], self.estimator.marker_pos[1]]
        
        # PathTracker의 update_path 호출
        self.tracker.update_path(
            self.estimator.marker_pos, 
            self.estimator.heading_angle, 
            self.parking_goal,
            self.sonar_dist_cm
        )
    
    def run(self):
        """메인 루프"""
        play = False
        detected_marker_id = None
        detected_cam_side = None
        
        while True:
            if play:
                ret0, self.f0 = self.cap0.read()
                ret1, self.f1 = self.cap1.read()
                if not ret0 or not ret1:
                    self.on_frame(0)
                    continue
                cv2.setTrackbarPos("Frame", self.win_name, 
                                  int(self.cap0.get(cv2.CAP_PROP_POS_FRAMES)))
            
            # 맵 생성
            img = self.visualizer.create_map()
            self.visualizer.draw_car(img)
            self.visualizer.draw_obstacles(img, self.dynamic_obstacles)
            
            # 주차 목표 그리기
            cv2.circle(img, (int(self.parking_goal[0]), int(self.parking_goal[1])), 12, (0, 255, 0), -1)
            cv2.putText(img, "GOAL", (int(self.parking_goal[0])-20, int(self.parking_goal[1])-20), 
                       0, 0.5, (0, 255, 0), 2)
            
            # 모니터 프레임
            mon0 = self.f0.copy() if self.f0 is not None else np.zeros((360,640,3), np.uint8)
            mon1 = self.f1.copy() if self.f1 is not None else np.zeros((360,640,3), np.uint8)

            # 포즈 추정 (마커 ID와 카메라 정보 추출)
            frames = {'cam0': self.f0, 'cam1': self.f1}
            marker_pos, heading_angle, is_init = self.estimator.detect_and_estimate(frames)
            
            # 현재 액션 초기화
            current_action = "NO PATH"
            linear_vel, angular_vel = 0.0, 0.0
            
            if is_init:
                # --- [핵심 추가] 후방 카메라 마커 가시성 업데이트 ---
                # cam0(후방)에서 검출된 리스트가 비어있지 않으면(0번이든 1번이든) True
                back_cam_list = self.estimator.last_det_by_cam.get('cam0', [])
                any_marker_back = len(back_cam_list) > 0
                self.phase_controller.update_marker_visibility(any_marker_back)
                # ------------------------------------------------

                yaw_deg = math.degrees(heading_angle)
                self.update_phase_from_detection(1, yaw_deg, 'cam1')
                self.update_phase_from_detection(0, yaw_deg, 'cam0')
                
                center = self.estimator.get_center_position(self.wc_l, self.map_scale)
                self.update_path()
                
                # --- 제어 및 액션 계산 시작 ---
                if self.tracker.use_phase_mode and self.phase_controller:
                    # 1. 픽셀 좌표 전달
                    rel_pos_pixels = [marker_pos[0], marker_pos[1]]
                    
                    # 2. PhaseController로부터 직접 '고정 명령어'와 '상태 메시지'를 받음
                    # 이제 리턴값이 (action_cmd, phase_text) 두 개입니다.
                    current_action, phase_mode_text = self.phase_controller.compute_control(
                        rel_pos_pixels, self.sonar_dist_cm
                    )
                    
                    # [추가] 실제 하드웨어나 서버로 명령을 발행해야 한다면 여기서 수행합니다.
                    # self.publisher.send(current_action) 

                    # 3. 시각화 호출 (전달받은 target_y 사용)
                    self.visualizer.draw_phase_guidance(
                        img, 
                        marker_pos, 
                        heading_angle, 
                        self.phase_controller.get_current_phase(),
                        self.phase_controller, 
                        (self.car_x, self.car_y), 
                        self.target_y
                    )
                    
                    # 4. 화면에 현재 액션(FORWARD, STOP 등) 표시
                    self.visualizer.draw_action_command(img, current_action)
                else:
                    # A* 제어 로직 수행
                    linear_vel, angular_vel, current_action = self.tracker.compute_action(
                        center, heading_angle
                    )
                
                # 휠체어 및 경로 렌더링
                path = self.tracker.get_path()
                if not self.tracker.use_phase_mode and len(path) >= 2:
                    self.visualizer.draw_path(img, path, marker_pos, heading_angle)
                
                self.visualizer.draw_wheelchair(img, marker_pos, heading_angle)
                self.visualizer.draw_action_command(img, current_action)
            
            # 도움말
            self.visualizer.draw_help_text(img)
            cv2.putText(img, "Mode Trackbar: 0=Phase(default), 1=A* | 'r'=Reset | 'c'=Clear Obstacles", 
                       (10, self.map_h - 10), 0, 0.4, (150, 150, 150), 1)
            
            # [수정] 출력용 이미지 리사이즈 (0.6배 예시)
            display_scale = 0.6
            img_resized = cv2.resize(img, (int(self.map_w * display_scale), int(self.map_h * display_scale)))
            
            # [수정] 모니터(카메라 뷰) 크기도 너무 크다면 함께 조절
            mon_resized = np.hstack([
                cv2.resize(mon1, (400, 225)), 
                cv2.resize(mon0, (400, 225))
            ])
            
            cv2.imshow(self.win_name, img_resized) # 리사이즈된 이미지 출력
            cv2.imshow("Monitor", mon_resized)    # 리사이즈된 모니터 출력
            
            key = cv2.waitKey(30) & 0xFF
            if key == ord(' '):
                play = not play
            elif key == ord('q'):
                break
            elif key == ord('c'):
                self.dynamic_obstacles.clear()
                self.update_path()
            elif key == ord('r'):
                # Phase Controller 리셋
                self.phase_controller.reset()
                self.tracker.clear_path()
                self.tracker.last_action = "STOP"
                self.tracker.stop_count = 0
                print("🔄 Phase Controller 및 Tracker 리셋")
        
        self.cap0.release()
        self.cap1.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    CompactTracker().run()