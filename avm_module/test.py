import cv2
import numpy as np
import math
import time

class DualCamArucoLocalizer:
    def __init__(self):
        # 1. 제원 및 카메라 파라미터 설정
        self.marker_size = 0.25
        self.orig_w = 1280
        self.orig_h = 720
        
        # 광각 170도 렌즈 초점거리 계산
        fov_rad = math.radians(167)
        self.focal_length = (self.orig_w / 2) / math.tan(fov_rad / 2)
        
        # 카메라별 설치 파라미터 (기존 설정 유지)
        self.cam_configs = {
            'left': {
                'offset': [1.3, 0.6],
                'height': 1.15,
                'pitch': math.radians(42),
                'yaw': math.radians(-95)
            },
            'back': {
                'offset': [0.0, -1.8],
                'height': 0.85,
                'pitch': math.radians(40),
                'yaw': math.radians(180)
            }
        }
        
        # 2. 영상 파일 초기화
        self.cap_left = cv2.VideoCapture('data/left.mp4')
        self.cap_back = cv2.VideoCapture('data/rear.mp4')
        
        # 3. ArUco 설정
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
        # OpenCV 버전에 따른 디텍터 설정
        try:
            self.params = cv2.aruco.DetectorParameters()
            self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.params)
            self.is_new_version = True
        except AttributeError:
            self.detector = None
            self.params = cv2.aruco.DetectorParameters_create()
            self.is_new_version = False
        
        # 4. 상태 변수
        self.rel_pos = None
        self.wheelchair_yaw = 0.0
        self.last_active_cam = "None"

    def process_frame(self, frame, cam_side):
        """개별 프레임 처리 로직"""
        if frame is None:
            return None, None
            
        target_idx = -1
        current_marker_id = -1
        detection_result = None
        display_frame = frame.copy()
        
        if self.is_new_version:
            corners, ids, _ = self.detector.detectMarkers(frame)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(frame, self.aruco_dict, parameters=self.params)
        
        if ids is not None:
            detected_ids = ids.flatten()
            if 0 in detected_ids:
                target_idx = np.where(detected_ids == 0)[0][0]
                current_marker_id = 0
            elif 1 in detected_ids:
                target_idx = np.where(detected_ids == 1)[0][0]
                current_marker_id = 1
            
            if target_idx != -1:
                camera_matrix = np.array([
                    [self.focal_length, 0, self.orig_w / 2],
                    [0, self.focal_length, self.orig_h / 2],
                    [0, 0, 1]
                ], dtype=np.float32)
                
                hs = self.marker_size / 2
                obj_points = np.array([[-hs, hs, 0], [hs, hs, 0], [hs, -hs, 0], [-hs, -hs, 0]], dtype=np.float32)
                
                success, rvec, tvec = cv2.solvePnP(obj_points, corners[target_idx][0].astype(np.float32), 
                                                   camera_matrix, np.zeros((5,1)))

                if success:
                    tx, ty, tz = tvec.flatten()
                    config = self.cam_configs[cam_side]
                    p, cy = config['pitch'], config['yaw']
                    
                    ground_z = tz * math.cos(p) - ty * math.sin(p)
                    ground_x = tx
                    
                    # 회전 및 부호 보정
                    vx_tmp = ground_x * math.cos(cy) - ground_z * math.sin(cy)
                    vy_tmp = ground_x * math.sin(cy) + ground_z * math.cos(cy)
                    
                    if cam_side == 'left':
                        vx, vy = vx_tmp, -vy_tmp # 잘 되었던 코드 로직
                    else:
                        vx, vy = -vx_tmp, vy_tmp

                    rel_pos = [vx + config['offset'][0], vy + config['offset'][1]]

                    # Yaw 계산
                    rmat, _ = cv2.Rodrigues(rvec)
                    yaw_cam = math.atan2(rmat[0, 2], rmat[2, 2])
                    if current_marker_id == 1: yaw_cam += math.pi

                    if cam_side == 'left':
                        w_yaw = yaw_cam + cy - (math.pi / 2)
                    else:
                        w_yaw = yaw_cam + cy

                    w_yaw = math.atan2(math.sin(w_yaw), math.cos(w_yaw))

                    detection_result = {'pos': rel_pos, 'yaw': w_yaw}
                    
                    # 프레임에 축 그리기
                    cv2.drawFrameAxes(display_frame, camera_matrix, np.zeros((5,1)), rvec, tvec, 0.2)

        return display_frame, detection_result

    def draw_mini_map(self):
        """통합 미니맵 시각화 (카메라 위치 L, B 표시 포함)"""
        m_size = 600
        mini_map = np.ones((m_size, m_size, 3), dtype=np.uint8) * 30
        center = m_size // 2
        scale = 40  # 1m 당 40픽셀
        
        # 1. 그리드 선 (1m 간격)
        grid_step = 40
        for i in range(0, m_size, grid_step):
            cv2.line(mini_map, (i, 0), (i, m_size), (50, 50, 50), 1)
            cv2.line(mini_map, (0, i), (m_size, i), (50, 50, 50), 1)
        
        # 2. 차량 본체 (회색 직사각형)
        # 실제 차량 크기에 맞춰 조정 가능 (예: 가로 0.7m, 세로 1m 가정)
        cv2.rectangle(mini_map, (center - 30, center - 72), (center + 30, center + 72), (100, 100, 100), -1)
        
        ramp_top = center + 72
        ramp_bottom = ramp_top + 72
        cv2.rectangle(mini_map, (center - 30, ramp_top), (center + 30, ramp_bottom), (200, 150, 0), 2)
        # 경사로임을 알리기 위한 빗금 또는 텍스트 추가 (선택 사항)
        cv2.putText(mini_map, "RAMP", (center - 20, ramp_top + 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 150, 0), 1)
        
        # 3. 카메라 설치 위치 표시
        for cam_side, config in self.cam_configs.items():
            # 차량 좌표계 offset을 미니맵 좌표로 변환
            # offset[0]은 X(좌우), offset[1]은 Y(전후)
            cam_x = center - int(config['offset'][0] * scale)
            cam_y = center - int(config['offset'][1] * scale)
            
            if cam_side == 'left':
                color = (255, 0, 0)  # 파란색 (Left)
                label = "L"
            else:
                color = (0, 0, 255)  # 빨간색 (Back)
                label = "B"
                
            cv2.circle(mini_map, (cam_x, cam_y), 6, color, -1)
            cv2.putText(mini_map, label, (cam_x + 10, cam_y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 4. 휠체어 현재 위치 및 방향 표시
        if self.rel_pos:
            wx = center - int(self.rel_pos[0] * scale)
            wy = center - int(self.rel_pos[1] * scale)
            
            # 휠체어 본체 (초록색)
            cv2.circle(mini_map, (wx, wy), 12, (0, 255, 0), -1)
            
            # 방향 화살표
            arrow_len = 35
            ax = int(wx - arrow_len * math.sin(self.wheelchair_yaw))
            ay = int(wy - arrow_len * math.cos(self.wheelchair_yaw))
            cv2.arrowedLine(mini_map, (wx, wy), (ax, ay), (255, 255, 255), 3, tipLength=0.3)
            
            # 현재 위치 좌표 출력
            pos_text = f"Pos: [{self.rel_pos[0]:.2f}, {self.rel_pos[1]:.2f}]m"
            cv2.putText(mini_map, pos_text, (wx + 15, wy + 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # 5. 상태 표시
        cv2.putText(mini_map, f"Active: {self.last_active_cam}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow("Integrated Mini-Map", mini_map)

    def run(self):
        """메인 실행 루프 (rclpy.spin 대체)"""
        print("프로그램 시작 (종료하려면 'q'를 누르세요)")
        
        while True:
            ret_l, frame_l = self.cap_left.read()
            ret_b, frame_b = self.cap_back.read()
            
            # 영상 반복 재생 설정
            if not ret_l:
                self.cap_left.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            if not ret_b:
                self.cap_back.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            # 프레임 처리
            disp_l, res_l = self.process_frame(frame_l, 'left')
            disp_b, res_b = self.process_frame(frame_b, 'back')

            if res_l:
                self.rel_pos, self.wheelchair_yaw, self.last_active_cam = res_l['pos'], res_l['yaw'], "Left"
            if res_b:
                self.rel_pos, self.wheelchair_yaw, self.last_active_cam = res_b['pos'], res_b['yaw'], "Back"

            # 화면 출력
            if disp_l is not None: cv2.imshow("Left Camera", cv2.resize(disp_l, (640, 360)))
            if disp_b is not None: cv2.imshow("Back Camera", cv2.resize(disp_b, (640, 360)))
            
            self.draw_mini_map()

            # 키 입력 대기 (약 20 FPS 유지)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

        self.cap_left.release()
        self.cap_back.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    localizer = DualCamArucoLocalizer()
    localizer.run()