import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import math

class BackCamArucoLocalizer(Node):
    def __init__(self):
        super().__init__('back_cam_aruco_localizer')
        
        # 1. 제원 및 카메라 파라미터 설정
        self.marker_size = 0.15 
        self.orig_w = 3456
        self.orig_h = 1934
        
        # 광각 170도 렌즈 초점거리 계산
        fov_rad = math.radians(170)
        self.focal_length = (self.orig_w / 2) / math.tan(fov_rad / 2) 

        # [설정] 후방 카메라 파라미터
        # 위치: 차량 중심 기준 뒤쪽으로 3.45m (좌표계 기준 -3.45)
        # Pitch: 55도 (지면을 내려다보는 각도)
        # Yaw: 차량 후방을 바라보므로 180도 (pi) 혹은 정반대 설정
        self.cam_config = {
            'offset': [0.0, -3.45], # X는 중심, Y는 뒤쪽 3.45m
            'height': 0.9,          # 높이 90cm
            'pitch': math.radians(55.0), 
            'yaw': math.radians(180) # 뒤를 바라봄
        }

        # 2. 영상 파일 및 초기화
        self.cap = cv2.VideoCapture('data/rear.mp4') # 후방 카메라 영상 파일
        
        # 3. ArUco 설정
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        try:
            self.params = cv2.aruco.DetectorParameters()
            self.params.adaptiveThreshWinSizeStep = 10
            self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.params)
            self.is_new_version = True
        except AttributeError:
            self.detector = None
            self.params = cv2.aruco.DetectorParameters_create()
            self.params.adaptiveThreshWinSizeStep = 10
            self.is_new_version = False

        self.rel_pos = None
        self.wheelchair_yaw = 0.0
        self.create_timer(0.05, self.process_loop)

    def process_loop(self):
        ret, frame = self.cap.read()
        if not ret: return

        # 마커 감지 로직 (기존 동일)
        if self.is_new_version:
            corners, ids, _ = self.detector.detectMarkers(frame)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(frame, self.aruco_dict, parameters=self.params)

        display_frame = frame.copy()

        if ids is not None:
            camera_matrix = np.array([
                [self.focal_length, 0, self.orig_w / 2],
                [0, self.focal_length, self.orig_h / 2],
                [0, 0, 1]
            ], dtype=np.float32)

            for i in range(len(ids)):
                hs = self.marker_size / 2
                obj_points = np.array([[-hs, hs, 0], [hs, hs, 0], [hs, -hs, 0], [-hs, -hs, 0]], dtype=np.float32)
                
                # solvePnP로 카메라 기준 마커 포즈 추출
                success, rvec, tvec = cv2.solvePnP(obj_points, corners[i][0].astype(np.float32), 
                                                   camera_matrix, np.zeros((5,1)), flags=cv2.SOLVEPNP_IPPE_SQUARE)

                if success:
                    tx, ty, tz = tvec.flatten()
                    p = self.cam_config['pitch']
                    cy = self.cam_config['yaw']
                    
                    # 1. Pitch 보정 (카메라 좌표 -> 지면 투영 좌표)
                    # 카메라가 아래를 보고 있으므로 tz(깊이)와 ty(높이방향)를 회전시켜 지면상의 z'를 구함
                    ground_z = tz * math.cos(p) - ty * math.sin(p)
                    ground_x = tx
                    
                    # 2. 차량 좌표계 변환 (회전 행렬)
                    # 후방 카메라는 뒤를 보고 있으므로 x, z 축이 차량 전방과 반대입니다.
                    vx = -(ground_x * math.cos(cy) - ground_z * math.sin(cy))
                    vy = ground_x * math.sin(cy) + ground_z * math.cos(cy)
                    
                    # 3. 설치 오프셋 합산
                    self.rel_pos = [vx + self.cam_config['offset'][0], 
                                    vy + self.cam_config['offset'][1]]

                    # 4. 방향(Yaw) 계산
                    rmat, _ = cv2.Rodrigues(rvec)
                    yaw_cam = math.atan2(rmat[0, 2], rmat[2, 2])
                    
                    # 마커 1번이 후면 마커라면 180도 보정이 필요할 수 있습니다.
                    # yaw_cam += math.pi
                    
                    # 차량 좌표계로 Yaw 통합
                    self.wheelchair_yaw = yaw_cam + cy
                    self.wheelchair_yaw = math.atan2(math.sin(self.wheelchair_yaw), 
                                                     math.cos(self.wheelchair_yaw))

                    # 화면 표시
                    text = f"Pos: {self.rel_pos[0]:.2f}, {self.rel_pos[1]:.2f}m"
                    cv2.putText(display_frame, text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 3)

        view_frame = cv2.resize(display_frame, (1280, 720)) 
        cv2.imshow("Back Camera Tracking", view_frame)
        self.draw_mini_map()
        cv2.waitKey(1)

    def draw_mini_map(self):
        m_size = 600
        mini_map = np.ones((m_size, m_size, 3), dtype=np.uint8) * 30
        center = m_size // 2
        scale = 40 

        # 차량 본체 (긴 직사각형)
        cv2.rectangle(mini_map, (center-35, center-200), (center+35, center+200), (100, 100, 100), -1)
        # 카메라 설치 위치 표시 (뒤쪽)
        cv2.circle(mini_map, (center, center + int(3.45 * scale)), 5, (0, 0, 255), -1)

        if self.rel_pos:
            wx = center - int(self.rel_pos[0] * scale)
            wy = center - int(self.rel_pos[1] * scale)
            cv2.circle(mini_map, (wx, wy), 10, (0, 255, 0), -1)
            
            # 휠체어 전방 방향 화살표
            arrow_len = 30
            ax = int(wx - arrow_len * math.sin(self.wheelchair_yaw))
            ay = int(wy - arrow_len * math.cos(self.wheelchair_yaw))
            cv2.arrowedLine(mini_map, (wx, wy), (ax, ay), (255, 255, 255), 2, tipLength=0.3)

        cv2.imshow("Mini-Map", mini_map)

def main():
    rclpy.init()
    node = BackCamArucoLocalizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()