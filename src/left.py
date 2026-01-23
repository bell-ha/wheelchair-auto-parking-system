import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import math

class LeftCamArucoLocalizer(Node):
    def __init__(self):
        super().__init__('left_cam_aruco_localizer')
        
        # 1. 제원 설정
        self.marker_size = 0.15 
        self.orig_w = 3456
        self.orig_h = 1934
        
        fov_rad = math.radians(170)
        self.focal_length = (self.orig_w / 2) / math.tan(fov_rad / 2) 

        # [수정] Left 카메라 설정 - Yaw를 0도(정면)가 아닌 측면 방향으로 설정
        # 카메라가 왼쪽을 바라보고 있으므로 차량 전방(+Y) 기준 90도 혹은 -90도 회전이 필요합니다.
        self.cam_config = {
            'offset': [-1.0, -0.7], 
            'height': 1.3,          
            'pitch': math.radians(12.5),
            'yaw': math.radians(-90)  # [핵심 수정] 카메라가 차량 좌측을 향하도록 설정
        }

        # 2. 영상 파일 불러오기 및 시작 시간 설정
        self.cap = cv2.VideoCapture('data/left.mp4')
        
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
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_MSEC, 16000)
            return

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
                
                success, rvec, tvec = cv2.solvePnP(obj_points, corners[i][0].astype(np.float32), 
                                                   camera_matrix, np.zeros((5,1)))

                if success:
                # 1. 카메라 좌표계 위치
                    tx, ty, tz = tvec.flatten()
                    p = self.cam_config['pitch']
                    
                    # [수정 1] 지면 투영 거리 보정 (삼각함수 정밀화)
                    # ty(이미지상 수직 거리)와 tz(깊이)를 조합하여 실제 지면상의 수평 거리를 구함
                    # 카메라가 높이(H)에 있을 때, 실제 거리 d = sqrt(tz^2 + ty^2 + tx^2)를 활용하거나
                    # 아래와 같이 회전 변환 후 Z값을 사용합니다.
                    ground_z = tz * math.cos(p) - ty * math.sin(p)
                    ground_x = tx
                    
                    # [수정 2] 진행 방향 반전 및 축 교정
                    # 실제 Y축 이동이 반대라면 Yaw 각도에 180도를 더하거나 sin/cos 부호를 바꿉니다.
                    cy = self.cam_config['yaw']  # 현재 -90도
                    
                    # 회전 변환 공식 적용
                    vx = ground_x * math.cos(cy) - ground_z * math.sin(cy)
                    vy = ground_x * math.sin(cy) + ground_z * math.cos(cy)
                    
                    # [수정 3] Y축 진행 방향 반전 (실제와 반대일 경우)
                    vy = -vy 
                    
                    x_bias = -0.5
                    y_bias = 1.5
                    # 4. 카메라 설치 오프셋 합산
                    # 겹침 현상이 심하면 offset[0](X) 값을 조절하여 차량 바깥으로 밀어낼 수 있습니다.
                    self.rel_pos = [vx + self.cam_config['offset'][0] + x_bias, 
                                    vy + self.cam_config['offset'][1] + y_bias]

                    # 5. Yaw 계산 및 90도 보정
                    rmat, _ = cv2.Rodrigues(rvec)
                    # 카메라 좌표계에서의 Yaw
                    yaw_cam = math.atan2(rmat[0, 2], rmat[2, 2])
                    
                    # [수정] 현재 시계방향으로 90도 뒤틀려 있다면 +90도(pi/2)를 더해줍니다.
                    # 환경에 따라 -math.pi/2를 해줘야 할 수도 있으니 아래 식을 적용해보세요.
                    self.wheelchair_yaw = yaw_cam + cy - (math.pi / 2)
                    
                    # 각도 정규화 (-PI ~ PI 범위로 유지)
                    self.wheelchair_yaw = math.atan2(math.sin(self.wheelchair_yaw), 
                                                     math.cos(self.wheelchair_yaw))

                    # 디버깅용 텍스트
                    text = f"X: {self.rel_pos[0]:.2f}m, Y: {self.rel_pos[1]:.2f}m"
                    cv2.putText(display_frame, text, (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 255, 0), 5)

        view_frame = cv2.resize(display_frame, (1280, 720)) 
        cv2.imshow("Tracking", view_frame)
        self.draw_mini_map()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            rclpy.shutdown()

    def draw_mini_map(self):
        m_size = 400
        mini_map = np.ones((m_size, m_size, 3), dtype=np.uint8) * 50
        center = m_size // 2
        scale = 50 

        # 자동차 (X축: 좌우, Y축: 상하)
        # ROS 좌표계 기준으로 그리기 위해 방향 주의
        cv2.rectangle(mini_map, (center-40, center-70), (center+40, center+70), (150, 150, 150), -1)

        if self.rel_pos:
            # 그리드 맵 그리기: 자동차 전방이 위(+Y), 왼쪽이 왼쪽(-X)
            wx = center + int(self.rel_pos[0] * scale)
            wy = center - int(self.rel_pos[1] * scale) # OpenCV Y축은 아래가 +이므로 - 처리
            cv2.circle(mini_map, (wx, wy), 8, (0, 255, 0), -1)
            
            # 방향 표시
            ax = int(wx + 25 * math.sin(self.wheelchair_yaw))
            ay = int(wy - 25 * math.cos(self.wheelchair_yaw))
            cv2.line(mini_map, (wx, wy), (ax, ay), (255, 255, 255), 3)

        cv2.imshow("Mini-Map", mini_map)

def main():
    rclpy.init()
    node = LeftCamArucoLocalizer()
    try:
        rclpy.spin(node)
    except Exception:
        pass
    finally:
        node.cap.release()
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()