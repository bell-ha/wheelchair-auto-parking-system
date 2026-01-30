import cv2
import numpy as np
import math

from planner import PathPlanner
from tracker import PathTracker
from localization import PoseEstimator
from visualizer import Visualizer


# 카메라 캘리브레이션 파라미터
K = np.array([[601.71923257, 0.0, 630.47700714],
              [0.0, 601.34529853, 367.21223657],
              [0.0, 0.0, 1.0]], dtype=np.float32)

D = np.array([-0.18495647, 0.02541005, -0.01068433, 0.00321714], dtype=np.float32)


class CompactTracker:
    def __init__(self):
        # 맵 설정
        self.map_w, self.map_h = 1200, 1200
        self.grid_w, self.grid_h = 800, 900
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
        
        # 시나리오 (간소화)
        self.parking_mode = True
        self.goals = [
            [(car_cx, car_rear_y+100, -90)],
            [(car_cx, car_rear_y+100, -90)],
            [(car_cx, car_rear_y-30, -90)]
        ]
        self.exit_goals = [
            [(car_cx, car_rear_y+70, -90)],
            [(car_cx-230, self.off_y+400, None), (car_cx+230, self.off_y+400, None)],
            [(car_cx-230, self.off_y+400, None), (car_cx+230, self.off_y+400, None)]
        ]
        
        self.stage, self.goal_idx = 0, 0
        self.exit_choice = 0
        self.goal_selected = False
        
        # 동적 장애물
        self.dynamic_obstacles = []
        
        # 모듈 초기화
        self.planner = PathPlanner(self.map_w, self.map_h, self.wc_w, self.map_scale)
        self.planner.set_obstacle_checker(self.is_obstacle)
        
        self.tracker = PathTracker(self.wc_l, self.map_scale)
        self.tracker.set_planner(self.planner)
        self.tracker.set_obstacle_checker(self.is_obstacle)
        
        self.estimator = PoseEstimator(K, D, self.cams, self.marker_size, self.marker_h, 
                                       self.dist_gain, self.alpha)
        
        self.visualizer = Visualizer(self.map_w, self.map_h, self.car_dim, 
                                     (self.car_x, self.car_y), self.wc_w, self.wc_l, self.map_scale)
        
        # 영상
        self.cap0 = cv2.VideoCapture('../rear_1.mp4')
        self.cap1 = cv2.VideoCapture('../left_1.mp4')
        self.total_frames = int(min(self.cap0.get(cv2.CAP_PROP_FRAME_COUNT), 
                                   self.cap1.get(cv2.CAP_PROP_FRAME_COUNT)))
        
        self.win_name = "Compact Tracker"
        cv2.namedWindow(self.win_name)
        cv2.setMouseCallback(self.win_name, self.mouse_callback)
        cv2.createTrackbar("Frame", self.win_name, 278, self.total_frames-1, self.on_frame)
        cv2.createTrackbar("Mode", self.win_name, 1, 1, self.on_mode)
        cv2.createTrackbar("ExitDir", self.win_name, 0, 1, self.on_exit)
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
    
    def on_mode(self, v):
        self.parking_mode = (v == 1)
        self.stage, self.goal_idx, self.goal_selected = 0, 0, False
        self.tracker.clear_path()
    
    def on_exit(self, v):
        self.exit_choice = v
        if not self.parking_mode and self.stage == 1:
            final = self.exit_goals[2][v][0:2]
            dists = [math.dist(final, g[0:2]) for g in self.exit_goals[1]]
            self.goal_idx = dists.index(min(dists))
    
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
    
    def get_goal(self):
        """현재 목표 반환"""
        goals = self.goals if self.parking_mode else self.exit_goals
        g = goals[self.stage][self.goal_idx]
        return (g[0], g[1]), g[2]
    
    def check_reached(self, pos):
        """목표 도달 확인"""
        gpos, gang = self.get_goal()
        dist = math.dist(pos, gpos)
        if dist < 15:
            if gang is not None:
                angle_diff = abs(math.atan2(
                    math.sin(math.radians(gang) - self.estimator.heading_angle), 
                    math.cos(math.radians(gang) - self.estimator.heading_angle)
                ))
                return angle_diff < math.radians(20)
            return True
        return False
    
    def advance(self):
        """다음 목표로 전환"""
        goals = self.goals if self.parking_mode else self.exit_goals
        if self.goal_idx < len(goals[self.stage]) - 1:
            self.goal_idx += 1
        elif self.stage < len(goals) - 1:
            self.stage += 1
            self.goal_idx = 0
            if not self.parking_mode and self.stage == 1:
                final = self.exit_goals[2][self.exit_choice][0:2]
                dists = [math.dist(final, g[0:2]) for g in self.exit_goals[1]]
                self.goal_idx = dists.index(min(dists))
        
        self.tracker.clear_path()
        self.goal_selected = False
        print(f"🏁 Stage {self.stage} 전환 - 기존 경로 초기화 및 재계획 예약")
    
    def select_nearest(self, pos):
        """가장 가까운 목표 선택"""
        goals = self.goals if self.parking_mode else self.exit_goals
        if self.goal_selected or self.stage != 0:
            return
        dists = [math.dist(pos, g[0:2]) for g in goals[0]]
        self.goal_idx = dists.index(min(dists))
        self.goal_selected = True
    
    def update_path(self):
        """경로 업데이트"""
        if not self.estimator.is_initialized:
            return
        
        gpos, _ = self.get_goal()
        self.tracker.update_path(self.estimator.marker_pos, self.estimator.heading_angle, gpos)
    
    def run(self):
        """메인 루프"""
        play = False
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
            
            # 목표 그리기
            goals = self.goals if self.parking_mode else self.exit_goals
            self.visualizer.draw_goals(img, goals, self.stage, self.goal_idx, self.parking_mode)
            if not self.parking_mode:
                self.visualizer.draw_exit_goals(img, self.exit_goals, self.exit_choice)
            
            # 모니터 프레임
            mon0 = self.f0.copy() if self.f0 is not None else np.zeros((360,640,3), np.uint8)
            mon1 = self.f1.copy() if self.f1 is not None else np.zeros((360,640,3), np.uint8)

            # 포즈 추정
            frames = {'cam0': self.f0, 'cam1': self.f1}
            marker_pos, heading_angle, is_init = self.estimator.detect_and_estimate(frames)
            
            if is_init:
                center = self.estimator.get_center_position(self.wc_l, self.map_scale)
                
                if self.parking_mode and self.stage == 0:
                    self.select_nearest(center)
                if self.check_reached(center):
                    self.advance()
                self.update_path()
                
                # 경로 및 휠체어 그리기
                path = self.tracker.get_path()
                if len(path) >= 2:
                    self.visualizer.draw_path(img, path, marker_pos, heading_angle)
                    self.visualizer.draw_stage_info(img, marker_pos, self.stage)
                
                self.visualizer.draw_wheelchair(img, marker_pos, heading_angle)
            
            # 도움말
            self.visualizer.draw_help_text(img)
            
            cv2.imshow(self.win_name, img)
            cv2.imshow("Monitor", np.hstack([cv2.resize(mon1,(640,360)), 
                                             cv2.resize(mon0,(640,360))]))
            
            key = cv2.waitKey(30) & 0xFF
            if key == ord(' '):
                play = not play
            elif key == ord('q'):
                break
            elif key == ord('c'):
                self.dynamic_obstacles.clear()
                self.update_path()
        
        self.cap0.release()
        self.cap1.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    CompactTracker().run()