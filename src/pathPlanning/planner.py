import cv2
import numpy as np
import math
import heapq

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
            'cam1': {'pos': np.array([car_cx-100, car_cy-135]), 'h': 110, 'focal': 841, 'map_angle': 157, 'yaw': 1, 'fov': 45, 'color': (255,120,100)},
            'cam0': {'pos': np.array([car_cx+1.4, car_cy+135]), 'h': 105, 'focal': 836, 'map_angle': 90, 'yaw': 1, 'fov': 45, 'color': (100,120,255)}
        }
        self.dist_gain, self.angle_gain, self.alpha = 1.03, 1.56, 0.75
        self.detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250), cv2.aruco.DetectorParameters())
        
        # 차량
        self.car_dim = [200, 360]
        self.car_x, self.car_y = car_cx - self.car_dim[0]/2, car_cy - self.car_dim[1]/1.6
        car_rear_y = self.car_y + self.car_dim[1] + 150
        
        # 시나리오 (간소화)
        self.parking_mode = True
        self.goals = [
            [(car_cx+240, car_rear_y, None), (car_cx-240, car_rear_y, None)],  # S0: 2개 중 선택
            [(car_cx, car_rear_y-70, -90)],  # S1: 정렬
            [(car_cx, car_rear_y+100, -90)]   # S2: 진입
        ]
        self.exit_goals = [
            [(car_cx, car_rear_y+70, -90)],  # S0: 후진
            [(car_cx+250, car_rear_y, None), (car_cx-250, car_rear_y, None)],  # S1: 경유지
            [(car_cx-250, self.off_y+400, None), (car_cx+250, self.off_y+400, None)]  # S2: 최종 (동적)
        ]
        
        self.stage, self.goal_idx = 0, 0
        self.exit_choice = 0
        self.goal_selected = False
        self.path = []
        
        # 상태
        self.marker_pos, self.heading_angle, self.is_initialized = None, 0.0, False
        
        # 동적 장애물
        self.dynamic_obstacles = []  # [(x, y, radius), ...]
        
        # 영상
        self.cap0 = cv2.VideoCapture('../wheelchairdetect/rear.mp4')
        self.cap1 = cv2.VideoCapture('../wheelchairdetect/left.mp4')
        self.total_frames = int(min(self.cap0.get(cv2.CAP_PROP_FRAME_COUNT), self.cap1.get(cv2.CAP_PROP_FRAME_COUNT)))
        
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
            # 클릭한 위치에 장애물 추가
            self.dynamic_obstacles.append((x, y, 30))  # 반경 30px
            print(f"➕ 장애물 추가: ({x}, {y})")
            # 즉시 경로 재계획 (초기화 상태 무관)
            if self.is_initialized:
                self.update_path()
        elif event == cv2.EVENT_RBUTTONDOWN:
            # 가까운 장애물 제거
            for i, (ox, oy, r) in enumerate(self.dynamic_obstacles):
                if math.sqrt((ox-x)**2 + (oy-y)**2) < r:
                    self.dynamic_obstacles.pop(i)
                    print(f"➖ 장애물 제거: ({ox}, {oy})")
                    if self.is_initialized:
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
        self.path = []
    
    def on_exit(self, v):
        self.exit_choice = v
        if not self.parking_mode and self.stage == 1:
            # 출차 방향에 따라 경유지 선택
            final = self.exit_goals[2][v][0:2]
            dists = [math.dist(final, g[0:2]) for g in self.exit_goals[1]]
            self.goal_idx = dists.index(min(dists))
    
    def is_obstacle(self, px, py):
        # 휠체어의 안전 반경 (휠체어 폭의 절반 + 여유분)
        safe_margin = (self.wc_w * self.map_scale / 2) + 15 
        
        # 1. 차량 장애물 (마진 포함)
        if (self.car_x - safe_margin) <= px <= (self.car_x + self.car_dim[0] + safe_margin) and \
           (self.car_y - safe_margin) <= py <= (self.car_y + self.car_dim[1] + safe_margin):
            return True
            
        # 2. 동적 장애물 (장애물 반경 + 휠체어 안전 반경)
        for ox, oy, r in self.dynamic_obstacles:
            dist = math.sqrt((px - ox)**2 + (py - oy)**2)
            if dist < (r + safe_margin): # 장애물 크기에 휠체어 크기 합산
                return True
        return False

    def astar(self, start, goal):
        # 직선 체크 시 샘플링을 더 촘촘하게 (100단계)
        steps = 100
        clear = True
        for i in range(steps + 1):
            t = i / steps
            px = start[0] * (1 - t) + goal[0] * t
            py = start[1] * (1 - t) + goal[1] * t
            if self.is_obstacle(px, py):
                clear = False
                break
        
        # 장애물이 없으면 직선 반환
        if clear:
            return [start, goal]
        
        # 장애물이 있으면 A* 실행
        sn, gn = (int(start[0]), int(start[1])), (int(goal[0]), int(goal[1]))
        
        # 시작점이 이미 장애물 안이라면 탈출을 위해 주변 가장 가까운 빈 공간을 찾아야 하나, 
        # 여기서는 간단히 직선을 반환하거나 에러 로그를 남깁니다.
        if self.is_obstacle(*sn):
            print("⚠️ 경고: 시작점이 장애물 내부에 있습니다.")
            return [start, goal]

        open_l = []
        heapq.heappush(open_l, (0, sn))
        came, g_s = {}, {sn: 0}
        
        while open_l:
            _, curr = heapq.heappop(open_l)
            
            # 목표 지점 근처 도달 시 (거리 20 이내)
            if math.dist(curr, gn) < 20:
                res = [list(curr)]
                while curr in came:
                    curr = came[curr]
                    res.append(list(curr))
                res.reverse()
                return self.simplify_path(res, epsilon=5.0)

            # 탐색 간격을 조금 더 좁혀서 세밀하게 탐색 (10px -> 8px)
            for dx, dy in [(0,8),(0,-8),(8,0),(-8,0),(6,6),(6,-6),(-6,6),(-6,-6)]:
                nb = (curr[0] + dx, curr[1] + dy)
                if not (0 <= nb[0] < self.map_w and 0 <= nb[1] < self.map_h): continue
                if self.is_obstacle(*nb): continue
                
                tg = g_s[curr] + math.dist(curr, nb)
                if nb not in g_s or tg < g_s[nb]:
                    came[nb], g_s[nb] = curr, tg
                    # 가중치(Heuristic)를 살짝 높여서 장애물 회피 시 목적지 지향성을 강화
                    f_score = tg + math.dist(nb, gn) * 1.2 
                    heapq.heappush(open_l, (f_score, nb))
        
        return [start, goal]
    def simplify_path(self, path, epsilon=5.0):
        """Douglas-Peucker 알고리즘 기반 경로 단순화"""
        if len(path) < 3: 
            return path
        
        # 리스트를 넘파이 배열로 변환 (계산 편의성)
        pts = np.array(path)
        
        def get_dist(p, a, b):
            """점 p와 직선 ab 사이의 거리 계산"""
            if np.array_equal(a, b): 
                return np.linalg.norm(p - a)
            return np.abs(np.cross(b - a, a - p)) / np.linalg.norm(b - a)
        
        # 가장 멀리 떨어진 점 찾기
        dmax, idx = 0, 0
        for i in range(1, len(pts) - 1):
            d = get_dist(pts[i], pts[0], pts[-1])
            if d > dmax:
                idx, dmax = i, d
        
        # 거리가 기준치(epsilon)보다 크면 분할 정복
        if dmax > epsilon:
            left = self.simplify_path(path[:idx+1], epsilon)
            right = self.simplify_path(path[idx:], epsilon)
            return left[:-1] + right
        
        # 기준치보다 작으면 시작점과 끝점만 반환
        return [path[0], path[-1]]
    
    def get_goal(self):
        goals = self.goals if self.parking_mode else self.exit_goals
        g = goals[self.stage][self.goal_idx]
        return (g[0], g[1]), g[2]
    
    def check_reached(self, pos):
        gpos,gang = self.get_goal()
        dist = math.dist(pos, gpos)
        if dist < 15:
            if gang is not None:
                angle_diff = abs(math.atan2(math.sin(math.radians(gang)-self.heading_angle), 
                                           math.cos(math.radians(gang)-self.heading_angle)))
                return angle_diff < math.radians(20)
            return True
        return False
    
    def advance(self):
        goals = self.goals if self.parking_mode else self.exit_goals
        if self.goal_idx < len(goals[self.stage])-1:
            self.goal_idx += 1
        elif self.stage < len(goals)-1:
            self.stage += 1
            self.goal_idx = 0
            # 출차 Stage 1 진입 시
            if not self.parking_mode and self.stage == 1:
                final = self.exit_goals[2][self.exit_choice][0:2]
                dists = [math.dist(final, g[0:2]) for g in self.exit_goals[1]]
                self.goal_idx = dists.index(min(dists))
    
    def select_nearest(self, pos):
        goals = self.goals if self.parking_mode else self.exit_goals
        if self.goal_selected or self.stage != 0:
            return
        dists = [math.dist(pos, g[0:2]) for g in goals[0]]
        self.goal_idx = dists.index(min(dists))
        self.goal_selected = True
    
    def update_path(self):
        if not self.is_initialized:
            print("⏸️ 휠체어 미감지 - 경로 계획 대기")
            return
        
        center = self.marker_pos + np.array([(self.wc_l/2)*self.map_scale*math.cos(self.heading_angle), 
                                             (self.wc_l/2)*self.map_scale*math.sin(self.heading_angle)])
        gpos, _ = self.get_goal()
        
        # Stage 2는 직선 (강제)
        if self.stage == 2:
            self.path = [list(center), list(gpos)]
            print(f"📍 Stage 2: 직선 경로 ({len(self.path)} points)")
        else:
            self.path = self.astar(center, gpos)
            print(f"🗺️ 경로 계획 완료: {len(self.path)} waypoints")
    
    def draw_map(self, img):
        # 그리드
        for i in range(0, self.map_w, 50):
            cv2.line(img, (i,0), (i,self.map_h), (25,25,25), 1)
        for i in range(0, self.map_h, 50):
            cv2.line(img, (0,i), (self.map_w,i), (25,25,25), 1)
        
        # 차량
        cv2.rectangle(img, (int(self.car_x), int(self.car_y)), 
                     (int(self.car_x+self.car_dim[0]), int(self.car_y+self.car_dim[1])), (35,35,45), -1)
        
        # 동적 장애물
        for ox, oy, r in self.dynamic_obstacles:
            cv2.circle(img, (ox, oy), r, (0, 0, 150), -1)
            cv2.circle(img, (ox, oy), r, (0, 0, 255), 2)
        
        # 목표
        goals = self.goals if self.parking_mode else self.exit_goals
        for si, stage_goals in enumerate(goals):
            for gi, g in enumerate(stage_goals):
                gp = (int(g[0]), int(g[1]))
                is_curr = (si == self.stage and gi == self.goal_idx)
                col = (0,255,0) if is_curr else (100,100,100)
                cv2.circle(img, gp, 10, col, -1 if is_curr else 2)
                cv2.putText(img, f"S{si}", (gp[0]-8, gp[1]-15), 0, 0.4, col, 1)
                if g[2] is not None:
                    ax = int(gp[0] + 25*math.cos(math.radians(g[2])))
                    ay = int(gp[1] + 25*math.sin(math.radians(g[2])))
                    cv2.arrowedLine(img, gp, (ax,ay), (150,150,255), 2, tipLength=0.4)
        
        # 출차 최종 목표
        if not self.parking_mode:
            for i, g in enumerate(self.exit_goals[2]):
                gp = (int(g[0]), int(g[1]))
                col = (255,100,0) if i == self.exit_choice else (80,80,80)
                cv2.circle(img, gp, 8, col, -1 if i == self.exit_choice else 2)
    
    def draw_path(self, img):
        if len(self.path) < 2:
            return
        cv2.polylines(img, [np.array(self.path, np.int32)], False, (0,255,255), 2)
        
        # 각도 정보
        pivot = self.marker_pos
        target = self.path[-1]
        dx, dy = target[0]-pivot[0], target[1]-pivot[1]
        target_yaw = math.atan2(dy, dx)
        yaw_err = math.degrees(math.atan2(math.sin(target_yaw-self.heading_angle), 
                                         math.cos(target_yaw-self.heading_angle)))
        
        # 호
        cv2.ellipse(img, (int(pivot[0]), int(pivot[1])), (45,45), 0, 
                   -math.degrees(self.heading_angle), -math.degrees(target_yaw), 
                   (0,200,255) if yaw_err>0 else (255,150,0), 2)
        
        # 텍스트
        cv2.putText(img, f"Rot: {yaw_err:+.1f}deg", (int(pivot[0])+50, int(pivot[1])-70), 
                   0, 0.5, (0,255,255), 1)
        cv2.putText(img, f"Stage: {self.stage}", (int(pivot[0])+50, int(pivot[1])-55), 
                   0, 0.4, (255,200,100), 1)
    
    def run(self):
        play = False
        while True:
            if play:
                ret0, self.f0 = self.cap0.read()
                ret1, self.f1 = self.cap1.read()
                if not ret0 or not ret1:
                    self.on_frame(0)
                    continue
                cv2.setTrackbarPos("Frame", self.win_name, int(self.cap0.get(cv2.CAP_PROP_POS_FRAMES)))
            
            img = np.ones((self.map_h, self.map_w, 3), dtype=np.uint8) * 15
            self.draw_map(img)
            
            mon0 = self.f0.copy() if self.f0 is not None else np.zeros((360,640,3), np.uint8)
            mon1 = self.f1.copy() if self.f1 is not None else np.zeros((360,640,3), np.uint8)
            
            detected = []
            for frame, mon, side in [(self.f0, mon0, 'cam0'), (self.f1, mon1, 'cam1')]:
                if frame is None:
                    continue
                corners, ids, _ = self.detector.detectMarkers(frame)
                if ids is not None:
                    cfg = self.cams[side]
                    c = corners[0].reshape(4,2)
                    px_h = (np.linalg.norm(c[0]-c[3]) + np.linalg.norm(c[1]-c[2]))/2
                    raw_d = (self.marker_size * cfg['focal'])/px_h
                    corr_d = raw_d * (1 + (self.dist_gain-1)*(raw_d/500))
                    d = math.sqrt(max(0, corr_d**2 - abs(cfg['h']-self.marker_h)**2))
                    rel_x = (np.mean(c[:,0]) - frame.shape[1]/2)/(frame.shape[1]/2)
                    t_rad = math.radians(cfg['map_angle'] + cfg['yaw'] + rel_x*cfg['fov']*self.angle_gain)
                    pos = cfg['pos'] + np.array([d*self.map_scale*math.cos(t_rad), 
                                                 d*self.map_scale*math.sin(t_rad)])
                    h = t_rad + math.atan2(c[0][1]-c[3][1], c[0][0]-c[3][0]) - math.pi/2
                    if ids[0][0] == 1:
                        h += math.pi
                    detected.append((pos, h))
            
            if detected:
                ap = np.mean([p[0] for p in detected], axis=0)
                ah = math.atan2(np.mean([math.sin(p[1]) for p in detected]), 
                               np.mean([math.cos(p[1]) for p in detected]))
                if not self.is_initialized:
                    self.marker_pos, self.heading_angle, self.is_initialized = ap, ah, True
                else:
                    self.marker_pos = self.marker_pos*(1-self.alpha) + ap*self.alpha
                    self.heading_angle = math.atan2(
                        math.sin(self.heading_angle)*(1-self.alpha)+math.sin(ah)*self.alpha,
                        math.cos(self.heading_angle)*(1-self.alpha)+math.cos(ah)*self.alpha)
                
                center = self.marker_pos + np.array([(self.wc_l/2)*self.map_scale*math.cos(self.heading_angle), 
                                                     (self.wc_l/2)*self.map_scale*math.sin(self.heading_angle)])
                
                # Stage 0에서만 가까운 목표 선택
                if self.parking_mode and self.stage == 0:
                    self.select_nearest(center)
                
                # 목표 도달 확인
                if self.check_reached(center):
                    self.advance()
                
                self.update_path()
            
            if self.is_initialized:
                self.draw_path(img)
                
                # 휠체어
                center = self.marker_pos + np.array([(self.wc_l/2)*self.map_scale*math.cos(self.heading_angle), 
                                                     (self.wc_l/2)*self.map_scale*math.sin(self.heading_angle)])
                w, l = (self.wc_w*self.map_scale)/2, (self.wc_l*self.map_scale)/2
                rot = np.array([[math.cos(self.heading_angle), -math.sin(self.heading_angle)],
                               [math.sin(self.heading_angle), math.cos(self.heading_angle)]])
                pts = np.dot([[-l,-w],[l,-w],[l,w],[-l,w]], rot.T) + center
                cv2.polylines(img, [pts.astype(np.int32)], True, (0,255,0), 2)
                cv2.line(img, tuple(pts[0].astype(int)), tuple(pts[3].astype(int)), (0,0,255), 3)
                cv2.arrowedLine(img, tuple(self.marker_pos.astype(int)), 
                              (int(self.marker_pos[0]+45*math.cos(self.heading_angle)), 
                               int(self.marker_pos[1]+45*math.sin(self.heading_angle))), 
                              (255,255,255), 2)
            
            # 도움말 표시
            cv2.putText(img, "L-Click: Add Obstacle | R-Click: Remove", (10, 30), 0, 0.5, (200,200,200), 1)
            
            cv2.imshow(self.win_name, img)
            cv2.imshow("Monitor", np.hstack([cv2.resize(mon1,(640,360)), cv2.resize(mon0,(640,360))]))
            
            key = cv2.waitKey(30) & 0xFF
            if key == ord(' '):
                play = not play
            elif key == ord('q'):
                break
            elif key == ord('c'):
                self.dynamic_obstacles.clear()
                print("🗑️ 모든 장애물 제거")
                self.update_path()
        
        self.cap0.release()
        self.cap1.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    CompactTracker().run()
