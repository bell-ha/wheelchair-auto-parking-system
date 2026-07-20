
import numpy as np
import math
import heapq


class PathPlanner:
    """A* 기반 경로 계획 모듈"""
    
    def __init__(self, map_w, map_h, wc_w, map_scale):
        self.map_w = map_w
        self.map_h = map_h
        self.wc_w = wc_w
        self.map_scale = map_scale
        self.obstacle_checker = None  # 외부에서 설정
    
    def set_obstacle_checker(self, checker):
        """장애물 검사 함수 설정"""
        self.obstacle_checker = checker
    
    def interpolate_path(self, path, interval=30.0):
        """웨이포인트 사이의 간격이 interval보다 크면 중간 점들을 채워넣음"""
        if len(path) < 2:
            return path
        
        new_path = []
        for i in range(len(path) - 1):
            p1 = np.array(path[i])
            p2 = np.array(path[i+1])
            dist = math.dist(p1, p2)
            
            new_path.append(path[i])
            
            # 두 점 사이의 거리가 interval보다 크면 중간에 점 추가
            if dist > interval:
                num_points = int(dist // interval)
                for j in range(1, num_points + 1):
                    # 선형 보간 계산
                    t = j / (num_points + 1)
                    inter_pt = p1 * (1 - t) + p2 * t
                    new_path.append(inter_pt.tolist())
                    
        new_path.append(path[-1])
        return new_path
    
    def astar(self, start, goal):
        """A* 경로 계획 알고리즘"""
        sn, gn = (int(start[0]), int(start[1])), (int(goal[0]), int(goal[1]))
        if self.obstacle_checker and self.obstacle_checker(*sn):
            return [start, goal]

        # 경사각 제한 설정
        ALLOWED_SLOPE = math.radians(20) 
        SLOPE_PENALTY_WEIGHT = 50.0 

        open_l = []
        heapq.heappush(open_l, (0, sn, (0, 0)))
        came, g_s = {}, {sn: 0}

        ROTATION_PENALTY = 100.0

        while open_l:
            _, curr, prev_dir = heapq.heappop(open_l)
            if math.dist(curr, gn) < 25:
                res = [list(curr)]
                while curr in came:
                    curr = came[curr]
                    res.append(list(curr))
                
                simplified = self.simplify_path(res[::-1], epsilon=20.0)
                return self.interpolate_path(simplified, interval=30.0)

            for dx, dy in [(0,12),(0,-12),(12,0),(-12,0),(9,9),(9,-9),(-9,9),(-9,-9)]:
                nb = (curr[0] + dx, curr[1] + dy)
                if not (0 <= nb[0] < self.map_w and 0 <= nb[1] < self.map_h):
                    continue
                if self.obstacle_checker and self.obstacle_checker(*nb):
                    continue
                
                move_cost = math.dist(curr, nb)
                
                # 1. 경사각 페널티 (수직 위주 주행 유도)
                slope_penalty = 0.0
                if dx != 0:
                    current_slope = math.atan2(abs(dx), abs(dy))
                    if current_slope > ALLOWED_SLOPE:
                        slope_penalty = SLOPE_PENALTY_WEIGHT * (current_slope / (math.pi/2))
                
                # 2. 회전 페널티
                rot_penalty = ROTATION_PENALTY if (prev_dir != (0, 0) and prev_dir != (dx, dy)) else 0
                
                # 비용 총합
                tg = g_s[curr] + move_cost + slope_penalty + rot_penalty
                
                if nb not in g_s or tg < g_s[nb]:
                    came[nb], g_s[nb] = curr, tg
                    f_score = tg + math.dist(nb, gn) * 1.5
                    heapq.heappush(open_l, (f_score, nb, (dx, dy)))
        
        return [start, goal]
    
    def simplify_path(self, path, epsilon=5.0):
        """Douglas-Peucker 알고리즘 기반 경로 단순화"""
        if len(path) < 3: 
            return path
        
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