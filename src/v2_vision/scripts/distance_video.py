"""
번호판 실측 크기로 카메라~차량 거리를 추정하는 스크립트 (단안 핀홀 모델).

원리: 거리(m) = 실제크기(m) × 초점거리(px) ÷ 박스크기(px)
- 거리는 번호판 "높이" 기준 (비스듬히 접근해도 요각 영향이 적어 폭보다 안정적)
- 요각: YOLO 박스 크롭 안에서 번호판 네 꼭짓점을 찾아 좌/우 변 높이
  비율(키스톤 효과)로 부호 있는 연속값 계산 + EMA. 음수=왼쪽(L), 양수=오른쪽(R)
  꼭짓점 검출 실패 시 폭/높이 비율(acos)로 크기만 표시 "(?)"
- 방위각(bearing): 번호판 가로 중심이 화면 중앙에서 벗어난 각도
  atan((plate_cx - 화면폭/2) / f). 왼쪽 음수 / 오른쪽 양수, EMA 스무딩
- 화면 표시는 YOLO 기본 박스 + 좌상단 HUD 라벨만 (추가 박스 없음)
- [실험, --tail] 요각 보조(후면 한정): 번호판이 두 후미등의 정중앙이라는
  배치를 이용, 좌우 간격 비대칭으로 각도를 추가 표시 "| tail +23deg"
  ※ 실영상 검증 결과 아직 신뢰 불가라 기본 꺼짐:
    근거리는 후미등 렌즈가 입체(모서리 감쌈)라 박스 중심이 밀려 과장,
    원거리는 신호가 거리^2에 반비례해 박스 떨림에 묻힘(±90 포화),
    다중 차량 장면에선 다른 차의 후미등과 짝지어짐.
    살리려면: 차량 단위 그룹핑 + 후미등 안쪽 변 기준 + 실측 간격 필요
- 프레임 간 떨림은 EMA(지수이동평균)로 완화

번호판 규격 (실측 확인: 레이 후면 = 단형 흰색 335×155mm)
- 단형: 335 × 155 mm
- 장형(유럽형): 520 × 110 mm

초점거리(px):
- 기본값: iPhone 13 mini 1x (35mm 환산 26mm) → f = 이미지폭 × 26/36
  * 영상 크롭 때문에 실제보다 작게 잡힐 수 있음 → 캘리브레이션 권장
- 캘리브레이션: 번호판이 정확히 아는 거리에 있는 영상으로
    python3.11 scripts/distance_video.py <영상> <모델.pt> --calib 1.0
  → 계산된 f를 scripts/camera_calib.json에 저장, 이후 자동 사용

사용법:
    python3.11 scripts/distance_video.py raw_videos/레이영상_1.mov models/best_v4_6cls.pt
    python3.11 scripts/distance_video.py <영상> <모델.pt> --conf 0.15
    python3.11 scripts/distance_video.py <영상> <모델.pt> --calib 1.0   # f 보정
결과: outputs/distance/<영상이름>_dist.mp4 (단순 탐지 결과와 분리해 저장)
"""

import argparse
import json
import math
import statistics
from pathlib import Path

import cv2
from ultralytics import YOLO

# ---- 상수 ----
PLATE_W_M = 0.335          # 단형 번호판 폭 (m)
PLATE_H_M = 0.155          # 단형 번호판 높이 (m)
TRUE_AR = PLATE_W_M / PLATE_H_M   # 정면에서의 폭/높이 비율 ≈ 2.16

F35_DEFAULT = 26.0         # iPhone 13 mini 1x, 35mm 환산 초점거리(mm)
CALIB_PATH = Path(__file__).parent / "camera_calib.json"

# 레이 후면: 좌우 후미등 중심 간 실거리(m). 도면 기반 추정치 — 실차 측정 후 교체 권장
TAIL_SPAN_M = 1.30
# 후미등 방식 유효 거리 창. 3m 미만: 후미등이 3D로 튀어나온 형태라 박스 중심이
# 시점에 따라 크게 밀려(모서리 감싸는 렌즈) 각도 과장. 7m 초과: 간격 비대칭
# 신호가 거리^2에 반비례해 박스 떨림에 묻힘
TAIL_MIN_DIST_M = 3.0
TAIL_MAX_DIST_M = 7.0

EMA_ALPHA = 0.3            # 거리 스무딩 계수 (작을수록 부드럽고 반응 느림)

# ---- 레이 top-down 좌표계 (미니맵용) ----
# 번호판 중심 = 원점(0,0), X = 차 폭 방향(후면을 마주보고 오른쪽 +),
# Y = 차 뒤쪽 방향(차에서 멀어질수록 +). 단위 m.
# ※ 도면 기반 추정치 — 실차 측정 후 이 딕셔너리만 교체하면 됨
PART_POS = {
    "license_plate": (0.0, 0.0),
    "tail_light_left": (-0.65, 0.0),   # 후면을 마주보고 왼쪽 후미등
    "tail_light_right": (0.65, 0.0),
}
CAR_SIZE_M = (1.6, 3.6)    # 레이 (폭, 길이) — 미니맵 차체 사각형용
MINIMAP_SIZE = 260         # 미니맵 한 변 (px)
MINIMAP_SCALE = 30         # px per m

# ---- 구역 판정 (검출 클래스 조합 기반) ----
ZONE_CONFIRM_FRAMES = 5    # 새 구역이 이 프레임 수만큼 연속 유지돼야 전환 (깜빡임 방지)
# 우선순위 순서대로 검사, 먼저 맞는 것으로 확정. counts = {클래스이름: 검출 개수}
ZONE_RULES = [
    ("후면",         lambda c: c["license_plate"] >= 1 and c["tail_light"] >= 2),
    ("후측면",       lambda c: c["tail_light"] >= 1 and c["door_handle"] >= 1),
    ("측면",         lambda c: c["side_mirror"] >= 1 and c["door_handle"] >= 1),
    ("측면(불확실)", lambda c: c["side_mirror"] >= 1 or c["door_handle"] >= 1),
]
# 미니맵 표기는 cv2 기본 폰트가 한글을 못 그려서 영문 사용 (콘솔 로그는 한글)
ZONE_HUD_EN = {"후면": "REAR", "후측면": "REAR-SIDE", "측면": "SIDE",
               "측면(불확실)": "SIDE?", "미확인": "UNKNOWN"}


def default_focal_px(image_w: int) -> float:
    """35mm 환산 초점거리에서 픽셀 초점거리 근사 (35mm 필름 가로 = 36mm)."""
    return image_w * F35_DEFAULT / 36.0


def load_focal_px(image_w: int) -> tuple[float, str]:
    """캘리브레이션 파일이 있으면 그 값, 없으면 iPhone 기본값."""
    if CALIB_PATH.exists():
        data = json.loads(CALIB_PATH.read_text())
        # 해상도가 다르면 비례 환산
        f = data["focal_px"] * image_w / data["image_w"]
        return f, f"캘리브레이션({CALIB_PATH.name})"
    return default_focal_px(image_w), f"iPhone 기본값(35mm환산 {F35_DEFAULT}mm)"


def estimate(box_w: float, box_h: float, focal_px: float) -> tuple[float, float]:
    """박스 크기(px) → (거리 m, 요각 deg). 거리는 높이 기준."""
    dist = PLATE_H_M * focal_px / box_h
    # 비스듬히 보면 폭만 cos(yaw)배로 줄어듦 → 비율로 요각 역산
    ar = (box_w / box_h) / TRUE_AR
    yaw = math.degrees(math.acos(min(1.0, ar)))
    return dist, yaw


def find_plate_quad(frame, x1, y1, bw, bh):
    """박스 안에서 번호판 흰 면의 네 꼭짓점을 찾음. 실패하면 None.

    YOLO 박스는 축 정렬 직사각형이라 기울기 정보가 없음 → 크롭 안에서
    실제 번호판 윤곽(사다리꼴)을 찾아야 좌/우 부호를 알 수 있음.
    """
    if bh < 40:  # 너무 작으면 꼭짓점 검출 불안정
        return None
    m = int(max(bw, bh) * 0.04)  # 마진은 최소한만 (크면 흰 차체를 물고 들어감)
    cx1, cy1 = max(0, int(x1) - m), max(0, int(y1) - m)
    cx2 = min(frame.shape[1], int(x1 + bw) + m)
    cy2 = min(frame.shape[0], int(y1 + bh) + m)
    crop = frame[cy1:cy2, cx1:cx2]

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # RETR_LIST: 흰 차체가 번호판을 둘러싸면 번호판 면이 "blob 속 구멍"이 되어
    # 외곽 윤곽(EXTERNAL)만으로는 안 잡힘 → 내부 윤곽까지 모두 검사
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    ch, cw = crop.shape[:2]
    crop_area = ch * cw
    cands = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < crop_area * 0.20:  # 번호판 면은 크롭의 큰 부분을 차지
            continue
        # 윤곽이 완벽한 4각형이 아닐 수 있어 근사 강도를 바꿔가며 시도
        approx = None
        for eps in (0.02, 0.04, 0.06):
            a = cv2.approxPolyDP(c, eps * cv2.arcLength(c, True), True)
            if len(a) == 4 and cv2.isContourConvex(a):
                approx = a
                break
        if approx is None:
            continue
        bx, by, bw2, bh2 = cv2.boundingRect(approx)
        touches = bx <= 2 or by <= 2 or bx + bw2 >= cw - 2 or by + bh2 >= ch - 2
        cands.append((touches, -area, approx.reshape(4, 2)))
    if not cands:
        return None

    # 크롭 경계에 닿는 후보(흰 차체 배경 사각형일 가능성)보다
    # 안쪽에 온전히 들어온 후보를 우선, 같은 조건이면 큰 것
    cands.sort(key=lambda t: (t[0], t[1]))
    pts = cands[0][2].astype(float)
    # 꼭짓점 순서 정렬: tl, tr, br, bl
    s = pts.sum(axis=1)
    d = pts[:, 1] - pts[:, 0]
    tl, br = pts[s.argmin()], pts[s.argmax()]
    tr, bl = pts[d.argmin()], pts[d.argmax()]

    # 검증: 흰 차체까지 물고 들어간 오검출 방지 —
    # YOLO 박스와 크기가 비슷하고 번호판다운 가로세로비여야 채택
    qw = ((tr[0] - tl[0]) + (br[0] - bl[0])) / 2
    qh = ((bl[1] - tl[1]) + (br[1] - tr[1])) / 2
    if qw <= 0 or qh <= 0:
        return None
    if not (0.5 * bw <= qw <= 1.2 * bw and 0.5 * bh <= qh <= 1.2 * bh):
        return None
    # 가로세로비: 비스듬히 보면 정면(2.16)보다 좁아지므로 하한은 느슨하게
    if not 0.6 <= qw / qh <= 4.5:
        return None

    quad = [[p[0] + cx1, p[1] + cy1] for p in (tl, tr, br, bl)]
    return quad


def taillight_yaw(plate, lights, dist_m, focal_px):
    """후미등 간격 비대칭으로 부호 있는 요각 추정. 불가하면 None.

    레이 후면에서 번호판은 두 후미등의 정중앙 → 비스듬히 보면 가까운 쪽
    간격이 넓게 찍힘. 기준선(TAIL_SPAN_M≈1.3m)이 번호판 폭(0.335m)보다
    훨씬 길어서 번호판 꼭짓점이 안 보이는 먼 거리에서도 동작.
    반환: (yaw_deg 부호 있음, 왼쪽 후미등 중심, 오른쪽 후미등 중심)
    """
    px1, py1, pw, ph, _ = plate
    pcx, pcy = px1 + pw / 2, py1 + ph / 2

    # 후보: 번호판 세로 근처(±4×번호판높이)의 후미등을 좌/우로 분리
    lefts, rights = [], []
    for (x1, y1, bw, bh) in lights:
        cx, cy = x1 + bw / 2, y1 + bh / 2
        if abs(cy - pcy) > 4 * ph:
            continue
        (lefts if cx < pcx else rights).append((cx, cy, bh))

    # 짝짓기: 세로 위치(±1.5×번호판높이)와 크기가 비슷한 것끼리.
    # 위 클러스터/아래 범퍼 램프가 각각 쌍이 되고,
    # 다른 차의 후미등은 높이·크기가 안 맞아 여기서 걸러짐
    pairs, used = [], set()
    for lt in sorted(lefts, key=lambda p: p[1]):
        cand = [(i, rt) for i, rt in enumerate(rights) if i not in used
                and abs(rt[1] - lt[1]) < 1.5 * ph
                and lt[2] > 0 and 0.4 < rt[2] / lt[2] < 2.5]
        if not cand:
            continue
        i, rt = min(cand, key=lambda c: abs(c[1][1] - lt[1]))
        used.add(i)
        pairs.append((lt, rt))

    yaws, pts = [], []
    for (lcx, lcy, _), (rcx, rcy, _) in pairs[:2]:
        g_left, g_right = pcx - lcx, rcx - pcx
        if g_left <= pw / 2 or g_right <= pw / 2:  # 겹침/오검출 방어
            continue
        # 물리 검증: 이 거리에서 실측 간격(1.3m)이 찍힐 픽셀 크기와 비교.
        # 다른 차의 후미등이거나 번호판이 잘려 거리가 틀렸으면 크게 어긋남
        k = (g_left + g_right) * dist_m / (focal_px * TAIL_SPAN_M)
        if not 0.35 < k < 1.15:  # cos(요각) 범위 밖 → 불신
            continue
        # r = 왼쪽/오른쪽 간격 비. 가까운 쪽이 넓게 찍히므로
        # r>1이면 왼쪽이 가까움 = 왼쪽에서 보는 중 → 음수
        ratio = g_left / g_right
        sin_t = 2 * dist_m * (1 - ratio) / (TAIL_SPAN_M * (1 + ratio))
        yaws.append(math.degrees(math.asin(max(-1.0, min(1.0, sin_t)))))
        pts.append(((lcx, lcy), (rcx, rcy)))

    if not yaws:
        return None
    if len(yaws) == 2 and abs(yaws[0] - yaws[1]) > 15:  # 두 쌍 불일치 → 불신
        return None
    return sum(yaws) / len(yaws), pts


def plate_yaw(quad, ar_yaw):
    """부호 있는 요각. 크기는 비율(acos) 방식, 부호는 키스톤에서. 불가하면 None.

    가까운 변이 화면에서 더 길게 찍힘(키스톤): 왼쪽 변이 길면 왼쪽에서
    보는 중(L, 음수), 오른쪽 변이 길면 오른쪽(R, 양수).
    변 길이는 유클리드 거리로 계산해 카메라 기울어짐(roll)에도 안전.
    ※ 키스톤으로 각도 "크기"까지 재는 건 꼭짓점 1~2px 오차에 수십 도가
      흔들려서 포기 — 크기는 검증된 폭/높이 비율 방식(ar_yaw)을 사용.
    """
    tl, tr, br, bl = quad
    len_left = math.dist(tl, bl)
    len_right = math.dist(tr, br)
    if len_left <= 0 or len_right <= 0:
        return None
    r = len_left / len_right
    if abs(r - 1) < 0.015:  # 변 길이 차 1.5% 미만 → 사실상 정면
        return 0.0
    return -ar_yaw if r > 1 else ar_yaw


def classify_zone(counts):
    """검출 클래스 조합으로 구역 판정. 매칭 없으면 None (직전 구역 유지용)."""
    for name, rule in ZONE_RULES:
        if rule(counts):
            return name
    return None


def update_zone(state, raw_zone):
    """구역 전환 확정 로직 (깜빡임 방지). state를 제자리 갱신.

    - raw_zone이 None(판정 불가)이면 직전 확정 구역 유지
    - 새 구역은 ZONE_CONFIRM_FRAMES 프레임 연속 관측될 때만 실제 전환
    # TODO: IMU 보간 지점 — 검출이 끊긴 동안(raw_zone=None) IMU 회전량으로
    #       구역을 추정 보간하면 "직전 유지"보다 정확해짐
    반환: (바뀌었으면 이전 구역 이름, 아니면 None)
    """
    if raw_zone is None or raw_zone == state["zone"]:
        state["cand"], state["n"] = None, 0
        return None
    if raw_zone == state["cand"]:
        state["n"] += 1
    else:
        state["cand"], state["n"] = raw_zone, 1
    if state["n"] >= ZONE_CONFIRM_FRAMES:
        old = state["zone"]
        state["zone"] = raw_zone
        state["cand"], state["n"] = None, 0
        return old
    return None


def topdown_pose(dist_m, yaw_deg, bearing_deg, tail_bearings):
    """차 좌표계(top-down)에서 카메라의 위치 (X, Y)와 광축 각도를 계산.

    요각 정의상 번호판→카메라 방향은 (sin yaw, cos yaw) → 기준 위치.
    카메라 광축 각도는 카메라→번호판 방향에서 bearing만큼 되돌려 구함
    (bearing 양수 = 번호판이 화면 오른쪽 = 광축은 번호판 방향보다 왼쪽).
    후미등이 보이면 부품 실좌표(PART_POS) − 카메라→부품 벡터로
    위치를 재추정해 번호판 추정과 평균.
    tail_bearings: [(bearing_deg, "left"|"right"), ...]
    반환: (X, Y, alpha_rad)  — alpha는 차 좌표계에서 광축 방향각
    """
    th = math.radians(yaw_deg)
    bp = math.radians(bearing_deg)
    x0, y0 = dist_m * math.sin(th), dist_m * math.cos(th)
    alpha = math.atan2(-y0, -x0) + bp
    ests = [(x0, y0)]
    for b_deg, side in tail_bearings:
        qx, qy = PART_POS["tail_light_" + side]
        d_i = math.hypot(x0 - qx, y0 - qy)   # 부품까지 거리 (기준 위치로 근사)
        ang = alpha - math.radians(b_deg)    # 카메라→부품 방향 (차 좌표계)
        ests.append((qx - d_i * math.cos(ang), qy - d_i * math.sin(ang)))
    x = sum(e[0] for e in ests) / len(ests)
    y = sum(e[1] for e in ests) / len(ests)
    return x, y, alpha


def draw_minimap(img, zone, side_hint, pose):
    """우하단에 top-down 미니맵 (매 프레임).

    - 차 = 흰 사각형, 지금 보고 있는 구역의 면 = 굵은 하이라이트 선
    - 내 위치(점)·광축(화살표)은 번호판 포즈가 계산된 프레임에만 (pose=None이면 생략)
    - side_hint: 측면 계열 구역에서 좌/우 어느 옆면인지
      (+1 오른쪽 / -1 왼쪽 / None 미상 → 양쪽 다 표시)
    """
    size, s = MINIMAP_SIZE, MINIMAP_SCALE
    ih, iw = img.shape[:2]
    mx, my = iw - size - 20, ih - size - 20      # 미니맵 좌상단
    ox, oy = mx + size // 2, my + 120            # 번호판 원점의 화면 위치

    # 반투명 배경
    roi = img[my:my + size, mx:mx + size]
    dark = roi.copy()
    dark[:] = (30, 30, 30)
    img[my:my + size, mx:mx + size] = cv2.addWeighted(roi, 0.35, dark, 0.65, 0)
    cv2.rectangle(img, (mx, my), (mx + size, my + size), (200, 200, 200), 1)

    # 차체 사각형 (위쪽 = 차 앞 방향), 번호판 원점은 노란 선
    half_w = int(CAR_SIZE_M[0] / 2 * s)
    length = int(CAR_SIZE_M[1] * s)
    cv2.rectangle(img, (ox - half_w, oy - length), (ox + half_w, oy),
                  (255, 255, 255), 1)
    cv2.line(img, (ox - 5, oy), (ox + 5, oy), (0, 255, 255), 3)

    # 구역 하이라이트: 지금 보고 있는 차의 면을 굵은 선으로
    hl = (0, 255, 255)
    sides = [side_hint] if side_hint else [-1, 1]
    if zone == "후면":
        cv2.line(img, (ox - half_w, oy), (ox + half_w, oy), hl, 4)
    elif zone == "후측면":
        cv2.line(img, (ox - half_w, oy), (ox + half_w, oy), hl, 4)
        for sd in sides:
            cv2.line(img, (ox + sd * half_w, oy),
                     (ox + sd * half_w, oy - length // 3), hl, 4)
    elif zone in ("측면", "측면(불확실)"):
        th = 4 if zone == "측면" else 2   # 불확실은 가는 선
        for sd in sides:
            cv2.line(img, (ox + sd * half_w, oy),
                     (ox + sd * half_w, oy - length), hl, th)
    # 구역 이름 (미니맵 상단)
    cv2.putText(img, ZONE_HUD_EN[zone], (mx + 10, my + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
    cv2.putText(img, ZONE_HUD_EN[zone], (mx + 10, my + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, hl, 2)

    if pose is None:
        return
    cam_x, cam_y, alpha = pose
    # 내 위치(점) + 광축(화살표). 미니맵 밖이면 가장자리로 클램프
    u = int(round(ox + cam_x * s))
    v = int(round(oy + cam_y * s))
    u = max(mx + 8, min(mx + size - 8, u))
    v = max(my + 8, min(my + size - 8, v))
    au = int(round(u + 24 * math.cos(alpha)))
    av = int(round(v + 24 * math.sin(alpha)))
    cv2.arrowedLine(img, (u, v), (au, av), (0, 255, 255), 2, tipLength=0.35)
    cv2.circle(img, (u, v), 5, (0, 255, 255), -1)


def main():
    ap = argparse.ArgumentParser(description="번호판 기반 거리 추정")
    ap.add_argument("video")
    ap.add_argument("model")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--calib", type=float, default=None,
                    help="캘리브레이션 모드: 영상 속 번호판까지의 실제 거리(m)")
    ap.add_argument("--tail", action="store_true",
                    help="[실험] 후미등 간격 기반 요각 병기. 다중 차량 장면에서"
                         " 다른 차와 짝지어질 수 있어 기본 꺼짐")
    args = ap.parse_args()

    model = YOLO(args.model)
    # 클래스 id를 이름으로 찾음 (전역 표준: 3=license_plate, 5=tail_light)
    plate_id = next(i for i, n in model.names.items() if n == "license_plate")
    tail_id = next(i for i, n in model.names.items() if n == "tail_light")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"영상을 열 수 없습니다: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    focal_px, focal_src = load_focal_px(w)
    print(f"초점거리: {focal_px:.0f}px ({focal_src})")
    print(f"번호판 규격: {PLATE_W_M*1000:.0f}×{PLATE_H_M*1000:.0f}mm (단형)")

    out_dir = Path(__file__).parent.parent / "outputs" / "distance"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_dist_tail.mp4" if args.tail else "_dist.mp4"
    out_path = str(out_dir / (Path(args.video).stem + suffix))
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"avc1"), fps, (w, h))

    ema_dist = ema_yaw = ema_bearing = None
    dists, calib_heights, tail_frames = [], [], []
    frame_idx = plate_frames = 0
    zone_state = {"zone": "미확인", "cand": None, "n": 0}
    last_side = None   # 미니맵 측면 하이라이트용: 마지막으로 알던 카메라 좌/우 (+1/-1)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # iou=0.5: 같은 부위에 겹치는 중복 박스(후미등 이중 검출 등) 억제
        result = model.predict(frame, conf=args.conf, iou=0.5, verbose=False)[0]
        # 번호판 박스 중 가장 큰 것(=가장 가까운 것) 하나만 사용
        # 단, 화면 경계에 잘린 번호판은 크기가 실제보다 작아 거리가 부풀려지므로 무시
        best, tails, pose = None, [], None
        counts = dict.fromkeys(model.names.values(), 0)  # 구역 판정용 클래스별 개수
        edge = 5
        for b in result.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            counts[model.names[int(b.cls)]] += 1
            if int(b.cls) == tail_id:
                tails.append((x1, y1, x2 - x1, y2 - y1))
            elif int(b.cls) == plate_id:
                if x1 < edge or y1 < edge or x2 > w - edge or y2 > h - edge:
                    continue
                if best is None or (y2 - y1) > best[3]:
                    best = (x1, y1, x2 - x1, y2 - y1, float(b.conf))

        annotated = result.plot()

        if best:
            plate_frames += 1
            x1, y1, bw, bh, bconf = best
            if args.calib is not None:
                calib_heights.append(bh)
            dist, ar_yaw = estimate(bw, bh, focal_px)
            ema_dist = dist if ema_dist is None else (
                EMA_ALPHA * dist + (1 - EMA_ALPHA) * ema_dist)
            dists.append(ema_dist)

            # 방위각(bearing): 번호판이 화면 중심에서 좌우로 벗어난 각도.
            # 왼쪽 음수 / 오른쪽 양수. 거리와 동일한 EMA로 스무딩
            plate_cx = x1 + bw / 2
            bearing = math.degrees(math.atan((plate_cx - w / 2) / focal_px))
            ema_bearing = bearing if ema_bearing is None else (
                EMA_ALPHA * bearing + (1 - EMA_ALPHA) * ema_bearing)

            # 꼭짓점 검출 → 키스톤으로 부호 있는 요각 (실패 시 비율 기반 크기만)
            quad = find_plate_quad(frame, x1, y1, bw, bh)
            s_yaw = plate_yaw(quad, ar_yaw) if quad is not None else None
            if s_yaw is not None:
                ema_yaw = s_yaw if ema_yaw is None else (
                    EMA_ALPHA * s_yaw + (1 - EMA_ALPHA) * ema_yaw)
                if abs(ema_yaw) < 3:
                    yaw_txt = "yaw ~0deg(front)"
                else:
                    yaw_txt = f"yaw {ema_yaw:+.0f}deg({'L' if ema_yaw < 0 else 'R'})"
            else:
                yaw_txt = f"yaw ~{ar_yaw:.0f}deg(?)"

            # 후미등 간격 방식 (원거리 후면 + 후미등 2개 검출 시). 꼭짓점 방식과 병기
            tl_est = (taillight_yaw(best, tails, dist, focal_px)
                      if args.tail and TAIL_MIN_DIST_M <= dist <= TAIL_MAX_DIST_M
                      else None)
            if tl_est is not None:
                t_yaw, pair_pts = tl_est
                for lpt, rpt in pair_pts:
                    for p in (lpt, rpt):
                        cv2.circle(annotated, (int(p[0]), int(p[1])), 10, (255, 0, 255), -1)
                    cv2.line(annotated, (int(lpt[0]), int(lpt[1])),
                             (int(rpt[0]), int(rpt[1])), (255, 0, 255), 2)
                yaw_txt += f" | tail {t_yaw:+.0f}deg"
                tail_frames.append((frame_idx, ema_dist, t_yaw))

            # top-down 포즈: 부호 있는 yaw가 있어야 좌/우 위치가 정해짐.
            # 번호판과 세로로 가까운 후미등만 좌/우 분류해 위치 추정에 합류
            if ema_yaw is not None:
                pcy = y1 + bh / 2
                tail_bearings = []
                for (tx1, ty1, tbw, tbh) in tails:
                    tcx, tcy = tx1 + tbw / 2, ty1 + tbh / 2
                    if abs(tcy - pcy) > 4 * bh:
                        continue
                    side = "left" if tcx < plate_cx else "right"
                    tail_bearings.append(
                        (math.degrees(math.atan((tcx - w / 2) / focal_px)), side))
                pose = topdown_pose(ema_dist, ema_yaw, ema_bearing, tail_bearings)
                # 측면 하이라이트용 좌/우 힌트 (원점 근처에선 유지)
                if abs(pose[0]) > 0.3:
                    last_side = 1 if pose[0] > 0 else -1

            # 추가 박스는 그리지 않음(YOLO 기본 박스로 충분).
            # 라벨은 다른 라벨과 겹치지 않게 화면 좌상단 고정(HUD)
            label = f"{ema_dist:.2f}m  {yaw_txt}  bearing {ema_bearing:+.0f}deg"
            cv2.putText(annotated, label, (24, 64),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 8)
            cv2.putText(annotated, label, (24, 64),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 255, 255), 3)

        # ---- 구역 판정 + 미니맵 (매 프레임, 번호판 유무와 무관) ----
        old = update_zone(zone_state, classify_zone(counts))
        if old is not None:
            print(f"\n[프레임 {frame_idx}] {old} → {zone_state['zone']}")
        draw_minimap(annotated, zone_state["zone"], last_side, pose)

        writer.write(annotated)
        end = "\r" if not (total and frame_idx == total) else "\n"
        print(f"  {frame_idx}{'/' + str(total) if total else ''} 프레임 처리 중...", end=end)

    cap.release()
    writer.release()

    # ---- 캘리브레이션 모드: f 역산 후 저장 ----
    if args.calib is not None:
        if not calib_heights:
            raise SystemExit("번호판이 한 프레임도 안 잡혀 캘리브레이션 불가")
        med_h = statistics.median(calib_heights)
        f = med_h * args.calib / PLATE_H_M
        CALIB_PATH.write_text(json.dumps(
            {"focal_px": f, "image_w": w,
             "note": f"번호판 높이 {PLATE_H_M}m, 기준거리 {args.calib}m로 보정"},
            ensure_ascii=False, indent=2))
        print(f"\n캘리브레이션 완료: f = {f:.0f}px (기존 추정 {focal_px:.0f}px)")
        print(f"저장: {CALIB_PATH} — 이후 실행부터 자동 적용")
        return

    print(f"\n완료: {frame_idx}프레임 중 번호판 검출 {plate_frames}프레임")
    print(f"결과 저장: {out_path}")
    if dists:
        print(f"거리 범위: {min(dists):.2f}m ~ {max(dists):.2f}m "
              f"(중앙값 {statistics.median(dists):.2f}m)")
    if tail_frames:
        print(f"후미등 방식 사용: {len(tail_frames)}프레임 "
              f"(예: " + ", ".join(
                  f"f{i} {d:.1f}m {y:+.0f}deg" for i, d, y in tail_frames[:5]) + ")")


if __name__ == "__main__":
    main()
