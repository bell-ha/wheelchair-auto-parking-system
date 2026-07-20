"""
[세그멘테이션 실험] 학습된 seg 모델을 검증 지점 프레임에 적용.

거리추정_정리.md에 육안 검증된 지점을 사용 (정답을 아는 프레임):
- 영상1 1초  : 후면 오른쪽 45° → yaw 부호 R(+) 이어야 함
- 영상1 10초 : 왼쪽 치우친 후면 → L(-)
- 영상3 20초 : 좁은 통로 오른쪽 비스듬 → R(+)
- 영상2 5초  : 원거리 후면

각 프레임에 대해:
1. 마스크 오버레이 이미지 저장 (눈으로 품질 확인용)
2. 번호판 마스크 → 최소면적사각형 → 좌/우 변 길이 비교로 yaw 부호 계산
   (distance_video.py의 Otsu 방식과 같은 키스톤 원리, 입력만 마스크로 교체)

사용법: python3.10 seg_experiment/eval_seg.py [모델경로]
       (모델 생략 시 runs/seg_v1/weights/best.pt)
"""

import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

HERE = Path(__file__).parent
ROOT = HERE.parent

# (영상, 초, 기대 yaw 부호 설명)
CHECKPOINTS = [
    ("레이영상_1.mov", 1.0, "우측 45deg -> R(+)"),
    ("레이영상_1.mov", 10.0, "좌측 치우침 -> L(-)"),
    ("레이영상_3.mov", 20.0, "우측 비스듬 -> R(+)"),
    ("레이영상_2.MOV", 5.0, "원거리 후면"),
]


def grab_frame(video: Path, sec: float):
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def plate_yaw_from_mask(mask_poly: np.ndarray):
    """번호판 마스크 폴리곤 → (좌/우 변 길이비, 부호 문자열).

    ※ minAreaRect는 항상 완전한 직사각형을 반환해 좌/우 변이 정의상
      같아지므로 키스톤 측정 불가 → convex hull을 4각형으로 근사해
      실제 사다리꼴 꼭짓점을 얻는다.
    r = 왼쪽변/오른쪽변. r>1 → 왼쪽이 가까움 → L(-).
    """
    hull = cv2.convexHull(mask_poly.astype(np.float32))
    quad = None
    for eps in (0.02, 0.04, 0.06, 0.08):
        a = cv2.approxPolyDP(hull, eps * cv2.arcLength(hull, True), True)
        if len(a) == 4:
            quad = a.reshape(4, 2)
            break
    if quad is None:
        return None
    box = quad
    s = box.sum(axis=1)
    d = box[:, 1] - box[:, 0]
    tl, br = box[s.argmin()], box[s.argmax()]
    tr, bl = box[d.argmin()], box[d.argmax()]
    len_l = float(np.linalg.norm(tl - bl))
    len_r = float(np.linalg.norm(tr - br))
    if len_l <= 0 or len_r <= 0:
        return None
    r = len_l / len_r
    if abs(r - 1) < 0.015:
        sign = "~0 (front)"
    else:
        sign = "L(-)" if r > 1 else "R(+)"
    return r, sign


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else HERE / "runs/seg_v1/weights/best.pt"
    model = YOLO(str(model_path))
    plate_id = next(i for i, n in model.names.items() if n == "license_plate")

    out_dir = HERE / "eval"
    out_dir.mkdir(exist_ok=True)
    print(f"모델: {model_path}\n")

    for vid_name, sec, expect in CHECKPOINTS:
        frame = grab_frame(ROOT / "raw_videos" / vid_name, sec)
        if frame is None:
            print(f"[{vid_name} @{sec}s] 프레임 추출 실패")
            continue
        res = model.predict(frame, conf=0.15, iou=0.5, verbose=False)[0]

        tag = f"{Path(vid_name).stem}_{sec:g}s"
        cv2.imwrite(str(out_dir / f"{tag}_masks.jpg"), res.plot())

        n_det = len(res.boxes) if res.boxes is not None else 0
        names = [model.names[int(c)] for c in res.boxes.cls] if n_det else []
        line = f"[{tag}] 기대: {expect} | 검출 {n_det}개 {sorted(set(names))}"

        # 번호판 마스크 → yaw 부호
        if res.masks is not None and n_det:
            plates = [(i, float(res.boxes.conf[i])) for i in range(n_det)
                      if int(res.boxes.cls[i]) == plate_id]
            if plates:
                i = max(plates, key=lambda t: t[1])[0]
                poly = res.masks.xy[i]
                if len(poly) >= 4:
                    out = plate_yaw_from_mask(poly)
                    if out:
                        r, sign = out
                        line += f" | plate mask 변길이비 {r:.3f} -> yaw 부호 {sign}"
                    # 마스크 윤곽 그린 확대 크롭도 저장
                    x, y, w2, h2 = cv2.boundingRect(poly.astype(np.int32))
                    m = 30
                    crop = frame[max(0, y-m):y+h2+m, max(0, x-m):x+w2+m].copy()
                    shifted = poly - [max(0, x-m), max(0, y-m)]
                    cv2.polylines(crop, [shifted.astype(np.int32)], True, (0, 255, 255), 2)
                    cv2.imwrite(str(out_dir / f"{tag}_plate_mask.jpg"), crop)
            else:
                line += " | 번호판 미검출"
        print(line)

    print(f"\n이미지 저장: {out_dir}/")


if __name__ == "__main__":
    main()
