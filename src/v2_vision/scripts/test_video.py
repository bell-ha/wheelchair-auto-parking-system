"""
통합(5클래스) 탐지 모델을 영상에 적용해 박스가 그려진 결과 영상을 만들고,
어떤 클래스가 몇 프레임에서 잡혔는지 클래스별로 집계하는 스크립트.

클래스: handle / mirror / plate / logo / fuel_cap

사용법:
    python3 test_general.py 레이영상_1.MOV best.pt 0.25
    (세 번째 인자 conf 생략 시 0.25)
"""

import sys
from collections import defaultdict
from pathlib import Path

import cv2
from ultralytics import YOLO

CONF_THRESHOLD_DEFAULT = 0.25


def main():
    if len(sys.argv) < 3:
        print("사용법: python3 test_general.py 레이영상_1.MOV best.pt 0.25")
        sys.exit(1)

    video_path = sys.argv[1]
    model_path = sys.argv[2]
    conf = float(sys.argv[3]) if len(sys.argv) > 3 else CONF_THRESHOLD_DEFAULT

    print(f"모델 로드 중: {model_path}")
    model = YOLO(model_path)
    print(f"클래스: {model.names}")  # {0:'handle', 1:'mirror', ...} 확인용

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"영상을 열 수 없습니다: {video_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    out_path = str(Path(video_path).with_name(Path(video_path).stem + "_detected.mp4"))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    frame_idx = 0
    detected_count = 0                       # 뭐든 하나라도 잡힌 프레임 수
    frames_with_class = defaultdict(int)     # 클래스별: 해당 클래스가 잡힌 프레임 수
    boxes_per_class = defaultdict(int)       # 클래스별: 누적 박스 수

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = model.predict(frame, conf=conf, verbose=False)[0]
        annotated = result.plot()

        if len(result.boxes) > 0:
            detected_count += 1
            # 이 프레임에 등장한 클래스 id 목록
            cls_ids = result.boxes.cls.int().tolist()
            for cid in set(cls_ids):
                frames_with_class[cid] += 1
            for cid in cls_ids:
                boxes_per_class[cid] += 1

        writer.write(annotated)
        frame_idx += 1
        end = "\r" if not (total_frames and frame_idx == total_frames) else "\n"
        print(f"  {frame_idx}{'/' + str(total_frames) if total_frames else ''} 프레임 처리 중...", end=end)

    cap.release()
    writer.release()

    # ---- 결과 요약 ----
    print(f"\n완료: 총 {frame_idx}프레임 중 {detected_count}프레임에서 무언가 검출됨")
    print(f"결과 저장: {out_path}\n")

    print("클래스별 검출 결과")
    print("-" * 44)
    print(f"{'class':10s} {'검출 프레임':>10s} {'누적 박스':>10s}")
    print("-" * 44)
    for cid, name in model.names.items():
        f = frames_with_class.get(cid, 0)
        b = boxes_per_class.get(cid, 0)
        mark = "" if f > 0 else "   <- 한 번도 안 잡힘"
        print(f"{name:10s} {f:>10d} {b:>10d}{mark}")
    print("-" * 44)


if __name__ == "__main__":
    main()