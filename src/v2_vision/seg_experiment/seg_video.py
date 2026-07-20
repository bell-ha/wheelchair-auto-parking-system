"""
[세그멘테이션 실험] seg 모델을 영상 전체에 적용해 마스크 오버레이 영상 생성.

결과: seg_experiment/eval/<영상이름>_seg.mp4

사용법:
    python3.10 seg_experiment/seg_video.py raw_videos/레이영상_1.mov
    python3.10 seg_experiment/seg_video.py <영상> [모델경로] [conf]
    (모델 생략 시 runs/seg_v1/weights/best.pt, conf 기본 0.15)
"""

import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

HERE = Path(__file__).parent


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    video = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else HERE / "runs/seg_v1/weights/best.pt"
    conf = float(sys.argv[3]) if len(sys.argv) > 3 else 0.15

    model = YOLO(str(model_path))
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"영상을 열 수 없습니다: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    out_dir = HERE / "eval"
    out_dir.mkdir(exist_ok=True)
    out_path = str(out_dir / (Path(video).stem + "_seg.mp4"))
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        idx += 1
        res = model.predict(frame, conf=conf, iou=0.5, verbose=False)[0]
        writer.write(res.plot())  # 마스크+박스 오버레이
        end = "\r" if not (total and idx == total) else "\n"
        print(f"  {idx}{'/' + str(total) if total else ''} 프레임 처리 중...", end=end)

    cap.release()
    writer.release()
    print(f"완료: {out_path}")


if __name__ == "__main__":
    main()
