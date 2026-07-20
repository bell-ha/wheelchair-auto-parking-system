"""
[세그멘테이션 실험] yolov8n-seg 를 레이 데이터(121장)로 학습.

- 레이 1대 특화 실험 (일반화 목적 아님 — README '실배치 레이 특화' 방향)
- 로컬 맥 MPS 사용. 121장 소규모라 로컬로 충분
- 결과: seg_experiment/runs/seg_v1/weights/best.pt

사용법: python3.10 seg_experiment/train_seg.py
"""

from pathlib import Path

from ultralytics import YOLO

HERE = Path(__file__).parent

model = YOLO("yolov8n-seg.pt")  # COCO 사전학습 seg 백본에서 시작
model.train(
    data=str(HERE / "dataset" / "data.yaml"),
    epochs=100,          # 소규모 데이터라 넉넉히, patience 로 조기종료
    patience=25,
    imgsz=640,
    batch=8,             # MPS 메모리 고려
    device="mps",
    project=str(HERE / "runs"),
    name="seg_v1",
    exist_ok=True,
    seed=42,
)
