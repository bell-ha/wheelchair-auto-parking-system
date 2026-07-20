"""
[세그멘테이션 실험 — 본 파이프라인과 분리된 seg_experiment/ 안에서만 동작]

ray_all(탐지용, 박스+폴리곤 혼재)에서 세그멘테이션 학습용 데이터셋 생성.

- 원본 my_data/labeled/ray_all 은 읽기만 함 (건드리지 않음)
- 출력: seg_experiment/dataset/
- 박스 라벨(5컬럼)은 4꼭짓점 사각 폴리곤으로 변환 (세그 학습은 폴리곤만 허용)
- ray_all은 전량이 train에 있고 valid가 비어 있음 → 여기서 85/15 분할
  (시드 고정. 같은 영상 프레임끼리 비슷해서 이상적으론 영상 단위 분할이 맞지만,
   영상이 3개뿐이라 영상 단위로 나누면 val 도메인이 통째로 빠짐 → 프레임 랜덤 분할)

사용법: python3.10 seg_experiment/prepare_seg_dataset.py
"""

import random
import shutil
from pathlib import Path

ROOT = Path(__file__).parent.parent
SRC = ROOT / "my_data" / "labeled" / "ray_all" / "train"
DST = Path(__file__).parent / "dataset"
VAL_RATIO = 0.15
SEED = 42
NAMES = ["car_emblem", "door_handle", "fuel_cap", "license_plate",
         "side_mirror", "tail_light"]


def convert_line(line: str) -> str:
    """박스(cls cx cy w h) → 사각 폴리곤. 이미 폴리곤이면 그대로."""
    parts = line.split()
    if len(parts) != 5:
        return line
    cls, cx, cy, w, h = parts[0], *map(float, parts[1:])
    x1, y1 = max(0.0, cx - w / 2), max(0.0, cy - h / 2)
    x2, y2 = min(1.0, cx + w / 2), min(1.0, cy + h / 2)
    pts = (x1, y1, x2, y1, x2, y2, x1, y2)
    return cls + " " + " ".join(f"{v:.6f}" for v in pts)


def main():
    imgs = sorted((SRC / "images").glob("*.jpg"))
    assert imgs, f"이미지 없음: {SRC / 'images'}"

    random.seed(SEED)
    val_set = set(random.sample(range(len(imgs)), int(len(imgs) * VAL_RATIO)))

    if DST.exists():
        shutil.rmtree(DST)
    stats = {"train": 0, "valid": 0, "box→poly": 0}
    for i, img in enumerate(imgs):
        split = "valid" if i in val_set else "train"
        (DST / split / "images").mkdir(parents=True, exist_ok=True)
        (DST / split / "labels").mkdir(parents=True, exist_ok=True)
        shutil.copy2(img, DST / split / "images" / img.name)

        label = SRC / "labels" / (img.stem + ".txt")
        lines = [l for l in label.read_text().splitlines() if l.strip()]
        stats["box→poly"] += sum(1 for l in lines if len(l.split()) == 5)
        out = "\n".join(convert_line(l) for l in lines) + "\n"
        (DST / split / "labels" / (img.stem + ".txt")).write_text(out)
        stats[split] += 1

    yaml = (f"train: {DST / 'train' / 'images'}\n"
            f"val: {DST / 'valid' / 'images'}\n\n"
            f"nc: {len(NAMES)}\nnames: {NAMES}\n")
    (DST / "data.yaml").write_text(yaml)

    print(f"완료: train {stats['train']}장 / valid {stats['valid']}장")
    print(f"박스→폴리곤 변환: {stats['box→poly']}개")
    print(f"data.yaml: {DST / 'data.yaml'}")


if __name__ == "__main__":
    main()
