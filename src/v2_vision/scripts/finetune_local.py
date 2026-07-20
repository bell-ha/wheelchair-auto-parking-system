"""
로컬(M2 Max / MPS) fine-tuning 스크립트.

목적:
  - 기존 통합 모델(best.pt)을 출발점으로, 신형 번호판 데이터를 추가해 번호판 성능을 보강.
  - 나머지 4종(handle/mirror/logo/fuel_cap)은 '리허설 서브셋'(클래스당 일부만)으로 유지 -> 망각 방지 + 학습 경량화.
  - 번호판(기존+신형)은 전량 사용 -> 번호판을 집중 보강.
  - 4종을 소량만 넣으면 미러 이미지 속 '라벨 안 된 번호판=배경' 억제 신호도 크게 줄어듦(부수 효과).

준비물(이 스크립트와 같은 폴더에 두세요):
  best.pt
  car-door-handle.v4i.yolov8.zip
  Side Mirror.v2-first-version.yolov8.zip
  license-plate-label.v5i.yolov8.zip       (기존 번호판)
  license plate.v1i.yolov8.zip             (신형 번호판)
  KIA Logo.v2i.yolov8.zip
  Fuel Cap.v15i.yolov8.zip

사용법:
  python3 finetune_local.py           # 기본 20 에포크로 fine-tuning
  python3 finetune_local.py 1         # 먼저 1 에포크만 -> 실제 소요시간 측정용(권장 첫 실행)

맥이 자지 않게(권장):
  caffeinate -i python3 finetune_local.py
"""

import sys
import shutil
import zipfile
import random
from pathlib import Path

import yaml

try:
    import torch
    from ultralytics import YOLO
except ImportError:
    print("먼저 설치하세요:  pip3 install ultralytics")
    sys.exit(1)

# ============================================================
# 설정
# ============================================================
BASE = Path(__file__).resolve().parent          # zip과 best.pt가 있는 폴더(=이 스크립트 위치)
WORK = BASE / "_ft_src"                          # 소스별 압축 해제 위치
MERGED = BASE / "_ft_merged"                     # fine-tuning용 병합 데이터셋
BEST = BASE / "best.pt"                          # 출발점 모델

REHEARSAL_PER_CLASS = 400   # 4종은 클래스당 최대 이만큼만 사용(망각 방지 + 경량화)
VAL_RATIO = 0.15
SEED = 42
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
IMGSZ = 640                 # 영상에서 번호판이 크게 나오므로 640으로 충분
BATCH = 16                  # 메모리 부족(MPS OOM) 시 8로 낮추세요

# 전역 클래스 매핑
GLOBAL_NAMES = ["handle", "mirror", "plate", "logo", "fuel_cap"]

# 소스 정의
#   keep: "all" -> 모든 클래스를 이 전역 클래스로 통합
#         "name:X" -> 이름 X 클래스만 유지(없으면 경고 후 전체 유지)
#   full: True면 전량 사용(번호판), False면 리허설 서브셋(클래스당 REHEARSAL_PER_CLASS)
SOURCES = [
    {"key": "handle",    "gid": 0, "zip": "car-door-handle.v4i.yolov8.zip",          "keep": "all",           "full": False},
    {"key": "mirror",    "gid": 1, "zip": "Side Mirror.v2-first-version.yolov8.zip", "keep": "name:perfect",  "full": False},
    {"key": "plate_old", "gid": 2, "zip": "license-plate-label.v5i.yolov8.zip",      "keep": "all",           "full": True},
    {"key": "plate_new", "gid": 2, "zip": "license plate.v1i.yolov8.zip",            "keep": "all",           "full": True},
    {"key": "logo",      "gid": 3, "zip": "KIA Logo.v2i.yolov8.zip",                 "keep": "all",           "full": False},
    {"key": "fuel_cap",  "gid": 4, "zip": "Fuel Cap.v15i.yolov8.zip",                "keep": "name:Fuel-Cap", "full": False},
]

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ============================================================
# 헬퍼
# ============================================================
def load_names(src_dir: Path):
    cfg = yaml.safe_load(open(src_dir / "data.yaml"))
    return cfg["names"]


def collect_pairs(src_dir: Path):
    """원본의 train/valid/test를 훑어 (img, label) 쌍 수집. 라벨 있는 것만."""
    pairs = []
    for split in ["train", "valid", "test"]:
        img_dir = src_dir / split / "images"
        lbl_dir = src_dir / split / "labels"
        if not img_dir.exists():
            continue
        for img in img_dir.iterdir():
            if img.suffix.lower() not in IMG_EXT:
                continue
            lbl = lbl_dir / (img.stem + ".txt")
            if lbl.exists():
                pairs.append((img, lbl))
    return pairs


def clean_and_remap(src):
    """keep 정책 적용 + 전역 gid로 리맵(라벨 파일 in-place 재작성). 유지된 (img,lbl) 반환."""
    src_dir = WORK / src["key"]
    names = load_names(src_dir)
    keep = src["keep"]
    gid = src["gid"]

    if keep == "all":
        keep_ids = set(range(len(names)))
    elif keep.startswith("name:"):
        cname = keep.split("name:", 1)[1]
        if cname in names:
            keep_ids = {names.index(cname)}
        else:
            print(f"  ⚠️ [{src['key']}] '{cname}' 없음 -> 전체 클래스 유지. 원본 names={names}")
            keep_ids = set(range(len(names)))
    else:
        raise ValueError(keep)

    print(f"  [{src['key']}] names={names} | 유지 {sorted(keep_ids)} -> 전역 {gid}({GLOBAL_NAMES[gid]})")

    kept = []
    for img, lbl in collect_pairs(src_dir):
        out = []
        for line in open(lbl):
            p = line.split()
            if p and int(p[0]) in keep_ids:
                p[0] = str(gid)
                out.append(" ".join(p))
        if out:
            open(lbl, "w").write("\n".join(out) + "\n")
            kept.append((img, lbl))
    return kept


def copy_pairs(pairs, split, key):
    for img, lbl in pairs:
        stem = f"{key}__{img.stem}"
        shutil.copy(img, MERGED / split / "images" / (stem + img.suffix))
        shutil.copy(lbl, MERGED / split / "labels" / (stem + ".txt"))


# ============================================================
# 1. 압축 해제
# ============================================================
def main():
    # 준비물 점검
    if not BEST.exists():
        print(f"❌ best.pt 없음: {BEST}")
        sys.exit(1)
    for s in SOURCES:
        if not (BASE / s["zip"]).exists():
            print(f"❌ zip 없음: {BASE / s['zip']}")
            sys.exit(1)

    print(f"MPS 사용가능: {torch.backends.mps.is_available()}")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    if device == "cpu":
        print("⚠️ MPS를 못 써서 CPU로 진행합니다(매우 느림). Apple Silicon + 최신 torch인지 확인하세요.")

    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)
    print("\n[1] 압축 해제")
    for s in SOURCES:
        with zipfile.ZipFile(BASE / s["zip"]) as z:
            z.extractall(WORK / s["key"])
        print(f"  {s['key']} <- {s['zip']}")

    # ============================================================
    # 2. 정리 + 리맵
    # ============================================================
    print("\n[2] 라벨 정리 + 전역 클래스 리맵")
    per_source = {}
    for s in SOURCES:
        per_source[s["key"]] = clean_and_remap(s)
    print("  유지 이미지 수:", {k: len(v) for k, v in per_source.items()})

    # ============================================================
    # 3. 병합 데이터셋 구성 (번호판=전량, 4종=리허설 서브셋)
    # ============================================================
    print("\n[3] 병합 (plate=전량 / 4종=클래스당 최대 %d)" % REHEARSAL_PER_CLASS)
    if MERGED.exists():
        shutil.rmtree(MERGED)
    for split in ["train", "valid"]:
        (MERGED / split / "images").mkdir(parents=True, exist_ok=True)
        (MERGED / split / "labels").mkdir(parents=True, exist_ok=True)

    rng = random.Random(SEED)
    summary = {}
    for s in SOURCES:
        pairs = per_source[s["key"]][:]
        rng.shuffle(pairs)
        if not s["full"]:
            pairs = pairs[:REHEARSAL_PER_CLASS]      # 리허설: 일부만
        n_val = max(1, int(len(pairs) * VAL_RATIO)) if pairs else 0
        copy_pairs(pairs[:n_val], "valid", s["key"])
        copy_pairs(pairs[n_val:], "train", s["key"])
        summary[s["key"]] = (len(pairs) - n_val, n_val)

    for k, (tr, va) in summary.items():
        print(f"  {k:10s} train={tr:5d}  valid={va:4d}")
    tot_tr = len(list((MERGED / "train" / "images").iterdir()))
    tot_va = len(list((MERGED / "valid" / "images").iterdir()))
    print(f"  합계 train={tot_tr}  valid={tot_va}")

    # ============================================================
    # 4. data.yaml
    # ============================================================
    cfg = {
        "train": str(MERGED / "train" / "images"),
        "val": str(MERGED / "valid" / "images"),
        "nc": len(GLOBAL_NAMES),
        "names": GLOBAL_NAMES,
    }
    yaml.safe_dump(cfg, open(MERGED / "data.yaml", "w"), allow_unicode=True)

    # ============================================================
    # 5. fine-tuning (best.pt에서 출발)
    # ============================================================
    print(f"\n[5] fine-tuning 시작  epochs={EPOCHS}  device={device}")
    print("    (첫 실행이면 'python3 finetune_local.py 1' 로 1에포크 시간부터 재보세요)")
    model = YOLO(str(BEST))            # ★ 기존 통합 모델을 출발점으로
    model.train(
        data=str(MERGED / "data.yaml"),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=device,
        patience=10,
        seed=SEED,
        project=str(BASE / "ft_runs"),
        name="carparts_ft",
        exist_ok=True,
    )
    print("\n완료. 결과 가중치:")
    print(f"  {BASE / 'ft_runs' / 'carparts_ft' / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()