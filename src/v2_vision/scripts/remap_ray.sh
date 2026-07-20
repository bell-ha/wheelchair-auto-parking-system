#!/usr/bin/env bash
#
# 레이 라벨 데이터셋의 클래스 인덱스를 전역(통합 모델) 기준으로 재매핑.
#
# 전역 클래스: 0:handle  1:mirror  2:plate  3:logo  4:fuel_cap
#
# 이 레이 데이터셋(원본):  0:car emblem  1:license plate  2:side mirror
# 재매핑:  0->3(logo)   1->2(plate)   2->1(mirror)
#
# 사용법:
#   cd ~/Desktop/휠체어프로젝트/my_data/labeled
#   bash remap_ray.sh ray_data      # ray_data = 압축 푼 폴더 이름

set -eu

DIR="${1:-ray_data}"
if [ ! -d "$DIR" ]; then
  echo "❌ 폴더 없음: $DIR  (압축 푼 폴더 이름을 인자로 주세요)"
  exit 1
fi

echo "재매핑 대상: $DIR"
echo "  0(car emblem)->3(logo) | 1(license plate)->2(plate) | 2(side mirror)->1(mirror)"
echo ""

# 원본 라벨 백업 (한 번만)
if [ ! -d "${DIR}_backup" ]; then
  cp -r "$DIR" "${DIR}_backup"
  echo "원본 백업 생성: ${DIR}_backup"
fi

# 모든 split의 라벨 txt를 재매핑
# 주의: 임시값(+10)을 거쳐 충돌 방지 (1->2, 2->1 이 서로 덮어쓰지 않도록)
count=0
find "$DIR" -path '*/labels/*.txt' | while read -r f; do
  awk '{
    c=$1
    if (c==0) $1=3        # car emblem -> logo
    else if (c==1) $1=2   # license plate -> plate
    else if (c==2) $1=1   # side mirror -> mirror
    print
  }' "$f" > "$f.tmp" && mv "$f.tmp" "$f"
done

# 재매핑된 라벨 통계
echo "재매핑 후 클래스 분포:"
for gid in 0 1 2 3 4; do
  n=$(find "$DIR" -path '*/labels/*.txt' -exec cat {} + 2>/dev/null | awk -v g="$gid" '$1==g' | wc -l | tr -d ' ')
  names=(handle mirror plate logo fuel_cap)
  echo "  $gid ${names[$gid]}: $n 박스"
done
echo ""

# data.yaml을 5클래스 전역 기준으로 재작성
cat > "$DIR/data.yaml" << 'EOF'
train: ../train/images
val: ../valid/images
test: ../test/images

nc: 5
names: ['handle', 'mirror', 'plate', 'logo', 'fuel_cap']
EOF
echo "data.yaml 재작성 완료 (nc=5, 전역 이름)"
echo ""
echo "✅ 완료. 이제 이 폴더를 Kaggle fine-tune의 새 소스로 추가하면 됩니다."
cat "$DIR/data.yaml"