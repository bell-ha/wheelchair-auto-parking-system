import cv2
import numpy as np
import os

# ===============================
# CANVAS 설정 (출력 크기)
# ===============================
W, H = 600, 720

# ===============================
# 경로 설정
# ===============================
LEFT_IMG_PATH = "data/left_result.jpg"
REAR_IMG_PATH = "data/rear_result.jpg"
OUT_PATH = "data/avm_final_smooth.jpg"

# ===============================
# 마스크 영역 정의 (Polygon)
# ===============================
LEFT_POLY = np.array([
    (0, 270), (200, 270), (200, 540), (0, 630), (0, 540), (0, 360)
], dtype=np.int32)

REAR_POLY = np.array([
    (200, 540), (400, 540), (600, 630), (600, 720), (0, 720), (0, 630)
], dtype=np.int32)

# ===============================
# 1. 색상 매칭 함수 (Color Transfer)
# ===============================
def match_color(source, target):
    """
    source 이미지의 색상 통계(평균, 표준편차)를 target에 맞춤.
    이 과정을 통해 두 카메라의 화이트밸런스와 밝기 차이를 줄임.
    """
    # BGR -> LAB 변환 (밝기와 색상 정보 분리)
    src_lab = cv2.cvtColor(source.astype(np.uint8), cv2.COLOR_BGR2LAB)
    tgt_lab = cv2.cvtColor(target.astype(np.uint8), cv2.COLOR_BGR2LAB)

    # 채널별 통계치 계산
    s_mean, s_std = cv2.meanStdDev(src_lab)
    t_mean, t_std = cv2.meanStdDev(tgt_lab)

    # 색상 전이 공식 적용
    result = src_lab.astype(np.float32)
    for i in range(3):
        result[:, :, i] = ((result[:, :, i] - s_mean[i]) * (t_std[i] / (s_std[i] + 1e-5))) + t_mean[i]

    # 범위 제한 및 BGR 복원
    result = np.clip(result, 0, 255).astype(np.uint8)
    return cv2.cvtColor(result, cv2.COLOR_LAB2BGR).astype(np.float32) / 255.0

# ===============================
# 2. 부드러운 마스크 생성 (Soft Mask)
# ===============================
def make_soft_mask(polygon, blur_k=51):
    """경계선이 흐릿한 마스크를 생성하여 합성을 자연스럽게 함"""
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)
    
    # 가우시안 블러로 경계선 스무딩
    mask = cv2.GaussianBlur(mask, (blur_k, blur_k), 0)
    return (mask.astype(np.float32) / 255.0)[..., None]

# ===============================
# 3. 이미지 로드 유틸
# ===============================
def load_img(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")
    img = cv2.imread(path)
    if img is None:
        raise RuntimeError(f"이미지 로드 실패: {path}")
    return cv2.resize(img, (W, H))

# ===============================
# MAIN 로직
# ===============================
def main():
    # 이미지 로드 (Raw BGR)
    left_raw = load_img(LEFT_IMG_PATH)
    rear_raw = load_img(REAR_IMG_PATH)

    print("🎨 색상 보정 및 합성 중...")

    # [핵심] 색상 보정: Left를 Rear의 톤에 맞춤
    left_corrected = match_color(left_raw, rear_raw)
    rear_normalized = rear_raw.astype(np.float32) / 255.0

    # 부드러운 마스크 생성 (blur_k가 클수록 더 넓게 섞임)
    left_mask = make_soft_mask(LEFT_POLY, blur_k=41)
    rear_mask = make_soft_mask(REAR_POLY, blur_k=41)

    # 알파 블렌딩 합성 (중첩 영역 처리)
    # 겹치는 부분에서 밝기가 튀지 않도록 합산 마스크로 나누어 정규화
    total_mask = left_mask + rear_mask + 1e-8
    canvas = (left_corrected * left_mask + rear_normalized * rear_mask) / total_mask

    # 결과물 변환
    result = (np.clip(canvas, 0, 1) * 255).astype(np.uint8)

    # 화면 표시 및 저장
    cv2.imshow("AVM FINAL (Smooth & Corrected)", result)
    cv2.imwrite(OUT_PATH, result)
    
    print(f"✅ 결과가 저장되었습니다: {OUT_PATH}")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()