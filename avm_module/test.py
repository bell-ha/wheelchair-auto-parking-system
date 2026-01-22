import cv2
import numpy as np
import os

def apply_calibration_to_file():
    # 1. 경로 설정
    input_path = 'test/rear.jpg'
    calib_path = 'data/calib_rear.npz'
    output_path = 'test/rear_calibrated.jpg'

    # 2. 이미지 로드
    if not os.path.exists(input_path):
        print(f"❌ 파일을 찾을 수 없습니다: {input_path}")
        return
    
    img = cv2.imread(input_path)
    h, w = img.shape[:2]

    # 3. 보정 데이터(npz) 로드 및 맵 생성
    if not os.path.exists(calib_path):
        print(f"❌ 보정 파일이 없습니다: {calib_path}")
        return

    try:
        data = np.load(calib_path)
        # 사용자의 fisheye 보정 로직 그대로 적용
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            data['mtx'], 
            data['dist'], 
            np.eye(3), 
            data['new_mtx'], 
            (w, h), 
            cv2.CV_16SC2
        )
        print(f"✅ {calib_path} 보정 데이터 로드 완료.")

        # 4. 왜곡 보정(Remap) 수행
        calibrated_img = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)

        # 5. 결과 저장 및 시각화
        cv2.imwrite(output_path, calibrated_img)
        print(f"📸 보정 완료! 저장 경로: {output_path}")

        # 화면에 비교 출력 (결과 확인용)
        res_orig = cv2.resize(img, (640, 360))
        res_calib = cv2.resize(calibrated_img, (640, 360))
        comparison = np.hstack((res_orig, res_calib))
        
        cv2.imshow("Result (Left: RAW / Right: Calibrated)", comparison)
        print("⌨️ 아무 키나 누르면 종료됩니다.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"❌ 오류 발생: {e}")

if __name__ == "__main__":
    apply_calibration_to_file()