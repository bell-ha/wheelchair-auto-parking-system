import cv2
import numpy as np
import glob
import os

# 1. 설정 (사용자님 격자 크기에 맞게 수정: 내부 교차점 개수)
CHECKERBOARD = (6, 6) 
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

def auto_calibrate(cam_name):
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

    objpoints = [] 
    imgpoints = [] 

    images = glob.glob(f'data/{cam_name}/*.jpg')
    if not images:
        print(f"[{cam_name}] 사진이 없습니다.")
        return

    valid_count = 0
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 코너 찾기 (성공한 것만 자동으로 추림)
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            valid_count += 1
            print(f"[{cam_name}] {os.path.basename(fname)}: 인식 성공")
        else:
            print(f"[{cam_name}] {os.path.basename(fname)}: 인식 실패 (자동 제외)")

    if valid_count < 5:
        print(f"!! 경고 !! {cam_name} 인식된 사진이 너무 적습니다. (최소 10장 권장)")
        return

    # 2. 카메라 보정 계산
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # 3. 블랙홀 자동 제거 (Alpha=0 설정)
    # alpha=0은 왜곡이 펴진 후 '검은 여백(블랙홀)'이 생기지 않도록 깔끔한 안쪽만 잘라냅니다.
    h, w = gray.shape[:2]
    new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))

    # 데이터 저장
    np.savez(f'data/calib_{cam_name}.npz', mtx=mtx, dist=dist, new_mtx=new_mtx, roi=roi)
    print(f"== {cam_name} 보정 완료 (사용된 사진: {valid_count}장) ==")

# 실행
auto_calibrate('left')
auto_calibrate('rear')