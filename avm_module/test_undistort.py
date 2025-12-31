import cv2
import numpy as np
import os

def run_avm_test():
    # 1. 파일 경로 설정 (data 폴더 내 calib_ 파일들)
    left_config = 'data/calib_left.npz'
    rear_config = 'data/calib_rear.npz'
    
    # 보정 파라미터 로드
    try:
        left_data = np.load(left_config)
        rear_data = np.load(rear_config)
        
        # 기본 매트릭스와 블랙홀 제거용 new_mtx 로드
        l_mtx, l_dist = left_data['mtx'], left_data['dist']
        l_new_mtx = left_data['new_mtx']
        
        r_mtx, r_dist = rear_data['mtx'], rear_data['dist']
        r_new_mtx = rear_data['new_mtx']
        
        print("보정 데이터를 'data/' 폴더에서 성공적으로 불러왔습니다.")
    except Exception as e:
        print(f"파일을 불러오는데 실패했습니다 (경로확인!): {e}")
        return

    # 2. 카메라 연결 (0: 후면, 1: 좌측)
    cap_rear = cv2.VideoCapture(0)
    cap_left = cv2.VideoCapture(1)

    while True:
        ret_r, frame_r = cap_rear.read()
        ret_l, frame_l = cap_left.read()

        if not ret_r or not ret_l:
            print("카메라 영상을 가져올 수 없습니다.")
            break

        # 3. 왜곡 보정 적용 (이미 계산된 new_mtx 사용)
        # 루프 안에서 getOptimalNewCameraMatrix를 매번 호출하지 않아 더 빠릅니다.
        undistorted_l = cv2.undistort(frame_l, l_mtx, l_dist, None, l_new_mtx)
        undistorted_r = cv2.undistort(frame_r, r_mtx, r_dist, None, r_new_mtx)

        # 4. 시각화 (원본과 보정본 비교)
        res_l = np.hstack((cv2.resize(frame_l, (400, 300)), cv2.resize(undistorted_l, (400, 300))))
        res_r = np.hstack((cv2.resize(frame_r, (400, 300)), cv2.resize(undistorted_r, (400, 300))))
        
        total_view = np.vstack((res_l, res_r))

        cv2.putText(total_view, "LEFT (Top) / REAR (Bottom) - LEFT: Original | RIGHT: Undistorted", 
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('AVM Undistort Test', total_view)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap_left.release()
    cap_rear.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 인자 없이 호출해도 알아서 data/ 폴더 안의 파일을 찾습니다.
    run_avm_test()