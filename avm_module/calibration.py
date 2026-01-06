import cv2
import numpy as np
import os

def fine_tune_calibration():
    # 1. 기존 데이터 로드
    # [체크] 폴더가 없으면 에러가 날 수 있으므로 미리 확인
    if not os.path.exists('data'):
        os.makedirs('data')

    file_path = 'data/calib_result_common.npz'
    if not os.path.exists(file_path):
        print("❌ 기존 보정 파일이 없습니다. 기본값을 설정합니다.")
        # 파일이 없을 경우를 위한 기본값 설정
        K = np.array([[500, 0, 640], [0, 500, 360], [0, 0, 1]], dtype=np.float32)
        D = np.array([[0, 0, 0, 0]], dtype=np.float32)
    else:
        data = np.load(file_path)
        K = data['mtx'].astype(np.float32)
        D = data['dist'].astype(np.float32)
    
    initial_f = int(K[0, 0])
    initial_cx = int(K[0, 2])
    initial_cy = int(K[1, 2])
    initial_k1 = int(D[0, 0] * 100 + 500)

    cap = cv2.VideoCapture(0) # 맥북인 경우 0 또는 1 확인
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    win_name = 'Fine Tuner (Left: Original | Right: Fine-tuned with Grid)'
    cv2.namedWindow(win_name)

    # 슬라이더 생성
    cv2.createTrackbar('f_scale', win_name, initial_f, 2000, lambda x: None)
    cv2.createTrackbar('cx', win_name, initial_cx, 1280, lambda x: None)
    cv2.createTrackbar('cy', win_name, initial_cy, 720, lambda x: None)
    cv2.createTrackbar('k1', win_name, initial_k1, 1000, lambda x: None)
    cv2.createTrackbar('balance', win_name, 0, 100, lambda x: None)

    print("💡 가이드: 's'를 눌러 저장, 'q'를 눌러 종료")

    while True:
        ret, frame = cap.read()
        if not ret: break

        f = cv2.getTrackbarPos('f_scale', win_name)
        cx = cv2.getTrackbarPos('cx', win_name)
        cy = cv2.getTrackbarPos('cy', win_name)
        k1 = (cv2.getTrackbarPos('k1', win_name) - 500) / 100.0
        bal = cv2.getTrackbarPos('balance', win_name) / 100.0

        # 새로운 행렬 생성 (내부 값 수정)
        new_K_tuned = K.copy()
        new_K_tuned[0, 0], new_K_tuned[1, 1] = f, f
        new_K_tuned[0, 2], new_K_tuned[1, 2] = cx, cy
        
        new_D_tuned = D.copy()
        new_D_tuned[0, 0] = k1

        # [최적화] 보정 적용 - 1280x720 원본으로 계산 후 합칠 때만 resize
        # 어안렌즈 보정 핵심 함수
        new_mtx = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            new_K_tuned, new_D_tuned, (1280, 720), np.eye(3), balance=bal
        )
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            new_K_tuned, new_D_tuned, np.eye(3), new_mtx, (1280, 720), cv2.CV_16SC2
        )
        undistorted = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)

        # --- 격자 그리기 로직 (수정됨) ---
        grid_img = undistorted.copy()
        gh, gw = grid_img.shape[:2]
        
        # 가로 격자 및 세로 격자 간격 조정 (80px 간격)
        for x in range(0, gw, 80):
            cv2.line(grid_img, (x, 0), (x, gh), (0, 255, 0), 1)
        for y in range(0, gh, 80):
            cv2.line(grid_img, (0, y), (gw, y), (0, 255, 0), 1) # y축 선 수정
            
        cv2.line(grid_img, (gw//2, 0), (gw//2, gh), (0, 0, 255), 2) # 중앙 수직
        cv2.line(grid_img, (0, gh//2), (gw, gh//2), (0, 0, 255), 2) # 중앙 수평

        # 화면 결합 (H-Stack)
        res_orig = cv2.resize(frame, (640, 360))
        res_tuned = cv2.resize(grid_img, (640, 360))
        display = np.hstack((res_orig, res_tuned))

        cv2.putText(display, f"F:{f} K1:{k1:.2f} Bal:{bal:.2f}", (660, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        cv2.imshow(win_name, display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            # 저장할 때 핵심 파라미터 3개를 모두 저장해야 나중에 remap할 수 있음
            np.savez(file_path, mtx=new_K_tuned, dist=new_D_tuned, new_mtx=new_mtx)
            print(f"💾 데이터 저장 완료! (F:{f}, K1:{k1}, Bal:{bal})")
            break
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    fine_tune_calibration()