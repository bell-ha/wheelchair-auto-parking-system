1. 가상환경 활성화 
source venv/bin/activate

2. 가상환경에 requirments.txt설치
pip install -r requirements.txt

3. 카메라 찾기(테스트 상으로 0번: 후면, 1번: 좌측)
find_my_cams.py

4. 캘리브레이션(calib_result.npz로 저장됨, 체커보드로 캘리브레이션 진행 후 수동 보정)
calibrate.py


7. 어라운드 뷰 만들기
main.py #캔버스 구성
aroundview_left.py #왼쪽 어라운드뷰 만들기
aroundview_rear.py #뒷쪽 어라운드 뷰 만들기

- 영상 촬영
record.py

- 촬영된 영상에서 아르코마커 찾기
record_video_aruco_detection.py

- 일반 카메라에서 아르코마커 찾기(성능 제일 좋음)
aruco_detection.py