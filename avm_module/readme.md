1. 가상환경 활성화 
source venv/bin/activate

2. 가상환경에 requirments.txt설치
pip install -r requirements.txt

3. 카메라 찾기(테스트 상으로 0번: 후면, 1번: 좌측)
find_my_cams.py

4. 카메라 촬영(find_my_cams.py을 통한 번호로 설정), data/calib_images에 저장됨
get_calibration_data.py 

5. 캘리브레이션 코드(data/calib_images를 불러오고, calib_result_common.npz로 저장됨)
calibrate.py

6. 왜곡 펼치기(4,5,6번 과정 반복해서 괜찮은 결과물 나올 수 있도록)
test_undistort.py

7. 어라운드 뷰 만들기