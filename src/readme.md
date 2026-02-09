1. 가상환경 활성화 
source venv/bin/activate

2. 가상환경에 requirments.txt설치
pip install -r requirements.txt

### etc폴더: 아르코마커 찾기, 카메라 찾기, 녹화 등
### 01_calibration: 어라운드 뷰 만드는 폴더
### wheelchairdetect: 휠체어 찾아서 캔버스에 그리는 기능

### 01_calibration : 카메라 캘리브레이션
### 02_aroundview : 어라운드 뷰 만드는 폴더
### 03_localization : 맵을 만드는 폴더
### 04_planning: 경로계획 폴더
### 05_hardware: 하드웨어 요구 폴더
### 06_angle_tunning : 각도 보정 폴더
### 07_map_calibration : 최종 맵 캘리브레이션(잘 작동되면, 03번 06번 삭제 가능)