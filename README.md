# AISys-JYD
Team 정영돈

## Test env
- ubuntu 20.04, python3.8, ROS noetic
- camera : intel realsense-d435
- 같은 wifi에 두 pc 연결

## Setup

1. Install realsense-ros 
``` shell
sudo apt-get install ros-noetic-realsense2-camera
```
2. Install yolov7 weight
  - move weight file in weights/yolov7.pt
  - link : [`yolov7.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt)

3. ip setup
``` shell
# ref : https://velog.io/@boris3853/ROS-%ED%86%B5%EC%8B%A0-%EA%B8%B0%EB%B3%B8-%EC%98%88%EC%A0%9C
sudo vim /etc/hosts
```
- add ip address and domain name

# Running our code
1. robot side running
``` shell
# 1. turn on realsense
roslaunch realsense2_camera rs_camera.launch
# 2. run robotside yolov7
python robot_pubilsher_layer4.py
```

2. server side running
``` shell
python server_subscriber_layer4.py
```


## todo
- RGB
  - [ ]  Accuracy 측정하기
  - [ ]  Ubuntu에서 네트워크 속도를 제한하기
- Frame Interpolation
  - [ ]  Accuracy 측정하기
- Frame Interpolation Feature
  - [ ]  Accuracy 측정하기
  - [ ]  FPS 측정하기
- 추가실험
  - [ ]  (JPEG, PNG 등 이미지 압축의 최적화를 고려하여) 단색 바탕으로 단순한 이미지와 복잡한 이미지 구분해서 파일용량과 속도를 측정하기
