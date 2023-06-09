# AISys-JYD
AI System 정영돈

- RGB
  - [ ]  ROS Client 세팅하기
  - [ ]  ROS Server 세팅하기
  - [ ]  Object Detection Model 선정하기
  - [ ]  Accuracy 측정하기
  - [ ]  FPS 측정하기
  - [ ]  Ubuntu에서 네트워크 속도를 제한하기
- Frame Interpolation
  - [ ]  Model 선정하기
  - [ ]  Accuracy 측정하기
- Object Detection Feature
  - [ ]  FPS 측정하기
- Frame Interpolation Feature
  - [ ]  Accuracy 측정하기
  - [ ]  FPS 측정하기
- 추가실험
  - [ ]  (JPEG, PNG 등 이미지 압축의 최적화를 고려하여) 단색 바탕으로 단순한 이미지와 복잡한 이미지 구분해서 파일용량과 속도를 측정하기

test env
- ubuntu 20.04, python3.8, 
- move weight file in weights/yolov7-tiny.pt
- link : [`yolov7-tiny.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt)

roslaunch
``` shell
roslaunch realsense2_camera rs_camera.launch
```

ip editing
``` shell
sudo vim /etc/hosts
```