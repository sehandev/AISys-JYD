import argparse
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils_yolo.datasets import letterbox
import numpy as np
from utils_yolo.general import check_img_size, non_max_suppression, scale_coords, set_logging
from utils_yolo.plots import plot_one_box
from utils_yolo.torch_utils import select_device, TracedModel

import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import Int16MultiArray, Float32MultiArray


class Yolov7:
    def __init__(self):
        # for hsr topic
        self.rgb_img = None
        self.pc = None
        self.object_len = 0
        self.yolo_bbox = []
        self.bridge = CvBridge()

        rgb_topic = '/camera/color/image_raw'
        self._rgb_sub = rospy.Subscriber(rgb_topic, Image, self._rgb_callback)
        self.yolo_pub = rospy.Publisher('/snu/yolo', Int16MultiArray, queue_size=10)

        self.yolo_img_pub = rospy.Publisher('/snu/yolo_img', Image, queue_size=10)
        self.ready()

    def _rgb_callback(self, img_msg):
        self.rgb_img = self.bridge.imgmsg_to_cv2(img_msg, 'bgr8')


    def ready(self):
        weights, imgsz = opt.weights, opt.img_size
        # Initialize
        set_logging()
        device = select_device(opt.device)
        print('device', device)
        half = device.type != 'cpu'  # half precision only supported on CUDA
        self.half = half
        print('half', half)
        # Load model
        print(weights)
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check img_size


        # model = TracedModel(model, device, opt.img_size)

        if half:
            model.half()  # to FP16


        cudnn.benchmark = True  # set True to speed up constant image size inference

        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        print('object list names', names)
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

        self.device, self.model, self.colors, self.names = device, model, colors, names
        print('self.model', self.model)

    def detect(self):
        if self.rgb_img is None:
            return None, None, None
        img, im0 = self.preprocess_img(self.rgb_img)
        device, model, colors, names = self.device, self.model, self.colors, self.names

        img = torch.from_numpy(img).to(device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        x = img
        y = []  # outputs
        # Inference
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            """[Robot side]"""
            for m in model.model[:1]:
                # print(m)
                if m.f != -1:  # if not from previous layer
                    x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
                x = m(x)  # run
                y.append(x if m.i in model.save else None)  # save output

            """[Server side]"""
            y = [x]  # outputs
            # y.append(x if m.i in model.save else None)
            print('x',x.shape)

            # print('y',y[0].shape)
            for idx, m in enumerate(model.model[1:]):
                if m.f != -1:  # if not from previous layer
                    x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
                if isinstance(x, list):
                    x = [t.to(device) for t in x]
                else:
                    x = x.to(device)
                x = m(x)  # run
                y.append(x if m.i in model.save else None)  # save output

        # Apply NMS
        pred = x[0]
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)[0]
        bbox_list = []
        bbox_with_conf_list = []

        # Process detections
        if len(pred):
            # Rescale boxes from img_size to im0 size
            pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], im0.shape).round()
            # Write results
            for *xyxy, conf, cls in reversed(pred):
                label = f'{names[int(cls)]} {conf:.2f}'
                c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                cent_x, cent_y = (c2[0] + c1[0]) // 2, (c2[1] + c1[1]) // 2
                width = c2[0] - c1[0]
                height = c2[1] - c1[1]

                plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
                bbox_list.append([cent_x, cent_y, width, height, int(cls)])
                bbox_with_conf_list.append([cent_x, cent_y, width, height, int(cls), int(conf * 100)]) #conf: bbox score
        # show result
        view_img = True
        if view_img:
            cv2.imshow('hsr_vision', im0)
            cv2.waitKey(1)  # 1 millisecond
        return bbox_list, bbox_with_conf_list, im0

    def preprocess_img(self, img):
        img0 = img.copy()
        img = letterbox(img, 640, stride=32)[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        return img, img0

    def yolo_publish(self, bbox_list):
        coord_list_msg = Int16MultiArray()
        coord_list_msg.data = [i for coord in bbox_list for i in coord]
        self.yolo_pub.publish(coord_list_msg)


    def yolo_img_publish(self, img):
        yolo_img_msg = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
        self.yolo_img_pub.publish(yolo_img_msg)

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='weights/yolov7-tiny.pt', help='model.pt path(s)')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    opt = get_opt()
    print(opt)
    rospy.init_node('aisys_robot', anonymous=True)
    yolov7_controller = Yolov7()
    image_resolution = (480, 640, 3)
    r = rospy.Rate(40)
    with torch.no_grad():
        while not rospy.is_shutdown():
            bbox_list, bag_bbox_list, img = yolov7_controller.detect() # bag bbox list is the version added confidence scores
            if bbox_list is None:
                continue
            yolov7_controller.yolo_publish(bbox_list)
            # yolov7_controller.yolo_with_conf_publish(bag_bbox_list)
            yolov7_controller.yolo_img_publish(img)
            r.sleep()

