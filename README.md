# Yolo_at_edge
Test codes for yolo on Jetson and Raspberry boards

This codes are used for test and benchmark the performance of both boards on a filed apliccation of Yolo Models
## Jetson Orin Nano
The board used is the Jetson Orin Nano developer kit (8GB) with the new JetPack 6.2, so it is, in NVIDIA terms, the Super Jetson Orin Nano.
To enable all its power run the set_max_power.sh script.

### Normal Model
The Jetson folder has a yolo_*_usb.py for each yolov8 model size (n,s,m,l,x). To run it just use python3 yolo_*_usb.py command. This code uses a usb camera attached to your board located at the /dev/video0. If your camera is in other device you need to change it in the code, more specifically in the cap =
cv2.VideoCapture (0). This files are used for calculating the FPS your images are being processed, if you want just to run your models normally, run yolo detect predict model=yolov8s.pt source='0' show=True in your terminal.

### TensorRT int8 Model
Beyond that it contains a yolo_s_trt.py file that utilizes the converted TensorRT .engine model and shows the FPS on screen.
If you want to run it only with bash commands you can run the tensorrt_int8.sh, witch converts the .pt model to .engine model and the runs it. Note that if you already have the converted .engine model in your directory you can just run yolo detect predict model=yolov8s.engine source='0' show=True in your terminal.


## Raspberry Pi 5 AI Kit
The Rasp 5 AI Kit includes the Halio8L TPU, witch makes possible for running Yolo inference at lower prices.
