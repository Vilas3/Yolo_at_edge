yolo export model=yolov8s.pt format=engine device=0 int8=true
yolo detect predict model=yolov8s.engine source='0' show=True 
