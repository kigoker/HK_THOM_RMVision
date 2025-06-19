from ultralytics import YOLO
import cv2
import numpy as np


def main():
    # 初始化摄像头
    capture = cv2.VideoCapture(0)
    if capture.isOpened() is False:
        print("Error opening the camera")
        return

    # 加载YOLO模型
    model = YOLO("yolov11_project_已完成/best.onnx")

    while capture.isOpened():
        ret, frame = capture.read()

        if ret is True:
            # 使用YOLO模型进行检测
            results = model(frame)
            
            # 在图像上绘制检测结果
            for result in results:
                boxes = result.boxes.cpu().numpy()
                for box in boxes:
                    # 获取边界框坐标
                    x1, y1, x2, y2 = box.xyxy[0].astype(int)
                    # 获取置信度
                    conf = box.conf[0]
                    # 获取类别
                    cls = int(box.cls[0])
                    # 获取类别名称
                    class_name = result.names[cls]
                    
                    # 绘制边界框
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # 添加类别标签和置信度
                    label = f'{class_name} {conf:.2f}'
                    cv2.putText(frame, label, (x1, y1 - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # 显示处理后的图像
            cv2.imshow('YOLO Detection', frame)
            
            # 按'q'键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # 释放资源
    capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()