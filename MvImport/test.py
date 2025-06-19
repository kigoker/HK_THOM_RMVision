import onnxruntime
import numpy as np
import cv2

def preprocess_image(image_path, input_size=(640, 640)):
    # 读取图片
    img = cv2.imread(image_path)
    # 调整图片大小
    img = cv2.resize(img, input_size)
    # 转换为RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 归一化
    img = img.astype(np.float32) / 255.0
    # 调整维度 [H,W,C] -> [1,C,H,W]
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, 0)
    return img

if __name__ == "__main__":
    # 1. 加载ONNX模型
    model_path = "best.onnx"  # 替换为您的onnx模型路径
    session = onnxruntime.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    
    # 2. 准备输入图片
    image_path = "/home/thom/文档/yolov11/yolov11_project/data/test/images/84_jpg.rf.e198bbc5eb377ccb34c85ad315fd3c42.jpg"
    input_image = preprocess_image(image_path)
    
    # 3. 获取模型输入名称
    input_name = session.get_inputs()[0].name
    
    # 4. 运行推理
    results = session.run(None, {input_name: input_image})
    
    # 5. 处理输出结果
    # ONNX模型的输出格式可能与原始YOLO模型不同
    # 通常输出是 [batch, num_boxes, 85] 的格式
    # 其中85 = 4(坐标) + 1(置信度) + 80(类别概率)
    output = results[0]
    
    # 打印检测结果
    for detection in output[0]:  # 第一张图片的所有检测框
        confidence = detection[4]  # 置信度
        if confidence > 0.5:  # 设置置信度阈值
            x1, y1, x2, y2 = detection[0:4]
            print(f"检测框坐标: ({x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f})")
            print(f"置信度: {confidence:.2f}")
            print("------------------------")
