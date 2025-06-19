from ultralytics import YOLO

if __name__ == '__main__':
    # Load a pretrained YOLOv8 model
    model = YOLO('yolo11n.pt')

    # Train the model on a custom dataset
    model.train(data='data/data.yaml', epochs=400, imgsz=640, batch=64, workers=8)

    # Save the trained model