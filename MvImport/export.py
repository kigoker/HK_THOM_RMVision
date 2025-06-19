from ultralytics import YOLO

if __name__ == '__main__':
    # Load a pretrained model
    model = YOLO('best.pt')

    # Export the model to ONNX format
    model.export(format='onnx')

    # Export the model to TorchScript format