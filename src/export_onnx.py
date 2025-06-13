from ultralytics import YOLO

# Load trained model and export to ONNX
model = YOLO("weights/best.pt")
model.export(format="openvino")