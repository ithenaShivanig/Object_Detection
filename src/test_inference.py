from ultralytics import YOLO

# Load your model
model = YOLO("weights_new/best.pt")  # Path to your custom trained model

# Run inference on saved webcam image
results = model("C:/Users/shivanig_ithena/object detection/dataset/augmented/test_aug/images/House_24_jpeg.rf.8e2aca4989dacbd962170800d6c1188f.jpg", show=True)

# Optional: print details
for r in results:
    for box in r.boxes:
        print(f"Class ID: {int(box.cls[0])}, Confidence: {float(box.conf[0]):.2f}")
