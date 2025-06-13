import cv2
from ultralytics import YOLO

# Load your custom YOLOv8 model
model = YOLO("weights/best.pt")  # Path to your trained model


# Confidence threshold
CONFIDENCE_THRESHOLD = 0.25

# Real-time detection function
def detect_and_display(frame):
    results = model(frame, verbose=False)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            if confidence >= CONFIDENCE_THRESHOLD:
                label = f"{model.names[class_id]} {confidence:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    return frame

# Main loop
def main():
    cap = cv2.VideoCapture(0)  # Use webcam (0 is default camera)

    if not cap.isOpened():
        print("Failed to open webcam.")
        return

    print("Starting real-time detection... Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame.")
            break

        frame = detect_and_display(frame)
        cv2.imshow("YOLOv8 Real-Time Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
