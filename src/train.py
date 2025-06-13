from ultralytics import YOLO

def train_model(data_yaml_path, model_output_path='weights_new/best.pt', epochs=30, batch=16):
    model = YOLO('yolov8n.pt')  # Or 'yolov8s.pt' depending on your needs

    model.train(
        data=data_yaml_path,
        epochs=epochs,
        batch=batch,
        imgsz=640,
        project='runs_new',
        name='custom_train',
        save=True
    )

    # Save model
    model_path = f'runs_new/detect/custom_train/weights_new/best.pt'
    model.save(model_output_path)

    return model
