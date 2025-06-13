from ultralytics import YOLO
import matplotlib.pyplot as plt

def evaluate_model(model_path, data_yaml_path):
    model = YOLO(model_path)
    metrics = model.val(data=data_yaml_path)

    print("\nEvaluation Metrics:")
    print(f"Precision: {metrics.box.p}")
    print(f"Recall: {metrics.box.r}")
    print(f"mAP50: {metrics.box.map50}")
    print(f"mAP50-95: {metrics.box.map}")

    # Example plot
    plt.bar(['Precision', 'Recall', 'mAP50', 'mAP50-95'],
            [metrics.box.p.mean(), metrics.box.r.mean(), metrics.box.map50, metrics.box.map])
    plt.title('Evaluation Metrics')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.show()
