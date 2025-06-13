from train import train_model
from evaluation import evaluate_model

DATA_YAML = 'dataset/augmented/data.yaml'
MODEL_SAVE_PATH = 'weights_new/best.pt'

if __name__ == '__main__':
    print("Training model...")
    model = train_model(DATA_YAML, MODEL_SAVE_PATH)

    print("\nEvaluating model...")
    evaluate_model(MODEL_SAVE_PATH, DATA_YAML)
