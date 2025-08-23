import torch

from . import config
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score

def evaluate_model(model, validation_loader):
    """
    Evaluates a trained model on the validation dataset and prints key metrics.
    """
    print("\nEvaluating Model Performance on Validation Set...")
    device = config.DEVICE
    model.to(device)
    model.eval()

    all_true_labels = []
    all_predicted_labels = []

    with torch.no_grad():
        for images, labels in tqdm(validation_loader, desc="Evaluating"):
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            all_true_labels.extend(labels.cpu().numpy())
            all_predicted_labels.extend(predicted.cpu().numpy())

    # Calculate Metrics using scikit-learn
    accuracy = accuracy_score(all_true_labels, all_predicted_labels)
    macro_f1 = f1_score(all_true_labels, all_predicted_labels, average='macro', zero_division=0)
    weighted_f1 = f1_score(all_true_labels, all_predicted_labels, average='weighted', zero_division=0)

    print("\nValidation Metrics:")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Macro F1-Score: {macro_f1:.4f}")
    print(f"Weighted F1-Score: {weighted_f1:.4f}")

   
    metrics = {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1
    }
    return metrics