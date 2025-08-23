import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from . import config
from tqdm import tqdm
from .model import SimpleCNN
from torch.utils.data import DataLoader
from .datasets import (load_and_process_data, CharacterDataset, UnlabeledCharacterDataset,
                                             PreIndexedDataset, MeanTeacherUnlabeledDataset)


def train_baseline_model(in_model=None, dataset_labeled=None, val_data=None):
    """
    Train a baseline model using only the labeled dataset.
    """
    print("\nRunning: Baseline Model Training Pipeline...")
    device = config.DEVICE

    # Load and process all data
    if dataset_labeled==None and val_data==None:
        labeled_train, _, validation = load_and_process_data()
    else:
        labeled_train = dataset_labeled
        validation = val_data
        
    # Create character mapping and DataLoaders
    all_chars = set(label for _, label in labeled_train)
    all_chars.update(label for _, label in validation)
    digits = sorted([c for c in all_chars if c.isdigit()])
    symbols = sorted([c for c in all_chars if not c.isdigit()])
    char_to_idx = {digit: int(digit) for digit in digits}
    next_idx = 10
    for symbol in symbols:
        char_to_idx[symbol] = next_idx
        next_idx += 1
    num_classes = len(char_to_idx)
    
    train_dataset = CharacterDataset(labeled_train, char_to_idx)
    validation_dataset = CharacterDataset(validation, char_to_idx)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # Instantiate and train the model
    if in_model==None:
        model = SimpleCNN(num_classes=num_classes)
    else:
        model = in_model

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_weights = copy.deepcopy(model.state_dict())

    for epoch in range(config.EPOCHS):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        model.eval()
        correct, total, val_loss = 0, 0, 0.0
        with torch.no_grad():
            for images, labels in validation_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_val_loss = val_loss / len(validation_loader)
        val_accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{config.EPOCHS}] | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2f}%")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_model_weights = copy.deepcopy(model.state_dict())
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= config.PATIENCE:
            print(f"Early stopping triggered.")
            break
    
    print("Finished baseline training.")
    model.load_state_dict(best_model_weights)
    return model, char_to_idx, validation_loader


def pseudo_labeling_pipeline(base_model):
    """
    Uses a base model and runs pseudo-labeling workflow.
    """
    print("\nRunning: Pseudo-Labeling Pipeline...")
    device = config.DEVICE

    # Load and process all data
    labeled_train, unlabeled_train, validation = load_and_process_data()

    # Create character mapping and initial DataLoaders
    all_chars = set(label for _, label in labeled_train)
    all_chars.update(label for _, label in validation)
    digits = sorted([c for c in all_chars if c.isdigit()])
    symbols = sorted([c for c in all_chars if not c.isdigit()])
    char_to_idx = {digit: int(digit) for digit in digits}
    next_idx = 10
    for symbol in symbols:
        char_to_idx[symbol] = next_idx
        next_idx += 1
    num_classes = len(char_to_idx)
    
    # labeled_dataset = CharacterDataset(labeled_train, char_to_idx)
    validation_dataset = CharacterDataset(validation, char_to_idx)
    # labeled_loader = DataLoader(labeled_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # Initial baseline model
    baseline_model = base_model

    # Generate pseudo-labels
    print("\n--- Generating pseudo-labels for unlabeled data ---")
    unlabeled_dataset = UnlabeledCharacterDataset(unlabeled_train)
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    baseline_model.eval()
    all_outputs, original_indices = [], []
    with torch.no_grad():
        for images, indices in tqdm(unlabeled_loader, desc="Predicting"):
            images = images.to(device)
            outputs = baseline_model(images)
            all_outputs.append(outputs.cpu())
            original_indices.append(indices.cpu())
            
    all_outputs = torch.cat(all_outputs, dim=0)
    original_indices = torch.cat(original_indices, dim=0)
    all_probs = F.softmax(all_outputs, dim=1)

    # Filter for high-confidence pseudo-labels
    pseudo_labeled_data = []
    confidences, predicted_indices = torch.max(all_probs, dim=1)
    for i in range(len(all_probs)):
        if confidences[i].item() >= config.CONFIDENCE_THRESHOLD:
            image = unlabeled_train[original_indices[i].item()]
            pseudo_label_idx = predicted_indices[i].item()
            pseudo_labeled_data.append((image, pseudo_label_idx))
    print(f"Generated {len(pseudo_labeled_data)} new pseudo-labeled samples.")

    # Train the final model on the augmented dataset
    # original_labeled_data_with_indices = [(img, char_to_idx[lbl]) for img, lbl in labeled_train]
    # augmented_train_data = original_labeled_data_with_indices + pseudo_labeled_data
    
    # augmented_dataset = PreIndexedDataset(augmented_train_data)
    # augmented_loader = DataLoader(augmented_dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    # final_model = SimpleCNN(num_classes=num_classes)
    # final_model = train_baseline_model(final_model, augmented_loader, validation_loader)


    idx_to_char = {v: k for k, v in char_to_idx.items()}
    pseudo_labeled_with_chars = [(img, idx_to_char[idx]) for img, idx in pseudo_labeled_data]
    augmented_train_data = labeled_train + pseudo_labeled_with_chars

    final_model = SimpleCNN(num_classes=num_classes)
    final_model, _, _ = train_baseline_model(final_model, augmented_train_data, validation)

    return final_model, char_to_idx, validation_loader


def train_pi_model():
    """
    train the SimpleCNN using the Pi-Model semi-supervised approach.
    """
    print("\nRunning: Pi-Model Training Pipeline...")
    device = config.DEVICE
    
    # Load data
    labeled_train, unlabeled_train, validation = load_and_process_data()
    all_chars = set(label for _, label in labeled_train)
    all_chars.update(label for _, label in validation)
    digits = sorted([c for c in all_chars if c.isdigit()])
    symbols = sorted([c for c in all_chars if not c.isdigit()])
    char_to_idx = {digit: int(digit) for digit in digits}
    next_idx = 10
    for symbol in symbols:
        char_to_idx[symbol] = next_idx
        next_idx += 1
    num_classes = len(char_to_idx)
    
    labeled_dataset = CharacterDataset(labeled_train, char_to_idx)
    unlabeled_dataset = UnlabeledCharacterDataset(unlabeled_train)
    validation_dataset = CharacterDataset(validation, char_to_idx)

    labeled_loader = DataLoader(labeled_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # Instantiate model and train using Pi-Model logic
    model = SimpleCNN(num_classes=num_classes).to(device)
    supervised_criterion = nn.CrossEntropyLoss()
    consistency_criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    best_val_accuracy = 0.0
    epochs_no_improve = 0
    best_model_weights = copy.deepcopy(model.state_dict())

    for epoch in range(config.EPOCHS):
        model.train()
        labeled_iter = iter(labeled_loader)
        unlabeled_iter = iter(unlabeled_loader)
        progress_bar = tqdm(range(len(unlabeled_loader)), desc=f"Epoch {epoch+1}/{config.EPOCHS}")
        for i in progress_bar:
            try:
                labeled_images, labels = next(labeled_iter)
            except StopIteration:
                labeled_iter = iter(labeled_loader)
                labeled_images, labels = next(labeled_iter)
            unlabeled_images, _ = next(unlabeled_iter)
            labeled_images, labels = labeled_images.to(device), labels.to(device)
            unlabeled_images = unlabeled_images.to(device)
            labeled_outputs = model(labeled_images)
            supervised_loss = supervised_criterion(labeled_outputs, labels)
            unlabeled_outputs_1 = model(unlabeled_images)
            unlabeled_outputs_2 = model(unlabeled_images)
            consistency_loss = consistency_criterion(F.softmax(unlabeled_outputs_1, dim=1), F.softmax(unlabeled_outputs_2, dim=1))
            total_loss = supervised_loss + (config.LAMBDA_WEIGHT * consistency_loss)
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in validation_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{config.EPOCHS}] | Val Accuracy: {val_accuracy:.2f}%")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            epochs_no_improve = 0
            best_model_weights = copy.deepcopy(model.state_dict())
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= config.PATIENCE:
            print(f"Early stopping triggered.")
            break
            
    print("Finished Pi-Model training.")
    model.load_state_dict(best_model_weights)
    return model, char_to_idx, validation_loader


def train_mean_teacher_model():
    """
    Trains the SimpleCNN using the Mean Teacher semi-supervised approach.
    """
    print("\nRunning: Mean Teacher Training Pipeline...")
    device = config.DEVICE
    
    # Loading Data
    labeled_train, unlabeled_train, validation = load_and_process_data()
    all_chars = set(label for _, label in labeled_train)
    all_chars.update(label for _, label in validation)
    digits = sorted([c for c in all_chars if c.isdigit()])
    symbols = sorted([c for c in all_chars if not c.isdigit()])
    char_to_idx = {digit: int(digit) for digit in digits}
    next_idx = 10
    for symbol in symbols:
        char_to_idx[symbol] = next_idx
        next_idx += 1
    num_classes = len(char_to_idx)
    
    # Augmentation chain 
    # strong transforms for student version
    strong_transform = T.Compose([
        T.ToPILImage(),
        T.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        T.ToTensor()
    ])
    # weak transforms for teacher version 
    weak_transform = T.Compose([
        T.ToPILImage(),
        T.RandomAffine(degrees=5, translate=(0.05, 0.05)),
        T.ToTensor()
    ])
    # no transform for validation data
    val_transform = T.Compose([T.ToPILImage(), T.ToTensor()])

    # prepare datasets
    labeled_dataset = CharacterDataset(labeled_train, char_to_idx, transform=strong_transform)
    unlabeled_dataset = MeanTeacherUnlabeledDataset(unlabeled_train, weak_transform=weak_transform, strong_transform=strong_transform)
    validation_dataset = CharacterDataset(validation, char_to_idx, transform=val_transform)

    labeled_loader = DataLoader(labeled_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # instanciate models
    student_model = SimpleCNN(num_classes=num_classes).to(device)
    teacher_model = SimpleCNN(num_classes=num_classes).to(device)
    
    teacher_model.load_state_dict(student_model.state_dict())
    for param in teacher_model.parameters():
        param.requires_grad = False

    # Loss and Optimizer (only for student)
    supervised_criterion = nn.CrossEntropyLoss()
    consistency_criterion = nn.MSELoss()
    optimizer = optim.Adam(student_model.parameters(), lr=config.LEARNING_RATE)
    
    best_val_accuracy = 0.0
    epochs_no_improve = 0
    best_model_weights = copy.deepcopy(student_model.state_dict())

    # training loop
    for epoch in range(config.EPOCHS):
        student_model.train()
        teacher_model.eval() 
        
        labeled_iter = iter(labeled_loader)
        unlabeled_iter = iter(unlabeled_loader)
        
        num_iterations = max(len(labeled_loader), len(unlabeled_loader))
        progress_bar = tqdm(range(num_iterations), desc=f"Epoch {epoch+1}/{config.EPOCHS}")

        for _ in progress_bar:
            try:
                labeled_images, labels = next(labeled_iter)
            except StopIteration:
                labeled_iter = iter(labeled_loader)
                labeled_images, labels = next(labeled_iter)
            
            try:
                unlabeled_weak, unlabeled_strong, _ = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabeled_loader)
                unlabeled_weak, unlabeled_strong, _ = next(unlabeled_iter)

            labeled_images, labels = labeled_images.to(device), labels.to(device)
            unlabeled_weak, unlabeled_strong = unlabeled_weak.to(device), unlabeled_strong.to(device)

            # Supervised Loss
            labeled_outputs = student_model(labeled_images)
            supervised_loss = supervised_criterion(labeled_outputs, labels)

            # Consistency Loss
            student_outputs_unlabeled = student_model(unlabeled_strong)
            with torch.no_grad():
                teacher_outputs_unlabeled = teacher_model(unlabeled_weak)
            
            consistency_loss = consistency_criterion(
                F.softmax(student_outputs_unlabeled, dim=1),
                F.softmax(teacher_outputs_unlabeled, dim=1)
            )

            # final loss and backpropagation
            total_loss = supervised_loss + (config.LAMBDA_WEIGHT * consistency_loss)
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # update teacher weights
            with torch.no_grad():
                for teacher_param, student_param in zip(teacher_model.parameters(), student_model.parameters()):
                    teacher_param.data.mul_(config.EMA_DECAY).add_(student_param.data, alpha=1 - config.EMA_DECAY)

        # evaluate
        student_model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in validation_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = student_model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{config.EPOCHS}] | Val Accuracy: {val_accuracy:.2f}%")

        # Early Stopping
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            epochs_no_improve = 0
            best_model_weights = copy.deepcopy(student_model.state_dict())
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= config.PATIENCE:
            print(f"Early stopping triggered.")
            break
            
    print("Finished Mean Teacher training.")
    student_model.load_state_dict(best_model_weights)
    return student_model, char_to_idx, validation_loader
    