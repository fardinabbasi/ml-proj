import os 
import time 
import torch

from . import config
from tqdm import tqdm
from torch.utils.data import DataLoader
from .model import ResNet50FPN, CustomFasterRCNN
from torchmetrics.detection import MeanAveragePrecision
from .dataset import CharacterLocalizationDataset, get_transform, collate_fn

def run_training():
    """
    Main function for training and validation process.
    """

    config.RESULTS_DIR.mkdir(exist_ok=True)

    # Create Datasets
    print("Creating datasets...")
    train_dataset = CharacterLocalizationDataset(
        image_dir=config.DATA_DIR / 'train/images',
        label_dir=config.DATA_DIR / 'train/labels',
        transforms=get_transform(train=True),
        ignore_list=config.TRAIN_IGNORE
    )
    val_dataset = CharacterLocalizationDataset(
        image_dir=config.DATA_DIR / 'valid/images',
        label_dir=config.DATA_DIR / 'valid/labels',
        transforms=get_transform(train=False),
        ignore_list=config.VAL_IGNORE
    )

    # reproducibility
    # g = torch.Generator()
    # g.manual_seed(config.SEED)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        # shuffle=True,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    print("DataLoaders created successfully.")

    # Instantiate the Model
    print("Initializing model...")
    custom_backbone = ResNet50FPN()
    model = CustomFasterRCNN(
        backbone=custom_backbone,
        num_classes=config.NUM_CLASSES,
    )
    model.to(config.DEVICE)
    print(f"Model created and moved to device: {config.DEVICE}")

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=config.LEARNING_RATE,
        momentum=config.MOMENTUM,
        weight_decay=config.WEIGHT_DECAY
    )
    print("Optimizer created successfully (SGD).")

    # # Optimizer
    # params = [p for p in model.parameters() if p.requires_grad]
    # optimizer = torch.optim.Adam(
    #     params,
    #     lr=config.LEARNING_RATE,
    #     weight_decay=config.WEIGHT_DECAY
    # )
    # print("Optimizer created successfully (Adam).")

    # Optimizer
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0005)
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer,
    #     T_max=config.NUM_EPOCHS,
    #     eta_min=0.00001
    # )

    # Training Loop Setup
    val_map_history = []
    best_map_score = -1.0
    metric = MeanAveragePrecision(iou_type="bbox").to(config.DEVICE)

    print("\n--- Starting Training ---")
    for epoch in range(config.NUM_EPOCHS):
        # Training Phase
        model.train()
        train_loss = 0.0
        train_progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS} [Training]")
#
        for images, targets in train_progress_bar:
            if os.path.exists(config.PAUSE_FILE):
                print("Training paused... remove pause.txt to resume.")
                while os.path.exists(config.PAUSE_FILE):
                    time.sleep(5)
                print("Resuming training...")
#
            images = list(image.to(config.DEVICE) for image in images)
            targets = [{k: v.to(config.DEVICE) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            train_loss += losses.item()
            train_progress_bar.set_postfix(loss=f"{losses.item():.4f}")

        avg_train_loss = train_loss / len(train_loader)

        # Validation Phase
        model.eval()
        with torch.no_grad():
            val_progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS} [Validation]")
            for images, targets in val_progress_bar:
                images = list(image.to(config.DEVICE) for image in images)
                targets_val = [{k: v.to(config.DEVICE) for k, v in t.items()} for t in targets]

                predictions = model(images)
                metric.update(predictions, targets_val)

        # lr_scheduler.step()
        
        map_stats = metric.compute()
        current_map = map_stats['map'].item()
        val_map_history.append(current_map)
        metric.reset()

        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  - Average Training Loss: {avg_train_loss:.4f}")
        print(f"  - Validation mAP: {current_map:.4f}")

        # Check improvement
        if current_map > best_map_score:
            best_map_score = current_map
            epochs_without_improvement = 0  # reset patience counter
            torch.save(model.state_dict(), config.BEST_MODEL_PATH)
            print(f"  - New best model saved to {config.BEST_MODEL_PATH} (mAP: {best_map_score:.4f})")
        else:
            epochs_without_improvement += 1
            print(f"  - No improvement for {epochs_without_improvement} epoch(s)")

        # Early stopping check
        if epochs_without_improvement >= config.PATIENCE:
            print(f"\nEarly stopping triggered after {config.PATIENCE} epochs without improvement.")
            break

    print("\nTraining Finished!")


if __name__ == '__main__':
    run_training()