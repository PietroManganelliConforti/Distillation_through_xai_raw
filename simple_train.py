import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import os
import numpy as np
import logging
from PIL import Image
import random

def set_deterministic_seed():
    """Imposta seed deterministico globale"""
    seed = 42  # Fisso
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    
    # # Per garantire riproducibilità completa
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


class ConvertToRGB(object):
    def __call__(self, img):
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return img

logger = logging.getLogger()

def get_train_and_test_loader_flowers102(data_folder='work/project/data/flowers102',
                                        batch_size=64,
                                        num_workers=4):
    """
    Load and prepare Flowers102 dataset with proper train/test split.
    """
    if data_folder != 'work/project/data/flowers102':
        logger.warning(f"Data folder path {data_folder} does not match expected 'work/project/data/flowers102'. "
                       "Ensure the dataset is in the correct location.")

    data_folder = os.path.join(data_folder)
    
    train_transform = transforms.Compose([
        ConvertToRGB(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        ConvertToRGB(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    try:
        train_dataset = datasets.Flowers102(root=data_folder, download=True, transform=train_transform)
        val_dataset = datasets.Flowers102(root=data_folder, download=True, transform=test_transform)
        logger.info(f"Successfully loaded Flowers102 dataset from {data_folder}")
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        print(f"Failed to load dataset from {data_folder}")
        print(f"Expected path structure: {data_folder}/flowers102/...")
        raise
        
    set_deterministic_seed()
    indices = list(range(len(train_dataset)))
    np.random.shuffle(indices)
    train_size = int(0.8 * len(train_dataset))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_set = Subset(train_dataset, train_indices)
    val_set = Subset(val_dataset, val_indices)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    n_classes = 102  # Flowers102 has 102 classes

    logger.info(f"Flowers102 loaded: {len(train_set)} train samples, {len(val_set)} val samples.")

    return train_loader, val_loader, n_classes

def get_train_and_test_loader_caltech256(data_folder='work/project/data/caltech256',
                                        batch_size=64,
                                        num_workers=4):
    """
    Load and prepare Caltech256 dataset with proper data splits
    """
    # Correct the path - the Caltech dataset should already be in the specified directory
    # based on the structure provided in the file path

    if data_folder != 'work/project/data/caltech256':
        logger.warning(f"Data folder path {data_folder} does not match expected 'work/project/data/caltech256'. "
                       "Ensure the dataset is in the correct location.")

    data_folder = os.path.join(data_folder)
    
    # Transformations for training and test
    train_transform = transforms.Compose([
        ConvertToRGB(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        ConvertToRGB(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
        

    # Load the full dataset (without automatic splitting)
    try:
        print(f"Loading Caltech256 dataset from {data_folder}")
        train_dataset = datasets.Caltech256(root=data_folder, download=False, transform=train_transform)
        val_dataset = datasets.Caltech256(root=data_folder, download=False, transform=test_transform)
        logger.info(f"Successfully loaded Caltech256 dataset from {data_folder}")
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        # Provide helpful debugging information
        print(f"Failed to load dataset from {data_folder}")
        print(f"Expected path structure: {data_folder}/caltech256/256_ObjectCategories/...")
        print("Please verify the dataset exists at this location or set download=True")
        raise

    set_deterministic_seed()
    # Split 80% train, 20% test
    indices = list(range(len(train_dataset)))

    np.random.shuffle(indices)
    train_size = int(0.8 * len(train_dataset))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_set = Subset(train_dataset, train_indices)
    val_set = Subset(val_dataset, val_indices)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    n_classes = 257  # Caltech256 has 257 classes

    logger.info(f"Caltech256 loaded: {len(train_set)} train samples, {len(val_set)} val samples.")

    return train_loader, val_loader, n_classes


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train model for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device):
    """Evaluate model on validation/test data"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def main():
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Training parameters
    batch_size = 32
    lr = 0.01
    epochs = 20
    
    # Import model_dict here to avoid circular imports
    from my_models import model_dict

    # Load data - fix the path to point to the parent directory
    data_path = 'work/project/data/flowers102'
    trainloader, testloader, num_classes = get_train_and_test_loader_flowers102(
        data_folder=data_path, 
        batch_size=batch_size, 
        num_workers=4
    )

    # Load pretrained ResNet18 model and adapt the last layer
    print(f"Creating model with {num_classes} output classes")
    model = model_dict["resnet18"](num_classes=num_classes, pretrained=True).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Training loop
    print(f"Starting training for {epochs} epochs")
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, trainloader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, testloader, criterion, device)

        print(f"Epoch {epoch+1}/{epochs}: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        scheduler.step()

    # Save the trained model
    save_path = "resnet18_caltech256.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Training complete! Model saved to {save_path}")








if __name__ == "__main__":
    # Configure basic logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    main()