import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch import nn
from torch.autograd import Variable

from trainings import test

from my_models import model_dict, ensemble_of_models
import timm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).to(device)

model.fc = nn.Linear(512, 10)

model.eval().to(device)

transform = transforms.Compose([
    transforms.Resize((224,224)),  # Ridimensiona le immagini a 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Carica il dataset Imagenette
test_dataset = ImageFolder(root='work/project/data/imagenette/imagenette2/val', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model.eval()

correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# Calcola e stampa l'accuratezza
accuracy = 100 * correct / total
print(f'Accuracy on the test set: {accuracy:.2f}%')



model = timm.create_model("resnet18", pretrained=True,num_classes=10).to(device)


model.eval().to(device)

transform = transforms.Compose([
    transforms.Resize((224,224)),  # Ridimensiona le immagini a 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Carica il dataset Imagenette
test_dataset = ImageFolder(root='work/project/data/imagenette/imagenette2/train', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model.eval()

correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# Calcola e stampa l'accuratezza
accuracy = 100 * correct / total
print(f'Accuracy on the test set: {accuracy:.2f}%')