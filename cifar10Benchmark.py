import torch
import argparse
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18
from torchvision.transforms import ToTensor
import time
from memory_profiler import profile
from utils import  get_training_dataloader, get_test_dataloader

parser = argparse.ArgumentParser(description='Training ResNet18 on CIFAR-100 with Contiguity Aware Augmentation')
parser.add_argument('--is_contiguous', type=str, choices=["true", "false"], default="true",
                    help='Whether to make the augmented tensors contiguous (true or false)')

args = parser.parse_args()
# Convert string input to boolean
is_contiguous = args.is_contiguous.lower() == "true"

# Define custom normalization
class CustomNormalize:
    def __init__(self, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        img = transforms.functional.normalize(img, mean=self.mean, std=self.std).contiguous()
        return img

class CustomRandomAugment_default(transforms.Compose):
    def __init__(self, transforms):
        super().__init__(transforms)
    
    def __call__(self, img):

        # Apply random augmentations sequentially
        for op in self.transforms:
            img = op(img)
        return img

class CustomRandomAugment(transforms.Compose):
    def __init__(self, transforms):
        super().__init__(transforms)

    def __call__(self, img):

        # Apply random augmentations sequentially
        for op in self.transforms:
            img = op(img).contiguous()
        return img


# Example usage
custom_augmentations = [
    transforms.RandomCrop(32, padding=4),  # Random cropping
    transforms.RandomVerticalFlip(),
    transforms.RandomHorizontalFlip(),    # Horizontal flip
    transforms.RandomRotation(degrees=(-180, 180)),  # Random rotation
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Color jittering
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),  # Random resized crop
    transforms.RandomGrayscale(p=0.2),      # Random grayscale conversion (optional)
]

transform = None
if is_contiguous:
    print(f'Contiguous transform')
# Define transform with custom random augmentations
    transform = transforms.Compose([
    CustomRandomAugment(custom_augmentations),
    CustomNormalize(),
])
else:
    print(f'Default transform')
    transform = transforms.Compose([
    CustomRandomAugment_default(custom_augmentations),
    CustomNormalize(),
])


# Function to check contiguity status
def is_contiguous(tensor):
    """Checks if a tensor is contiguous in memory."""
    return tensor.is_contiguous()


#@profile
def data_augmentation_loop(inputs, transform):
    """Performs data augmentation on a single batch of inputs."""
    start_op_time = time.time()
    for op in transform.transforms:
        inputs = op(inputs)
    end_op_time = time.time()
    return inputs


#Load CIFAR-10 dataset
train_dataset = CIFAR10(root='./data', train=True, download=True, transform=ToTensor())
test_dataset = CIFAR10(root='./data', train=False, download=True, transform=ToTensor())

#Define data loaders
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=1)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=1)

cifar100_training_loader = get_training_dataloader(
        num_workers=1,
        batch_size=32,
        shuffle=True,
    )
cifar100_test_loader = get_test_dataloader(
        num_workers=1,
        batch_size=32,
        shuffle=True,
    )


# Define model
model = resnet18(pretrained=False, num_classes=100)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

total_augmentAndNonContigTime_oneEpoch = 0

for epoch in range(1):
    model.train()
    running_loss = 0.0
    
    total_augmentAndNonContigTime_oneEpoch = 0
    
    start_time = time.time()   

    #for inputs, labels in cifar100_training_loader:
    for inputs, labels in train_loader:
        # Apply random augmentations sequentially with timing measurement
        start_op_time = time.time()
       
        inputs = data_augmentation_loop(inputs, transform)

        end_op_time = time.time()
        total_augmentAndNonContigTime_oneEpoch += (end_op_time - start_op_time)

if is_contiguous:
    print(f"Contiguity aware: {total_augmentAndNonContigTime_oneEpoch}")
else:
    print(f"Pytorch default: {total_augmentAndNonContigTime_oneEpoch}")

end_time = time.time()
epoch_time = end_time - start_time
#print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss}, Time: {epoch_time:.6f} seconds")

