import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18
from torchvision.transforms import ToTensor
import time

from memory_profiler import profile

def my_is_contiguous(tensor):
    expected_stride = 1
    for size, stride in zip(reversed(tensor.size()), reversed(tensor.stride())):
        if stride != expected_stride:
            return False
        expected_stride *= size
    return True

# Define custom normalization
class CustomNormalize:
    def __init__(self, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        # Convert PIL image to PyTorch tensor
        #img = transforms.functional.to_tensor(img)
        # Normalize the image
        img = transforms.functional.normalize(img, mean=self.mean, std=self.std)
        print(f"After Normalize: device={img.device}, Shape: {img.shape}, Pytorch contiguous={img.is_contiguous()}, Custom check: {my_is_contiguous(img)},  stride: {img.stride()}, memory address: {img.data_ptr()}")
        #print(f"After {op.__class__.__name__}: contiguous={img.is_contiguous()}")
        return img

class CustomRandomAugment(transforms.Compose):
    def __init__(self, transforms):
        super().__init__(transforms)
    
    def __call__(self, img):
        # Convert PIL image to PyTorch tensor (potentially non-contiguous)
        #img = transforms.functional.to_tensor(img)
        print(f"Initialize with converting to_tensor: Shape: {img.shape}, Pytorch contiguous={img.is_contiguous()},  Custom check: {my_is_contiguous(img)},  stride: {img.stride()}, memory address: {img.data_ptr()}")


        # Apply random augmentations sequentially
        for op in self.transforms:
            # Use this to make sure tensor is contiguous after the operation applied, forcefully.
            img = op(img).contiguous()
            # Use this for pytorch default memory layout for the tensor after the operation
            #img = op(img)
            #print(f"{op.__class__.__name__}: device={img.device}, Shape: {img.shape}, Pytorch contiguous={img.is_contiguous()},  Custom check: {my_is_contiguous(img)}, stride: {img.stride()}, memory address: {img.data_ptr()}")

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
    transforms.Pad(4),  # Pad the image by 4 pixels on each side
    transforms.CenterCrop(32),  # Center crop to 32x32
    transforms.RandomVerticalFlip(p=1.0),  # Always flip vertically
    transforms.RandomHorizontalFlip(p=1.0),  # Always flip horizontally
    transforms.RandomRotation(degrees=45),  # Rotate by 45 degrees
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Apply fixed color jitter
    transforms.Resize((224, 224)),  # Resize to 224x224
    transforms.Grayscale(num_output_channels=3)
]


# Define transform with custom random augmentations
transform = transforms.Compose([
    CustomRandomAugment(custom_augmentations),
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
    print(f"After {op.__class__.__name__}: contiguous={inputs.is_contiguous()}, Time taken: {end_op_time - start_op_time:.6f} seconds")
    return inputs

# Load CIFAR-10 dataset
train_dataset = CIFAR10(root='./data', train=True, download=True, transform=ToTensor())
test_dataset = CIFAR10(root='./data', train=False, download=True, transform=ToTensor())

# Define data loaders
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=1)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=1)

# Define model
model = resnet18(pretrained=False, num_classes=10)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(1):
    model.train()
    running_loss = 0.0
    
    total_augmentAndNonContigTime_oneEpoch = 0
    
    

    for inputs, labels in train_loader:
        #inputs, labels = inputs.to(device), labels.to(device)
        
        # Apply random augmentations sequentially with timing measurement
        start_op_time = time.time()
       
        inputs = data_augmentation_loop(inputs, transform)

        end_op_time = time.time()
        #print(f"After {op.__class__.__name__}: contiguous={inputs.is_contiguous()}, Time taken: {end_op_time - start_op_time:.6f} seconds")
        total_augmentAndNonContigTime_oneEpoch += (end_op_time - start_op_time)
        # Forward pass, backward pass, and optimization
        #optimizer.zero_grad()
        #outputs = model(inputs)
        #loss = criterion(outputs, labels)
        #loss.backward()
        #optimizer.step()
        
        #running_loss += loss.item()

    print(f"Whole epoch augment with force contig time: {total_augmentAndNonContigTime_oneEpoch}")
    end_time = time.time()
    epoch_time = end_time - start_time
    #print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss}, Time: {epoch_time:.6f} seconds")

