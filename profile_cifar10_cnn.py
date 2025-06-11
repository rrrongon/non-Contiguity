import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.profiler import profile, ProfilerActivity
import time

def enforce_contiguity(fn):
    def wrapper(*args, **kwargs):
        new_args = [arg.contiguous() if isinstance(arg, torch.Tensor) and not arg.is_contiguous() else arg for arg in args]
        new_kwargs = {k: v.contiguous() if isinstance(v, torch.Tensor) and not v.is_contiguous() else v for k, v in kwargs.items()}
        return fn(*new_args, **new_kwargs)
    return wrapper

class CNNBasic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 3)  # 32x32 -> 30x30
        self.conv2 = torch.nn.Conv2d(16, 32, 3) # 30x30 -> 28x28
        self.fc = torch.nn.Linear(32 * 28 * 28, 10)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = x.transpose(2, 3)
        x = torch.relu(self.conv2(x))
        x = x.reshape(x.size(0), -1)
        return self.fc(x)

class CNNDecorated(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 3)
        self.conv2 = torch.nn.Conv2d(16, 32, 3)
        self.fc = torch.nn.Linear(32 * 28 * 28, 10)
    
    @enforce_contiguity
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = x.transpose(2, 3)
        x = torch.relu(self.conv2(x))
        x = x.reshape(x.size(0), -1)
        return self.fc(x)

class CNNManual(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 3)
        self.conv2 = torch.nn.Conv2d(16, 32, 3)
        self.fc = torch.nn.Linear(32 * 28 * 28, 10)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = x.transpose(2, 3)
        if not x.is_contiguous():
            x = x.contiguous()
        x = torch.relu(self.conv2(x))
        x = x.reshape(x.size(0), -1)
        return self.fc(x)

# Dataset and Augmentations
def get_cifar10_loader(augment=False):
    transform_list = [transforms.ToTensor()]
    if augment:
        transform_list = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
        ]
    transform = transforms.Compose(transform_list)
    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    return DataLoader(dataset, batch_size=32, shuffle=True)

def run_test(models, loader, name, augment_str):
    print("\n" + "="*60)
    print(f"CIFAR-10 {name.upper()} ({augment_str}) FORWARD PASS COMPARISON")
    print("="*60)
    
    for model_name, model in models.items():
        print(f"\nTesting {model_name.lower()} model...")
        try:
            data_iter = iter(loader)
            images, _ = next(data_iter)
            output = model(images)
            print(f"{model_name} model succeeded! Output shape:", output.shape)
        except Exception as e:
            print(f"{model_name} model failed: {str(e)}")
        
        print(f"\nProfiling {model_name.lower()} model...")
        try:
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True) as prof:
                output = model(images)
            print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=5))
        except Exception as e:
            print(f"Profiling failed: {str(e)}")

def time_comparison(models, loader, augment_str, repeats=100):
    print("\n" + "="*60)
    print(f"CIFAR-10 EXECUTION TIME COMPARISON ({augment_str}, AVERAGE OVER {repeats} RUNS)")
    print("="*60)
    
    data_iter = iter(loader)
    images, _ = next(data_iter)
    
    def time_op(fn, *args):
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(repeats):
            fn(*args)
            torch.cuda.synchronize()
        return (time.time() - start) / repeats
    
    for name, model in models.items():
        try:
            avg_time = time_op(model, images)
            print(f"{name} model succeeded: {avg_time*1000:.3f} ms")
        except Exception as e:
            print(f"{name} model failed: {str(e)}")

if __name__ == "__main__":
    models = {"Basic": CNNBasic(), "Manual": CNNManual(), "Decorated": CNNDecorated()}
    
    # Without augmentations
    loader_no_aug = get_cifar10_loader(augment=False)
    run_test(models, loader_no_aug, "CNN", "No Augmentation")
    time_comparison(models, loader_no_aug, "No Augmentation")
    
    # With augmentations
    loader_aug = get_cifar10_loader(augment=True)
    run_test(models, loader_aug, "CNN", "With Augmentation")
    time_comparison(models, loader_aug, "With Augmentation")
