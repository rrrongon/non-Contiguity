from torchvision.transforms import transforms
from RandAugment import RandAugment

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
])

# Add RandAugment with N, M(hyperparameter)
transform_train.transforms.insert(0, RandAugment(N, M))
