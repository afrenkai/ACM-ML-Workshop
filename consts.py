from pathlib import Path
import torchvision
import torchvision.transforms as transforms
import os
CIFAR_DIR = Path('Data/CIFAR10')
CIFAR_DIR.mkdir(parents = True, exist_ok = True)

#you can add more and bump the numbers even more up
normalize = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

TRAIN_DATA = torchvision.datasets.CIFAR10(root = CIFAR_DIR, train = True, transform = normalize, download = True)
TEST_DATA = torchvision.datasets.CIFAR10(root = CIFAR_DIR, train = False, transform = normalize, download = True)


CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


