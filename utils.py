import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import torch

def get_dataset():
    root = './mnist'
    m_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (1.0,))
    ])
    train_dataset = MNIST(root, transform=m_transform, train=True, download=True)
    val_datset = MNIST(root, transform=m_transform, train=False, download=True)

    return train_dataset, val_datset

def get_dataloader(train_dataset, val_dataset):
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True, pin_memory=True, drop_last = True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=1024, shuffle=False, pin_memory=True, drop_last = True)
    return train_loader, val_loader