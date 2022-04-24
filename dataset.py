from torchvision.datasets import MNIST, CIFAR10, CIFAR100, STL10
import torchvision.transforms as transforms
import torch
import os
from PIL import Image
import scipy.io as sio

def mnist_dataset():
    data_path = './data/mnist'
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (1.0,))
    ])
    train_dataset = MNIST(data_path, train=True, download=True, transform=transform)
    val_dataset = MNIST(data_path, train=False, download=True, transform=transform)

    return train_dataset, val_dataset

def cifar10_dataset():
    data_path = './data/cifar10'
    train_dataset = CIFAR10(data_path, train=True, download=True,
                                    transform=transforms.Compose(
                                        [
                                            transforms.RandomCrop(32, padding=4),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                        ]
                                    )
    )
    val_dataset = CIFAR10(data_path, train=False, download=True, 
                                    transform=transforms.Compose(
                                        [
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                        ]
                                    )
    )
    
    return train_dataset, val_dataset

def cifar100_dataset():
    data_path = './data/cifar100'
    train_dataset = CIFAR100(data_path, train=True, download=True,
                                    transform=transforms.Compose(
                                        [
                                            transforms.RandomCrop(32, padding=4),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                        ]
                                    )
    )
    val_dataset = CIFAR100(data_path, train=False, download=True, 
                                    transform=transforms.Compose(
                                        [
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                        ]
                                    )
    )
    
    return train_dataset, val_dataset

def stl10_dataset():
    data_path = '/data/stl10'
    train_dataset = STL10(data_path, download=True, split='train',
                                    transform=transforms.Compose(
                                        [
                                            transforms.RandomCrop(96, padding=4),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                        ]
                                    )
    )
    val_dataset = STL10(data_path, download=True, split='test',
                                    transform=transforms.Compose(
                                        [
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                        ]
                                    )
    )

    return train_dataset, val_dataset


def imagenet_dataset():
    data_path = '../sdb1/ImageNet'
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip,
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    train_dataset = ImagenetTrainDataset(data_path, transform=train_transform)
    val_dataset = ImagenetTestDataset(data_path, transform=val_transform)

    return train_dataset, val_dataset

def small_imagenet_dataset():
    data_path = '../sdb1/ImageNet'
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    train_dataset = SmallImagenetTrainDataset(data_path, transform=transform)
    val_dataset = ImagenetTestDataset(data_path, transform=transform)

    return train_dataset, val_dataset


# train
# n{label}_{id}.JPEG
# 1281167
# 1000 classes

# val
# validation ground truth
class ImagenetTrainDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, transform=None):
        self.data_root = data_root + '/train/'
        self.transform = transform
        # self.label = self.get_label()

        names = os.listdir(self.data_root)
        mat_file = sio.loadmat(data_root + '/meta.mat')
        label_dict = {}
        for i in range(1000):
            label_dict[mat_file['synsets'][i][0][1][0]] = i

        self.data_list = []
        for name in names:
            id = name.split('_')[0]
            self.data_list.append((name, label_dict[id]))
        # print(self.data_list)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        x = Image.open(self.data_root + self.data_list[idx][0])
        x = x.convert('RGB')
        y = self.data_list[idx][1]
        if self.transform:
            x = self.transform(x)
        return x, y

class SmallImagenetTrainDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, transform=None):
        self.data_root = data_root + '/train/'
        self.transform = transform
        # self.label = self.get_label()

        names = os.listdir(self.data_root)
        mat_file = sio.loadmat(data_root + '/meta.mat')
        label_dict = {}
        label_count = {}
        for i in range(1000):
            label_dict[mat_file['synsets'][i][0][1][0]] = i
            label_count[i] = 0

        self.data_list = []
        for name in names:
            id = name.split('_')[0]
            if label_count[label_dict[id]] < 100:
                label_count[label_dict[id]] += 1
                self.data_list.append((name, label_dict[id]))
            
        # print(self.data_list)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        x = Image.open(self.data_root + self.data_list[idx][0])
        x = x.convert('RGB')
        y = self.data_list[idx][1]
        if self.transform:
            x = self.transform(x)
        return x, y


class ImagenetTestDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, transform=None):
        self.data_root = data_root + '/val/'
        self.label_root = data_root
        self.names = sorted(os.listdir(self.data_root))
        self.transform = transform
        self.label = self.get_label()

    def __len__(self):
        return len(self.names)

    def get_label(self):
        labels = []
        ground_truth = open(self.label_root + '/ILSVRC2012_validation_ground_truth.txt', 'r')
        while True:
            line = ground_truth.readline()
            if not line: break
            labels.append(int(line)-1)
        ground_truth.close()
        
        return labels

    def __getitem__(self, idx):
        x = Image.open(self.data_root + self.names[idx])
        x = x.convert('RGB')
        y = self.label[idx]
        if self.transform:
            x = self.transform(x)
        return x, y

if __name__=='__main__':
    names = os.listdir('./ImageNet/train')
    print(len(names))
    names = os.listdir('./ImageNet/val')
    print(len(names))