import dataset
import torch
import torch.optim as optim

from models import NetworkBuilder

def get_dataset(args):
    dataset = args.dataset

    if dataset == 'mnist':
        train_dataset, val_dataset = dataset.mnist_datset()
    elif dataset == 'cifar10':
        train_dataset, val_dataset = dataset.cifar10_dataset()
    elif dataset == 'cifar100':
        train_dataset, val_dataset = dataset.cifar100_dataset()
    elif dataset == 'stl10':
        train_dataset, val_dataset = dataset.stl10_dataset()
    elif dataset == 'imagenet':
        train_dataset, val_dataset = dataset.imagenet_dataset()
    elif dataset == 'smallimgaenet':
        train_dataset, val_dataset = dataset.smallimagenet_dataset()
    return train_dataset, val_dataset

def get_dataloader(train_dataset, val_dataset, args):
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last = True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=args.val_batch_size, shuffle=False, pin_memory=True, drop_last = True)
    return train_loader, val_loader

def get_optim(net, args):
    if args.optim == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'rmsprop':
        optimizer = optim.RMSprop(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    return optimizer

def get_network(args):
    
    if args.net == 'simple_conv1':
        return SimpleConv1(args)
    elif args.net == 'simple_conv2':
        return SimpleConv2(args)
    elif args.net == 'simple_linear1':
        return SimpleLinear1(args)
    elif args.net == 'simple_linear2':
        return SimpleLinear2(args)
    elif args.net == 'resnet18':
        return ResNet18(args)


def train(net, train_loader, val_loader, optimizer, args):
    # var
    
    #for epoch
        #data, train, if exp backprop backward, no_grad, dfa backward, step, if exp analysis (alignment, ...)