import dataset
import torch
import torch.optim as optim
import trainer

from models import simple_conv1, simple_conv2, simple_linear2, resnet18
from models import simple_linear1

def get_dataset(args):

    if args.dataset == 'mnist':
        train_dataset, val_dataset = dataset.mnist_dataset()
    elif args.dataset == 'cifar10':
        train_dataset, val_dataset = dataset.cifar10_dataset()
    elif args.dataset == 'cifar100':
        train_dataset, val_dataset = dataset.cifar100_dataset()
    elif args.dataset == 'stl10':
        train_dataset, val_dataset = dataset.stl10_dataset()
    elif args.dataset == 'imagenet':
        train_dataset, val_dataset = dataset.imagenet_dataset()
    elif args.dataset == 'smallimgaenet':
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

def get_network(device, args):
    
    if args.net == 'simple_conv1':
        return simple_conv1.SimpleConv1(device=device, args=args).to(device)
    elif args.net == 'simple_conv2':
        return simple_conv2.SimpleConv2(device=device, args=args).to(device)
    elif args.net == 'simple_linear1':
        return simple_linear1.SimpleLinear1(device=device, args=args, activation='tanh').to(device)
    elif args.net == 'simple_linear2':
        return simple_linear2.SimpleLinear2(device=device, args=args).to(device)
    elif args.net == 'resnet18':
        return resnet18.ResNet18(device=device, args=args).to(device)


def train(net, train_loader, val_loader, optimizer, device, args):
    if args.model == 'backprop':
        for epoch in range(args.epochs):
            print(f'epoch: {epoch}')
            trainer.train_backprop(net, train_loader, optimizer, device, args)
            trainer.val(net, val_loader, device, args)
    elif args.model == 'dfa' or args.model == 'dfa2':
        for epoch in range(args.epochs):
            print(f'epoch: {epoch}')
            trainer.train_dfa(net, train_loader, optimizer, device, args)
            trainer.val(net, val_loader, device, args)

    #for epoch
        #data, train, if exp backprop backward, no_grad, dfa backward, step, if exp analysis (alignment, ...)