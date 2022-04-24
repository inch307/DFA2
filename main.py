import argparse
from email.policy import default
import torchvision
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from net import Net
from net import DFANet
from net import DFA2Net
from utils import *
from tqdm import tqdm

parser = argparse.ArgumentParser(description='a')
parser.add_argument('-m', '--model', default='backprop', help='backprop for backprop, dfa for normal dfa, dfa2 for proposed method')
parser.add_argument('-n', '--net', default='simple_conv1', help = 'simple_conv1 / simple_conv2 / simple_linear1 / simple_linear2 / resnset18')
parser.add_argument('-d', '--dataset', default='mnist')
parser.add_argument('-o', '--optim', default='sgd', help='adam, rmsprop')
parser.add_argument('-e', '--epochs', default=50, type=int)
parser.add_argument('--device', default='cuda')
parser.add_argument('--drop_out', default=0, type=float)
parser.add_argument('--lr', default=1e-2, type=float)
parser.add_argument('--weight_decay', default=0, type=float)
parser.add_argumnet('--batch_size', default=64, type=int)
parser.add_argument('--val_batch_size', default=1024, type=int)
parser.add_argument('--experiment', action='store_true', default=False)

def main():
    args = parser.parse_args()
    device = torch.device(args.device)

    train_dataset, val_dataset = get_dataset(args)
    train_loader, val_loader = get_dataloader(train_dataset, val_dataset)
    optimizer = get_optim(args)

    net = get_network(args)

    train(net, train_loader, val_loader, optimizer, args)

    if args.model == 'sup':
        net = Net(device=device)
        # optimizer = torch.optim.SGD(net.parameters(), lr=0.005)
        optimizer = torch.optim.RMSprop(net.parameters(), lr=0.001)
        for i in range(args.epochs):
            train_backprop(net, train_loader, optimizer, device)
            val(net, val_loader, device)    
    elif args.model == 'dfa':
        net = DFANet(device=device)
        # optimizer = torch.optim.SGD(net.parameters(), lr=0.001)
        optimizer = torch.optim.RMSprop(net.parameters(), lr=0.001)
        # net.init('zero')
        for i in range(args.epochs):
            train_dfa(net, train_loader, optimizer, device, lr)    
            val(net, val_loader, device)
    elif args.model == 'dfa2' and args.BFirst == False:
        net = DFA2Net(device=device)
        # optimizer = torch.optim.SGD(net.parameters(), lr=0.001)
        optimizer = torch.optim.RMSprop(net.parameters(), lr=0.001)
        # net.init('zero')
        for i in range(args.epochs):
            train_dfa2(net, train_loader, optimizer, device, lr, args.post)    
            val(net, val_loader, device)
    elif args.model == 'dfa2' and args.BFirst == True:
        net = DFA2Net(device=device)
        # optimizer = torch.optim.SGD(net.parameters(), lr=0.001)
        optimizer = torch.optim.RMSprop(net.parameters(), lr=0.001)
        net.init('zero')
        for i in range(args.epochs):
            train_dfa3(net, train_loader, optimizer, device, lr, 2)    
            val(net, val_loader, device)


main()