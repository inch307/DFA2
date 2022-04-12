import argparse
import torchvision
import torch
from torch.utils.data import DataLoader
import argparse
from net import Net
from utils import *

parser = argparse.ArgumentParser(description='a')
parser.add_argument('-m', '--model', default='dfa')
parser.add_argument('-d', '--device', default='cuda')

def val():
    pass

def train():
    pass

def main():
    args = parser.parse_args()
    device = torch.device(args.device)

    train_dataset, val_dataset = get_dataset()
    train_loader, val_loader = get_dataloader(train_dataset, val_dataset)
    net = Net(args.model)

    optimizer = torch.optim.SGD(net.parameters(), lr=0.001)
    



main()