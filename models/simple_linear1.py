import torch.nn as nn
import torch
import math
from dfa import dfa_backward
from .linear_block import LinearBlock
from .output_linear import OutputLinear

## simple linear network only support mnist, cifar10, cifar100

class SimpleLinear1(nn.Module):
    def __init__(self, *args, **kwargs):
        super(SimpleLinear1, self).__init__()
        self.device = kwargs['device']
        self.args = kwargs['args']
        self.activation = kwargs['activation']

        if self.args.dataset == 'mnist' or self.args.dataset == 'cifar10' or self.args.dataset == 'stl10':
            self.num_classes = 10
        elif self.args.dataset == 'cifar100':
            self.num_classes = 100
        elif self.args.dataset == 'imagenet':
            self.num_classes = 1000

        ############ make layers ##################

        self.layer = []

        ## input layer
        if self.args.dataset == 'mnist':
            fc1 = LinearBlock(in_features=28*28, out_features=800, bias=True, num_classes=self.num_classes, activation=self.activation, device=self.device)
        elif self.args.dataset in ['cifar10', 'cifar100']:
            fc1 = LinearBlock(in_features=32*32*3, out_features=800, bias=True, num_classes=self.num_classes, activation=self.activation, device=self.device)
        self.layer.append(fc1)

        fc2 = LinearBlock(in_features=800, out_features=400, bias=True, num_classes=self.num_classes, activation=self.activation, device=self.device)
        self.layer.append(fc2)
        
        for i in range(3):
            self.layer.append(LinearBlock(in_features=400, out_features=400, bias=True, num_classes=self.num_classes, activation=self.activation, device=self.device))
        fc3 = LinearBlock(in_features=400, out_features=100, bias=True, num_classes=self.num_classes, activation=self.activation, device=self.device)
        self.layer.append(fc3)
        
        self.out = OutputLinear(in_features=100, num_classes=10, bias=True)
        self.layer.append(self.out)
        # self.layer_lst.append(nn.Softmax())

        self.layer = nn.ModuleList(self.layer)

        ############ make layers ##################

        # TODO:
        if self.args.model == 'dfa':
            for idx, module in enumerate(self.layer):
                module.random_projection_matrix_init()
                module.zero_init()

        elif self.args.model == 'dfa2':
            for idx, module in enumerate(self.layer):
                module.random_projection_matrix_init()
                module.zero_init()


        # elif self.args.model == 'dfa2':
        #     # register forward hook
        #     self.reg_forward_hook()

        #     if self.args.dataset == 'mnist' or self.args.dataset == 'cifar10' or self.args.dataset == 'stl10':
        #         output_size = 10
        #     elif self.args.dataset == 'cifar100':
        #         output_size = 100
        #     elif self.args.dataset == 'imagenet':
        #         output_size = 1000

        #     for name, module in self.sequential_layer.named_modules():
        #         if isinstance(module, nn.Linear):
        #             module.B = self.get_projection_matrix(module.out_features, self.layer_lst[-1].in_features, math.sqrt(1/self.layer_lst[-1].in_features))
        #     self.layer_lst[-3].B = self.get_projection_matrix(self.layer_lst[-3].out_features, output_size)

    def forward(self, x):
        for module in self.layer:
            x = module(x)

        return x

    def dfa_backward(self, e, idx):
        for i in idx:
            self.layer[i].dfa_backward(e)

    def dfa_B_update(self, z, lr):
        for i in range(len(self)-1):
            self.layer[i].dfa_B_update(z, lr)

    def dfa_grad(self, idx):
        for i in idx:
            self.layer[i].dfa_grad()

    def __len__(self):
        return len(self.layer)