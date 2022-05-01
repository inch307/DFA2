import torch.nn as nn
import torch

## simple linear network only support mnist, cifar10, cifar100

class SimpleLinear1(nn.Module):
    def __init__(self, *args, **kwargs):
        super(SimpleLinear1, self).__init__()
        self.device = kwargs['device']
        self.args = kwargs['args']
        if self.args.activation == 'tanh':
            self.activation = nn.Tanh()
        elif self.args.activation == 'relu':
            self.activation = nn.ReLU()

        self.layer_lst = []

        ## input layer
        if self.args.dataset == 'mnist':
            fc1 = nn.Linear(in_features=28*28, out_features=800, bias=True)
        elif self.args.dataset in ['cifar10', 'cifar100']:
            fc1 = nn.Linear(in_features=32*32*3, out_features=800, bias=True)
        self.append_layer(fc1)
        fc2 = nn.Linear(800, 400, bias=True)
        self.append_layer(fc2)
        
        for i in range(2):
            self.append_layer(nn.Linear(400, 400, bias=True))
        fc3 = nn.Linear(400, 100, bias=True)
        self.append_layer(fc3)
        
        self.out = nn.Linear(100, 10, bias=True)
        self.layer_lst.append(self.out)
        # self.layer_lst.append(nn.Softmax())

        self.sequential_layer = nn.Sequential(*self.layer_lst)

        # TODO:
        if self.args.model in 'dfa':
            # register forward hook

            self.B1 = self.get_projection_matrix(800)
            self.B2 = self.get_projection_matrix(400)
            self.B_lst = []
            for i in range(30):
                self.B_lst.append(self.get_projection_matrix(400))
            self.B3 = self.get_projection_matrix(100)

    def append_layer(self, layer):
        self.layer_lst.append(layer)
        self.layer_lst.append(self.activation)
        # if self.activation == 'tanh':
        #     self.layer_lst.append(nn.Tanh())
        # elif self.activation == 'relu':
        #     self.layer_lst.append(nn.ReLU())
        

    def forward(self, x):
        x = self.sequential_layer(x)

        return x
