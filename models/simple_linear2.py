import torch.nn as nn
import torch
import math

## simple linear network only support mnist, cifar10, cifar100

class SimpleLinear2(nn.Module):
    def __init__(self, *args, **kwargs):
        super(SimpleLinear2, self).__init__()
        self.device = kwargs['device']
        self.args = kwargs['args']
        if self.args.activation == 'tanh':
            self.activation = nn.Tanh()
        elif self.args.activation == 'relu':
            self.activation = nn.ReLU()

        ############ make layers ##################

        self.layer_lst = []

        ## input layer
        if self.args.dataset == 'mnist':
            fc1 = nn.Linear(in_features=28*28, out_features=800, bias=True)
        elif self.args.dataset in ['cifar10', 'cifar100']:
            fc1 = nn.Linear(in_features=32*32*3, out_features=800, bias=True)
        self.append_layer(fc1)
        fc2 = nn.Linear(800, 400, bias=True)
        self.append_layer(fc2)
        
        for i in range(20):
            self.append_layer(nn.Linear(400, 400, bias=True))
        fc3 = nn.Linear(400, 100, bias=True)
        self.append_layer(fc3)
        
        self.out = nn.Linear(100, 10, bias=True)
        self.layer_lst.append(self.out)
        # self.layer_lst.append(nn.Softmax())

        self.sequential_layer = nn.Sequential(*self.layer_lst)

        ############ make layers ##################

        # TODO:
        if self.args.model == 'dfa':
            # register forward hook
            self.reg_forward_hook()

            if self.args.dataset == 'mnist' or self.args.dataset == 'cifar10' or self.args.dataset == 'stl10':
                output_size = 10
            elif self.args.dataset == 'cifar100':
                output_size = 100
            elif self.args.dataset == 'imagenet':
                output_size = 1000

            for name, module in self.sequential_layer.named_modules():
                if isinstance(module, nn.Linear):
                    module.B = self.get_projection_matrix(module.out_features, output_size, math.sqrt(1/output_size))

        elif self.args.model == 'dfa2':
            # register forward hook
            self.reg_forward_hook()

            if self.args.dataset == 'mnist' or self.args.dataset == 'cifar10' or self.args.dataset == 'stl10':
                output_size = 10
            elif self.args.dataset == 'cifar100':
                output_size = 100
            elif self.args.dataset == 'imagenet':
                output_size = 1000

            for name, module in self.sequential_layer.named_modules():
                if isinstance(module, nn.Linear):
                    module.B = self.get_projection_matrix(module.out_features, self.layer_lst[-1].in_features, math.sqrt(1/self.layer_lst[-1].in_features))
            self.layer_lst[-3].B = self.get_projection_matrix(self.layer_lst[-3].out_features, output_size, math.sqrt(1/output_size))

        self.weight_init()

    def reg_forward_hook(self):
        # named_modules return [squentals, layer0, layer1, ...]
        for name, module in self.sequential_layer.named_modules():
            if isinstance(module, nn.Linear):
                module.register_forward_hook(self.save_forward_hook(name))

    def append_layer(self, layer):
        self.layer_lst.append(layer)
        self.layer_lst.append(self.activation)
        # if self.activation == 'tanh':
        #     self.layer_lst.append(nn.Tanh())
        # elif self.activation == 'relu':
        #     self.layer_lst.append(nn.ReLU())

    # TODO: appropriate init
    def get_projection_matrix(self, out_features, output_size, std=1):
        # B = torch.randn(output_size, out_features).to(self.device)
        B = (torch.rand((output_size, out_features), device=self.device) - 0.5) * 2 * math.sqrt(3) * std
        return B

    def save_forward_hook(self, layer_id):
        def hook(module, input, output):
            module.input = input[0]
            module.output = output
        return hook

    def forward(self, x):
        x = self.sequential_layer(x)

        return x

    def weight_init(self):
        if self.args.init == 'zero':
            for name, module in self.sequential_layer.named_modules():
                if isinstance(module, nn.Linear):
                    module.weight.data.fill_(0)
                    if module.bias is not None:
                        module.bias.data.fill_(0)
        else:
            for name, module in self.sequential_layer.named_modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_normal_(module.weight) #TODO: relu, other activations
                    if module.bias is not None:
                        module.bias.data.fill_(0)

            