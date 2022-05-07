import torch
import torch.nn as nn
import math

class LinearBlock(nn.Module):
    def __init__(self, *args, **kwargs):
        super(LinearBlock, self).__init__()
        self.device = kwargs['device']
        # self.args = kwargs['args']
        self.in_features = kwargs['in_features']
        self.out_features = kwargs['out_features']
        self.bias = kwargs['bias']
        self.num_classes = kwargs['num_classes'] # for DFA
        
        self.fc = nn.Linear(in_features=self.in_features, out_features=self.out_features, bias=self.bias)
        if kwargs['activation'] == 'tanh':
            self.activation = nn.Tanh()
        elif kwargs['activation'] == 'relu':
            self.activation = nn.ReLU()
        
    def forward(self, x):
        self.input = x
        self.output = self.fc(x)
        self.activation_out = self.activation(self.output)

        return self.activation_out

    def zero_init(self):
        self.fc.weight.data.fill_(0)
        if self.fc.bias is not None:
            self.fc.bias.data.fill_(0)

    def kaiming_init(self):
        pass

    def xavier_init(self):
        pass

    def get_projection_matrix(self, out_features, num_classes, std=1):
        # B = torch.randn(output_size, out_features).to(self.device)
        B = (torch.rand((num_classes, out_features), device=self.device) - 0.5) * 2 * math.sqrt(3) * std
        return B

    def random_projection_matrix_init(self):
        self.B = self.get_projection_matrix(self.fc.out_features, self.num_classes)
        return

    def dfa_grad(self):
        self.fc.weight.grad = self.fc.weight.dfa_grad.detach()
        if self.fc.bias is not None:
            self.fc.bias.grad = self.fc.bias.dfa_grad.detach()

    def dfa_backward(self, e):
        if isinstance(self.activation, nn.Tanh):
            with torch.no_grad():
                dx = torch.matmul(e, self.B) * (1-torch.tanh(self.output) ** 2) # TODO: tanh -> relu?
                # dx_norm = torch.linalg.vector_norm(dx, dim=1)
                # print('dx')
                # print(dx_norm)
                self.fc.weight.dfa_grad = torch.matmul(torch.t(dx), self.input) / self.input.size(0)
                
                if self.fc.bias is not None:
                    self.fc.bias.dfa_grad = torch.sum(dx, 0) / self.input.size(0)
        return 