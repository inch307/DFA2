import torch
import torch.nn as nn
import math

class OutputLinear(nn.Module):
    def __init__(self, *args, **kwargs):
        super(OutputLinear, self).__init__()
        # self.device = kwargs['device']
        # self.args = kwargs['args']
        self.in_features = kwargs['in_features']
        self.bias = kwargs['bias']
        self.num_classes = kwargs['num_classes'] # for DFA
        
        self.fc = nn.Linear(in_features=self.in_features, out_features=self.num_classes, bias=self.bias)
        self.activation = nn.Softmax(dim=1)

    def forward(self, x):
        self.input = x
        self.output = self.fc(x)
        self.y_hat = self.activation(self.output)
        log_y_hat = torch.log(self.y_hat)

        return log_y_hat

    def zero_init(self):
        self.fc.weight.data.fill_(0)
        if self.fc.bias is not None:
            self.fc.bias.data.fill_(0)

    def kaiming_init(self):
        pass

    def xavier_init(self):
        pass

    def random_projection_matrix_init(self):
        return

    def dfa_grad(self):
        self.fc.weight.grad = self.fc.weight.dfa_grad.detach()
        if self.fc.bias is not None:
            self.fc.bias.grad = self.fc.bias.dfa_grad.detach()

    def dfa_backward(self, e):
        with torch.no_grad():
            # print(e)
            # print(e.shape)
            self.fc.weight.dfa_grad = torch.matmul(torch.t(e), self.input) / self.input.size(0)
            if self.fc.bias is not None:
                self.fc.bias.dfa_grad = torch.sum(e, 0) / self.input.size(0)
        return 