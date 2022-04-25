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
            fc1 = nn.Linear(in_features=784, out_features=800, bias=True)
        elif self.args.dataset in 'cifar':
            fc1 = nn.Linear(in_features=1024, out_features=800, bias=True)
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
        
    # TODO: appropriate init
    def get_projection_matrix(self, hidden):
        B = torch.randn(10, hidden)

        return B

    def forward(self, x):
        x = self.sequential_layer(x)

        return x

    def dfa_backward(self, e, x):
        x = x.view(-1, 784)

        dx1 = torch.matmul(e, self.B1) * (1-torch.tanh(self.y1) ** 2)
        # print(dx1.shape)
        self.fc1.weight.grad = torch.matmul(torch.t(dx1), x) / 64 # batch szie
        # self.fc1.bias.grad = torch.sum(dx1, 0)

        dx2 = torch.matmul(e, self.B2) * (1-torch.tanh(self.y2) ** 2)
        self.fc2.weight.grad = torch.matmul(torch.t(dx2), self.z1) / 64 # batch szie
        # self.fc2.bias.grad = torch.sum(dx2, 0)

        dx = torch.matmul(e, self.B_lst[0]) * (1-torch.tanh(self.y_lst[0]**2))
        self.fc_lst[0].weight.grad = torch.matmul(torch.t(dx), self.z2)
        for i in range(20):
            dx = torch.matmul(e, self.B_lst[i+1]) * (1-torch.tanh(self.y_lst[i+1]**2))
            self.fc_lst[i+1].weight.grad = torch.matmul(torch.t(dx), self.z_lst[i])

        dx3 = torch.matmul(e, self.B3) * (1-torch.tanh(self.y3) ** 2)
        self.fc3.weight.grad = torch.matmul(torch.t(dx3), self.z_lst[-1]) / 64 # batch szie
        # self.fc3.bias.grad = torch.sum(dx3, 0)

        self.out.weight.grad = torch.matmul(torch.t(e), self.z3) / 64 # batch szie
        # self.out.bias.grad = torch.sum(e, 0)

        return

    def init(self, method):
        if method == 'zero':
            self.fc1.weight.data.fill_(0)
            # self.fc1.bias.data.fill_(0)
            self.fc2.weight.data.fill_(0)
            # self.fc2.bias.data.fill_(0)
            self.fc3.weight.data.fill_(0)
            # self.fc3.bias.data.fill_(0)
            self.out.weight.data.fill_(0)
            # self.out.bias.data.fill_(0)

    def get_B_loss(self, output):
        y1_hat = self.softmax(torch.matmul(self.z1, torch.t(self.B1)))
        y2_hat = self.softmax(torch.matmul(self.z2, torch.t(self.B2)))
        y3_hat = self.softmax(torch.matmul(self.z3, torch.t(self.B3)))

        y1_loss = torch.sum((output - y1_hat)**2) / 64
        y2_loss = torch.sum((output - y2_hat)**2) / 64
        y3_loss = torch.sum((output - y3_hat)**2) / 64

        return y1_loss, y2_loss, y3_loss