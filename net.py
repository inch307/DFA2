from ssl import HAS_TLSv1_3
import torch.nn as nn
import torch

class Net(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Net, self).__init__()
        self.device = kwargs['device']

        self.fc1 = nn.Linear(in_features=784, out_features=800, bias=True)
        self.fc2 = nn.Linear(800, 400, bias=True)
        self.fc3 = nn.Linear(400, 100, bias=True)
        self.out = nn.Linear(100, 10, bias=True)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.out(x)
        # x = self.softmax(x)

        return x

class DFANet(nn.Module):
    def __init__(self, *args, **kwargs):
        super(DFANet, self).__init__()
        self.device = kwargs['device']

        self.fc1 = nn.Linear(in_features=784, out_features=800, bias=True)
        self.fc2 = nn.Linear(800, 400, bias=True)
        self.fc3 = nn.Linear(400, 100, bias=True)
        self.out = nn.Linear(100, 10, bias=True)

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()

        self.B1 = self.get_projection_matrix(800)
        self.B2 = self.get_projection_matrix(400)
        self.B3 = self.get_projection_matrix(100)

    def get_projection_matrix(self, hidden):
        B = torch.randn(10, hidden)

        return B

    def forward(self, x):
        x = x.view(-1, 784)
        self.y1 = self.fc1(x)
        self.z1 = self.tanh(self.y1)
        self.y2 = self.fc2(self.z1)
        self.z2 = self.tanh(self.y2)
        self.y3 = self.fc3(self.z2)
        self.z3 = self.tanh(self.y3)
        self.y4 = self.out(self.z3)
        self.z4 = self.softmax(self.y4)

        return self.z4

    def backward(self, e, x):
        x = x.view(-1, 784)

        dx1 = torch.matmul(e, self.B1) * (1-torch.tanh(self.y1) ** 2)
        # print(dx1.shape)
        self.fc1.weight.grad = torch.matmul(torch.t(dx1), x) / 64 # batch szie
        # self.fc1.bias.grad = torch.sum(dx1, 0)

        dx2 = torch.matmul(e, self.B2) * (1-torch.tanh(self.y2) ** 2)
        self.fc2.weight.grad = torch.matmul(torch.t(dx2), self.z1) / 64 # batch szie
        # self.fc2.bias.grad = torch.sum(dx2, 0)

        dx3 = torch.matmul(e, self.B3) * (1-torch.tanh(self.y3) ** 2)
        self.fc3.weight.grad = torch.matmul(torch.t(dx3), self.z2) / 64 # batch szie
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

class DFA2Net(nn.Module):
    def __init__(self, *args, **kwargs):
        super(DFA2Net, self).__init__()
        self.device = kwargs['device']

        self.fc1 = nn.Linear(in_features=784, out_features=800, bias=True)
        self.fc2 = nn.Linear(800, 400, bias=True)
        self.fc3 = nn.Linear(400, 100, bias=True)
        self.out = nn.Linear(100, 10, bias=True)

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()

        self.B1 = self.get_projection_matrix(800)
        self.B2 = self.get_projection_matrix(400)
        self.B3 = self.get_projection_matrix(100)

    def get_projection_matrix(self, hidden):
        B = torch.randn(10, hidden)

        return B

    def forward(self, x):
        x = x.view(-1, 784)
        self.y1 = self.fc1(x)
        self.z1 = self.tanh(self.y1)
        self.y2 = self.fc2(self.z1)
        self.z2 = self.tanh(self.y2)
        self.y3 = self.fc3(self.z2)
        self.z3 = self.tanh(self.y3)
        self.y4 = self.out(self.z3)
        self.z4 = self.softmax(self.y4)

        return self.z4

    def update_B(self, lr, e):
        d = torch.t(e * self.z4 * (1 - self.z4))
        self.B1 = self.B1*(1- lr*0.01/64) - lr * torch.matmul(d, self.z1) / 64 # batch szie
        self.B2 = self.B2*(1- lr*0.01/64) - lr * torch.matmul(d, self.z2) / 64
        self.B3 = self.B3*(1- lr*0.01/64) - lr * torch.matmul(d, self.z3) / 64
        
        return

    def new_update_B(self, lr, output):
        y1_hat = self.softmax(torch.matmul(self.z1, torch.t(self.B1)))
        y2_hat = self.softmax(torch.matmul(self.z2, torch.t(self.B2)))
        y3_hat = self.softmax(torch.matmul(self.z3, torch.t(self.B3)))

        e1 = (output - y1_hat)
        e2 = (output - y2_hat)
        e3 = (output - y3_hat)

        d1 = torch.t(e1 * y1_hat * (1 - y1_hat))
        d2 = torch.t(e2 * y2_hat * (1 - y2_hat))
        d3 = torch.t(e3 * y3_hat * (1 - y3_hat))

        self.B1 = self.B1*(1- lr*0.01/64) - lr * torch.matmul(d1, self.z1) / 64 # batch szie
        self.B2 = self.B2*(1- lr*0.01/64) - lr * torch.matmul(d2, self.z2) / 64
        self.B3 = self.B3*(1- lr*0.01/64) - lr * torch.matmul(d3, self.z3) / 64
        
        return

    def backward(self, e, x):
        x = x.view(-1, 784)

        dx1 = torch.matmul(e, self.B1) * (1-torch.tanh(self.y1) ** 2)
        self.fc1.weight.grad = torch.matmul(torch.t(dx1), x) / 64 # batch szie
        # self.fc1.bias.grad = torch.sum(dx1, 0)

        dx2 = torch.matmul(e, self.B2) * (1-torch.tanh(self.y2) ** 2)
        self.fc2.weight.grad = torch.matmul(torch.t(dx2), self.z1) / 64 # batch szie
        # self.fc2.bias.grad = torch.sum(dx2, 0)

        dx3 = torch.matmul(e, self.B3) * (1-torch.tanh(self.y3) ** 2)
        self.fc3.weight.grad = torch.matmul(torch.t(dx3), self.z2) / 64 # batch szie
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