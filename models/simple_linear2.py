import torch
import torch.nn as nn

class SimpleLinear2(nn.Module):
    def __init__(self, *args, **kwargs):
        super(SimpleLinear2, self).__init__()
        self.device = kwargs['device']
        self.args = kwargs['args']

        ## input layer
        if self.args.dataset == 'mnist':
            self.fc1 = nn.Linear(in_features=784, out_features=800, bias=True)
        elif self.args.dataset in 'cifar':
            self.fc1 = nn.Linear(in_features=1024, out_features=800, bias=True)
        self.fc2 = nn.Linear(800, 400, bias=True)
        self.fc_lst = []
        for i in range(30):
            self.fc_lst.append(nn.Linear(400, 400))
        self.fc3 = nn.Linear(400, 100, bias=True)
        self.out = nn.Linear(100, 10, bias=True)

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()

        self.B1 = self.get_projection_matrix(800)
        self.B2 = self.get_projection_matrix(400)
        self.B_lst = []
        for i in range(30):
            self.B_lst.append(self.get_projection_matrix(400))
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
        
        self.y_lst = []
        self.z_lst = []
        self.y_lst.append(self.fc_lst[0](self.z2))
        self.z_lst.append(self.tanh(self.y_lst[0]))
        for i in range(29):
            self.y_lst.append(self.z_lst[i])
            self.z_lst.append(self.y_lst[i+1])

        self.y3 = self.fc3(self.z_lst[-1])
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