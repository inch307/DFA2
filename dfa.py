import torch
import torch.nn as nn

def dfa_backward(net, y_hat):
    x = x.view(-1, 784)

    dx1 = torch.matmul(e, net.B1) * (1-torch.tanh(net.y1) ** 2)
    # print(dx1.shape)
    net.fc1.weight.grad = torch.matmul(torch.t(dx1), x) / 64 # batch szie
    # net.fc1.bias.grad = torch.sum(dx1, 0)

    dx2 = torch.matmul(e, net.B2) * (1-torch.tanh(net.y2) ** 2)
    net.fc2.weight.grad = torch.matmul(torch.t(dx2), net.z1) / 64 # batch szie
    # net.fc2.bias.grad = torch.sum(dx2, 0)

    dx = torch.matmul(e, net.B_lst[0]) * (1-torch.tanh(net.y_lst[0]**2))
    net.fc_lst[0].weight.grad = torch.matmul(torch.t(dx), net.z2)
    for i in range(20):
        dx = torch.matmul(e, net.B_lst[i+1]) * (1-torch.tanh(net.y_lst[i+1]**2))
        net.fc_lst[i+1].weight.grad = torch.matmul(torch.t(dx), net.z_lst[i])

    dx3 = torch.matmul(e, net.B3) * (1-torch.tanh(net.y3) ** 2)
    net.fc3.weight.grad = torch.matmul(torch.t(dx3), net.z_lst[-1]) / 64 # batch szie
    # net.fc3.bias.grad = torch.sum(dx3, 0)

    net.out.weight.grad = torch.matmul(torch.t(e), net.z3) / 64 # batch szie
    # net.out.bias.grad = torch.sum(e, 0)

    return

def dfa2_backward(net, args):
    pass

def measure_alignment(net, args):
    pass

def get_B_loss(net, output):
    y1_hat = net.softmax(torch.matmul(net.z1, torch.t(net.B1)))
    y2_hat = net.softmax(torch.matmul(net.z2, torch.t(net.B2)))
    y3_hat = net.softmax(torch.matmul(net.z3, torch.t(net.B3)))

    y1_loss = torch.sum((output - y1_hat)**2) / 64
    y2_loss = torch.sum((output - y2_hat)**2) / 64
    y3_loss = torch.sum((output - y3_hat)**2) / 64

    return y1_loss, y2_loss, y3_loss

def weight_init(net, method):
    if method == 'zero':
        net.fc1.weight.data.fill_(0)
        # net.fc1.bias.data.fill_(0)
        net.fc2.weight.data.fill_(0)
        # net.fc2.bias.data.fill_(0)
        net.fc3.weight.data.fill_(0)
        # net.fc3.bias.data.fill_(0)
        net.out.weight.data.fill_(0)
        # net.out.bias.data.fill_(0)


# TODO: appropriate init
def get_projection_matrix(net, hidden):
    B = torch.randn(10, hidden)

    return B

def dfa_grad(net):
    pass