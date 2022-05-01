import torch
import torch.nn as nn

def dfa_backward(net, y_hat, one_hot_target):
    e = y_hat - one_hot_target
    with torch.no_grad():
        output_layer = True
        for module in reversed(net.layer_lst):
            if isinstance(module, nn.Linear):
                if output_layer:
                    module.weight.dfa_grad = torch.matmul(torch.t(e), module.input) / net.args.batch_size
                    if module.bias is not None:
                        module.bias.dfa_grad = torch.sum(e, 0)
                    output_layer = False
                else:
                    dx = torch.matmul(e, module.B) * (1-torch.tanh(module.output) ** 2) # TODO: tanh -> relu?
                    module.weight.dfa_grad = torch.matmul(torch.t(dx), module.input) / net.args.batch_size
                    if module.bias is not None:
                        module.bias.dfa_grad = torch.sum(dx, 0)
    return

def dfa2_backward(net, y_hat, one_hot_target):
    with torch.no_grad():
        e = y_hat - one_hot_target
        do = torch.matmul(e, net.layer_lst[-1].weight) * (1-torch.tanh(net.layer_lst[-3].output) ** 2)
        output_layer = True
        output_layer2 = True
        for module in reversed(net.layer_lst):
            if isinstance(module, nn.Linear):
                if output_layer:
                    module.weight.dfa_grad = torch.matmul(torch.t(e), module.input) / net.args.batch_size
                    if module.bias is not None:
                        module.bias.dfa_grad = torch.sum(e, 0)
                    output_layer = False
                elif output_layer2:
                    module.weight.dfa_grad = torch.matmul(torch.t(do), module.input) / net.args.batch_size
                    if module.bias is not None:
                        module.bias.dfa_grad = torch.sum(do, 0)
                    output_layer2 = False
                else:
                    dx = torch.matmul(do, module.B) * (1-torch.tanh(module.output) ** 2) # TODO: tanh -> relu?
                    module.weight.dfa_grad = torch.matmul(torch.t(dx), module.input) / net.args.batch_size
                    if module.bias is not None:
                        module.bias.dfa_grad = torch.sum(dx, 0)
    return


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


def dfa_grad(net):
    for name, module in net.sequential_layer.named_modules():
        if isinstance(module, nn.Linear):
            module.weight.grad = module.weight.dfa_grad.detach()
            module.weight.dfa_grad = None
            if module.bias is not None:
                    module.bias.grad = module.bias.dfa_grad