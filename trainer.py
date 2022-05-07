import torch
import torch.nn as nn
import sklearn.metrics
import torch.nn.functional as F
import dfa

def one_hot(target, batch_size, output_size):
    r = []
    for i in range(batch_size):
        lst = [0] * output_size
        lst[target[i]] = 1
        r.append(lst)

    return torch.tensor(r)


def topk_correct(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        correct = correct.contiguous()

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k)
        return res

def val(net, val_loader, device, args):
    net.eval()
    criterion = nn.NLLLoss(reduction='sum')
    test_loss = 0
    test_B_loss = 0
    test_B_acc = [0, 0, 0]
    accuracy = [0, 0, 0]
    label = []
    pred_lst = []
    
    with torch.no_grad():
        for _batch_idx, (data, target) in enumerate(val_loader):
            data, target = data.to(device), target.to(device)
            if args.net in ['simple_linear1', 'simple_linear2']:
                data = data.view(args.val_batch_size, -1)
            output = net(data)
            test_loss += criterion(output, target).item()
            c1, c3, c5 = topk_correct(output, target, (1, 3, 5))
            accuracy[0] += c1
            accuracy[1] += c3
            accuracy[2] += c5

            pred = F.softmax(output, dim=1)
            pred_lst.append(pred)
            label.append(target)

            # if args.model in ['dfa', 'dfa2']:
            #     test_B_loss = net.get_B_loss(output)
            #     test_B_loss_1 += test_B_loss[0] * 64 / len(val_loader.dataset)
            #     test_B_loss_2 += test_B_loss[1] * 64 / len(val_loader.dataset)
            #     test_B_loss_3 += test_B_loss[2] * 64 / len(val_loader.dataset)

    test_loss = test_loss / len(val_loader.dataset)

    pred = torch.cat(pred_lst, dim=0).to(torch.device('cpu'))
    label = torch.cat(label, dim=0).to(torch.device('cpu'))

    auroc = sklearn.metrics.roc_auc_score(label, pred, multi_class='ovr')
            
    accuracy[0] = accuracy[0] / len(val_loader.dataset) * 100
    accuracy[1] = accuracy[1] / len(val_loader.dataset) * 100
    accuracy[2] = accuracy[2] / len(val_loader.dataset) * 100

    print(f'validation, test_loss: {test_loss},  acc1: {accuracy[0]}, acc3: {accuracy[1]}, acc5: {accuracy[2]}, auroc: {auroc}')
    # print(f'validation, acc1: {acc1}, acc3: {acc3}, acc5: {acc5}, test_B_loss_y1: {test_B_loss_1}, test_B_loss_y2: {test_B_loss_2}, test_B_loss_y3: {test_B_loss_3}')

def train_backprop(net, train_loader, optimizer, device, args):
    net.train()
    criterion = nn.NLLLoss()
    losses = []
    n_iter = 0
    
    for _batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        if args.net in ['simple_linear1', 'simple_linear2']:
            data = data.view(args.batch_size, -1)
        optimizer.zero_grad()
        output = net(data)
        loss = criterion(output, target)
        loss.backward()      
        optimizer.step()
    
        
    # train accuracy and loss
    net.eval()
    with torch.no_grad():
        train_loss = 0
        criterion_train_loss = nn.NLLLoss(reduction='sum')
        accuracy = [0, 0, 0]
        label = []
        pred_lst = []
        for _batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            if args.net in ['simple_linear1', 'simple_linear2']:
                data = data.view(args.batch_size, -1)
            output = net(data)
            c1, c3, c5 = topk_correct(output, target, (1, 3, 5))
            accuracy[0] += c1
            accuracy[1] += c3
            accuracy[2] += c5
            loss = criterion_train_loss(output, target)

            pred = F.softmax(output, dim=1)
            pred_lst.append(pred)
            label.append(target)

            train_loss += loss.item()

        pred = torch.cat(pred_lst, dim=0).to(torch.device('cpu'))
        label = torch.cat(label, dim=0).to(torch.device('cpu'))

        auroc = sklearn.metrics.roc_auc_score(label, pred, multi_class='ovr')

        train_loss = train_loss / len(train_loader.dataset)

        accuracy[0] = accuracy[0] / len(train_loader.dataset) * 100
        accuracy[1] = accuracy[1] / len(train_loader.dataset) * 100
        accuracy[2] = accuracy[2] / len(train_loader.dataset) * 100

    print(f'train_loss: {train_loss}, acc1: {accuracy[0]}, acc3: {accuracy[1]}, acc5: {accuracy[2]}, auroc: {auroc}')

           
    return train_loss, accuracy

def train_dfa(net, train_loader, optimizer, device, args):
    net.train()
    criterion = nn.NLLLoss() 
    losses = []
    n_iter = 0

    if args.dataset == 'mnist' or args.dataset == 'cifar10' or args.dataset == 'stl10':
        output_size = 10
    elif args.dataset == 'cifar100':
        output_size = 100
    elif args.dataset == 'imagenet':
        output_size = 1000


    # if experiment, calculate alignment
    if args.experiment:
        for _batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            if args.net in ['simple_linear1', 'simple_linear2']:
                data = data.view(args.batch_size, -1)
            # backprop backward
            optimizer.zero_grad()
            output = net(data)
            loss = criterion(output, target)
            loss.backward()

            # loss = net.get_B_loss() TODO

            with torch.no_grad():
                one_hot_target = one_hot(target, args.batch_size, output_size)

                # dfa backward
                y_hat = net.out.y_hat
                e = y_hat - one_hot_target
                idx_lst = [i for i in range(len(net))]
                print(idx_lst)
                if args.model == 'dfa':
                    net.dfa_backward(e, idx_lst)
                elif args.model == 'dfa2':
                    dfa.dfa2_backward(net, y_hat, one_hot_target)
                # align = dfa.measure_alignment(net) TODO
                optimizer.zero_grad()
                net.dfa_grad(idx_lst)
                optimizer.step()

    else:
        with torch.no_grad():
            for _batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                if args.net in ['simple_linear1', 'simple_linear2']:
                    data = data.view(args.batch_size, -1)
                optimizer.zero_grad()
                output = net(data)
                loss = criterion(output, target)

                # loss = net.get_B_loss() TODO

                
                one_hot_target = one_hot(target, args.batch_size, output_size).to(device)

                # dfa backward
                y_hat = net.out.y_hat
                e = y_hat - one_hot_target
                idx_lst = [i for i in range(len(net))]
                if args.model == 'dfa':
                    net.dfa_backward(e, idx_lst)
                elif args.model == 'dfa2':
                    dfa.dfa2_backward(net, y_hat, one_hot_target)
                net.dfa_grad(idx_lst)
                optimizer.step()


    # train accuracy and loss
    net.eval()
    with torch.no_grad():
        train_loss = 0
        criterion_train_loss = nn.NLLLoss(reduction='sum')
        accuracy = [0, 0, 0]
        label = []
        pred_lst = []
        for _batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            if args.net in ['simple_linear1', 'simple_linear2']:
                data = data.view(args.batch_size, -1)
            output = net(data)
            c1, c3, c5 = topk_correct(output, target, (1, 3, 5))
            accuracy[0] += c1
            accuracy[1] += c3
            accuracy[2] += c5
            loss = criterion_train_loss(output, target)

            pred = F.softmax(output, dim=1)
            pred_lst.append(pred)
            label.append(target)

            train_loss += loss.item()

        pred = torch.cat(pred_lst, dim=0).to(torch.device('cpu'))
        label = torch.cat(label, dim=0).to(torch.device('cpu'))

        auroc = sklearn.metrics.roc_auc_score(label, pred, multi_class='ovr')

        train_loss = train_loss / len(train_loader.dataset)

        accuracy[0] = accuracy[0] / len(train_loader.dataset) * 100
        accuracy[1] = accuracy[1] / len(train_loader.dataset) * 100
        accuracy[2] = accuracy[2] / len(train_loader.dataset) * 100

    print(f'train_loss: {train_loss}, acc1: {accuracy[0]}, acc3: {accuracy[1]}, acc5: {accuracy[2]}, auroc: {auroc}')

           
    return train_loss, accuracy