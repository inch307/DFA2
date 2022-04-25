import torch
import torch.nn as nn
import sklearn.metrics
import torch.nn.functional as F

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
    criterion = nn.CrossEntropyLoss(reduction='sum')
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
    criterion = nn.CrossEntropyLoss()
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

    with torch.no_grad():
        train_loss = 0
        criterion_train_loss = nn.CrossEntropyLoss(reduction='sum')
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

def train_dfa(net, train_loader, optimizer, device, lr):
    net.train()
    criterion = nn.CrossEntropyLoss()
    losses = []
    correct1 = 0
    correct3 = 0
    correct5 = 0
    train_B_loss_1 = 0
    train_B_loss_2 = 0
    train_B_loss_3 = 0
    n_iter = 0
    for _batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = net(data)
        c1, c3, c5 = topk_correct(output, target, (1, 3, 5))
        correct1 += c1
        correct3 += c3
        correct5 += c5
        loss = criterion(output, target)
        target = one_hot(target, 64, 10)
        net.backward(output-target ,data)

        if n_iter % 30 == 0:
            y1_loss, y2_loss, y3_loss = net.get_B_loss(output)
            print(f'n_iter: {n_iter}, B_loss is: y1: {y1_loss}, y1: {y2_loss}, y1: {y3_loss}')
            

        optimizer.step()
        
        losses.append(loss.item())
        
        if n_iter % 500 == 0:
            acc1_i = c1 / target.size(0) * 100
            acc3_i = c3 / target.size(0) * 100
            acc5_i = c5 / target.size(0) * 100
            print(f'acc1: {acc1_i}, acc3: {acc3_i}, acc5: {acc5_i}, loss: {loss}')
        n_iter += 1
    
    correct1 = 0
    correct3 = 0
    correct5 = 0
    train_B_loss_1 = 0
    train_B_loss_2 = 0
    train_B_loss_3 = 0
    with torch.no_grad():
        for _batch_idx, (data, target) in enumerate(tqdm(train_loader)):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = net(data)
            c1, c3, c5 = topk_correct(output, target, (1, 3, 5))
            correct1 += c1
            correct3 += c3
            correct5 += c5
            loss = criterion(output, target)
            target = one_hot(target, 64, 10)
            train_B_loss = net.get_B_loss(output)
            train_B_loss_1 += train_B_loss[0] * 64 / len(train_loader.dataset)
            train_B_loss_2 += train_B_loss[1] * 64 / len(train_loader.dataset)
            train_B_loss_3 += train_B_loss[2] * 64 / len(train_loader.dataset)
        print(f'train B_loss is: y1: {train_B_loss_1}, y1: {train_B_loss_2}, y1: {train_B_loss_3}')
        acc1 = correct1 / len(train_loader.dataset) * 100
        acc3 = correct3 / len(train_loader.dataset) * 100
        acc5 = correct5 / len(train_loader.dataset) * 100

        print(f'acc1: {acc1}, acc3: {acc3}, acc5: {acc5}')

    
    
def train_dfa2(net, train_loader, optimizer, device, lr, post):
    net.train()
    criterion = nn.CrossEntropyLoss()
    losses = []
    correct1 = 0
    correct3 = 0
    correct5 = 0
    train_B_loss_1 = 0
    train_B_loss_2 = 0
    train_B_loss_3 = 0
    n_iter = 0
    for _batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = net(data)
        c1, c3, c5 = topk_correct(output, target, (1, 3, 5))
        correct1 += c1
        correct3 += c3
        correct5 += c5
        loss = criterion(output, target)
        target = one_hot(target, 64, 10)

        if post:
            if n_iter % 30 == 0:
                y1_loss, y2_loss, y3_loss = net.get_B_loss(output)
                print(f'n_iter: {n_iter}, B_loss is: y1: {y1_loss}, y1: {y2_loss}, y1: {y3_loss}')
            net.backward(output-target ,data)
            # net.update_B(lr, output-target)
            net.new_update_B(lr, output)

        else:
            if n_iter % 30 == 0:
                y1_loss, y2_loss, y3_loss = net.get_B_loss(output)
                print(f'n_iter: {n_iter}, B_loss is: y1: {y1_loss}, y1: {y2_loss}, y1: {y3_loss}')
            net.update_B(lr, output-target)
            # net.new_update_B(lr, output)
            net.backward(output-target ,data)

        optimizer.step()
        
        losses.append(loss.item())
        
        if n_iter % 500 == 0:
            acc1_i = c1 / target.size(0) * 100
            acc3_i = c3 / target.size(0) * 100
            acc5_i = c5 / target.size(0) * 100
            print(f'acc1: {acc1_i}, acc3: {acc3_i}, acc5: {acc5_i}, loss: {loss}')
        n_iter += 1

    correct1 = 0
    correct3 = 0
    correct5 = 0
    train_B_loss_1 = 0
    train_B_loss_2 = 0
    train_B_loss_3 = 0
    with torch.no_grad():
        for _batch_idx, (data, target) in enumerate(tqdm(train_loader)):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = net(data)
            c1, c3, c5 = topk_correct(output, target, (1, 3, 5))
            correct1 += c1
            correct3 += c3
            correct5 += c5
            loss = criterion(output, target)
            target = one_hot(target, 64, 10)
            train_B_loss = net.get_B_loss(output)
            train_B_loss_1 += train_B_loss[0] * 64 / len(train_loader.dataset)
            train_B_loss_2 += train_B_loss[1] * 64 / len(train_loader.dataset)
            train_B_loss_3 += train_B_loss[2] * 64 / len(train_loader.dataset)
        print(f'train B_loss is: y1: {train_B_loss_1}, y1: {train_B_loss_2}, y1: {train_B_loss_3}')
        acc1 = correct1 / len(train_loader.dataset) * 100
        acc3 = correct3 / len(train_loader.dataset) * 100
        acc5 = correct5 / len(train_loader.dataset) * 100

        print(f'acc1: {acc1}, acc3: {acc3}, acc5: {acc5}')

def train_dfa3(net, train_loader, optimizer, device, lr, epochs):
    net.train()
    criterion = nn.CrossEntropyLoss()
    losses = []
    correct1 = 0
    correct3 = 0
    correct5 = 0
    n_iter = 0
    for _batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = net(data)
        c1, c3, c5 = topk_correct(output, target, (1, 3, 5))
        correct1 += c1
        correct3 += c3
        correct5 += c5
        loss = criterion(output, target)
        target = one_hot(target, 64, 10)

        
        if n_iter % 30 == 0:
            y1_loss, y2_loss, y3_loss = net.get_B_loss(output)
            print(f'n_iter: {n_iter}, B_loss is: y1: {y1_loss}, y1: {y2_loss}, y1: {y3_loss}')
        # for i in range(epochs):
        #     net.update_B(lr, output-target)
        for i in range(epochs):
            net.new_update_B(lr, output)
        net.backward(output-target ,data)

        optimizer.step()
        
        losses.append(loss.item())
        
        if n_iter % 500 == 0:
            acc1_i = c1 / target.size(0) * 100
            acc3_i = c3 / target.size(0) * 100
            acc5_i = c5 / target.size(0) * 100
            print(f'acc1: {acc1_i}, acc3: {acc3_i}, acc5: {acc5_i}, loss: {loss}')
        n_iter += 1
    
    correct1 = 0
    correct3 = 0
    correct5 = 0
    train_B_loss_1 = 0
    train_B_loss_2 = 0
    train_B_loss_3 = 0
    with torch.no_grad():
        for _batch_idx, (data, target) in enumerate(tqdm(train_loader)):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = net(data)
            c1, c3, c5 = topk_correct(output, target, (1, 3, 5))
            correct1 += c1
            correct3 += c3
            correct5 += c5
            loss = criterion(output, target)
            target = one_hot(target, 64, 10)
            train_B_loss = net.get_B_loss(output)
            train_B_loss_1 += train_B_loss[0] * 64 / len(train_loader.dataset)
            train_B_loss_2 += train_B_loss[1] * 64 / len(train_loader.dataset)
            train_B_loss_3 += train_B_loss[2] * 64 / len(train_loader.dataset)
        print(f'train B_loss is: y1: {train_B_loss_1}, y1: {train_B_loss_2}, y1: {train_B_loss_3}')
        acc1 = correct1 / len(train_loader.dataset) * 100
        acc3 = correct3 / len(train_loader.dataset) * 100
        acc5 = correct5 / len(train_loader.dataset) * 100

        print(f'acc1: {acc1}, acc3: {acc3}, acc5: {acc5}')