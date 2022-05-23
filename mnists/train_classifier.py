import argparse
import repackage
repackage.up()

import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

from mnists.models.classifier import C8SteerableCNN,SO2SteerableCNN
from mnists.models.classifier import CNN
from mnists.models.mnist_ses import MNIST_SES_Scalar, MNIST_SES_V
from mnists.dataloader import get_tensor_dataloaders, TENSOR_DATASETS

import numpy as np

import json

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        # stats
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
        loss, correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))

    return loss.detach().cpu().item(), correct, len(train_loader.dataset)

def test(model, device, test_loader, name=""):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    

    print(f'\nTest set {name}: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.3f}%)')

    return test_loss, correct, len(test_loader.dataset)

def main(args):
    # model and dataloader
    functions = {"CNN" : CNN , \
                  "C8SteerableCNN" :  C8SteerableCNN, \
                  "C8SteerableCNNSmall" :  C8SteerableCNN, \
                  "SO2SteerableCNN" : SO2SteerableCNN, \
                  "SES" : MNIST_SES_Scalar, \
                  "SES_V" : MNIST_SES_V}
    #model = CNN()
    if "Small" in args.model:
        model = functions[args.model](small=True)
    else:
        model = functions[args.model]()

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])

    print(f"\nNumber of trainable parameters: {params}\n")
    
    dl_train, dl_test = get_tensor_dataloaders(args.dataset, args.batch_size)

    # Optimizer
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    # push to device and train
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    results_dict = {'results': {name: [] for name, _ in dl_test.items()}}
    results_dict['results']['train'] = []
    for epoch in range(1, args.epochs + 1):
        train_loss, train_correct, train_len = train(args, model, device, dl_train, optimizer, epoch)
        results_dict['results']['train'].append({'loss': train_loss, 'accuracy': 100 * train_correct/train_len})
        for name, dl in dl_test.items():
            test_loss, test_correct, test_len = test(model, device, dl, name)
            results_dict['results'][name].append({'loss': test_loss, 'accuracy': 100 * test_correct/test_len})
        scheduler.step()

    results_dict['train_len'] = train_len
    results_dict['test_len'] = test_len
    results_dict['dataset'] = args.dataset
    results_dict['num_params'] = float(params)

    with open(f'mnists/results/{args.model}_{args.dataset}.json', 'w') as f:
        json.dump(results_dict, f)
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=TENSOR_DATASETS,
                        help='Provide dataset name.')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--model',type=str, default="CNN", choices=["CNN","C8SteerableCNN","C8SteerableCNNSmall", "SO2SteerableCNN","SES","SES_V"])
    args = parser.parse_args()

    print(args)
    main(args)
