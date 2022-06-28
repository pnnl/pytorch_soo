#!/usr/bin/env python3
"""
Copyright 2021 Nanmiao Wu, Eric Silk

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute,
sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT
OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os
import json
import hashlib
import math
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Union
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
from time import perf_counter
from experiments.nn import HASH_LENGTH

from pytorch_sso import mfcr_levenberg
import fast_mnist
import fast_cifar10

HASH_LENGTH = 10


def args_to_fname(arg_dict_, ext):
    # Don't want the output directory influencing the hash
    # Old one did this as boolean, so let's add that back in
    outdir = deepcopy(arg_dict_["record"])
    arg_dict = deepcopy(arg_dict_)
    del arg_dict["record"]
    arg_dict["record"] = outdir is not None
    sorted_args = sorted(arg_dict.items())
    sorted_arg_values = [i[1] for i in sorted_args]
    num_args = len(sorted_arg_values)

    hasher = hashlib.sha256()

    fname = "TEST" + ("_{}" * num_args)
    fname = fname.format(*sorted_arg_values)
    hasher.update(bytes(fname, "UTF-8"))
    digest = hasher.hexdigest()[:HASH_LENGTH]
    fname = f"TEST_{arg_dict['opt']}_{digest}.{ext}"
    if outdir is not None:
        fname = os.path.join(outdir, fname)

    return fname


def get_args():
    parser = argparse.ArgumentParser(description="PyTorch")
    parser.add_argument(
        "--batch_size_train",
        type=int,
        default=60000,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--batch_size_test",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--opt", type=str, required=True, help="the optimizer option: sgd / levenberg"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        default="mnist",
        help="the dataset option: mnist / cifar10",
    )
    parser.add_argument("--hidden", type=int, default=15, help="size of hidden layer")
    parser.add_argument(
        "--max_cr", type=int, default=10, help="max number of conjugate residual"
    )
    parser.add_argument(
        "--max_newton", type=int, default=10, help="max number of newton iterations"
    )
    parser.add_argument(
        "--cr_tol",
        type=int,
        default=1.0e-3,
        help="tolerance for conjugate residual iteration" "convergence",
    )
    parser.add_argument(
        "--newton_tol",
        type=int,
        default=1.0e-3,
        help="tolerance for Newton iteration convergence",
    )
    parser.add_argument(
        "--num_epoch", type=int, default=12, help="max number of epochs"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.3333, help="training update weight"
    )
    parser.add_argument(
        "--momentum", type=float, default=0.9, help="momentum for SGD algorithm"
    )
    parser.add_argument(
        "--lambda0",
        type=float,
        default=1.0,
        help="The initial value of lambda for the Levenberg/Levenberg-Marquardt algorithms",
    )
    parser.add_argument(
        "--nu",
        type=float,
        default=(2**0.5),
        help="The lambda scaler for the Levenberg/Levenberg-Marquardt algorithms",
    )
    parser.add_argument(
        "--max_lambda",
        type=float,
        default=1000,
        help="The maximum value of lambda for the Levenberg/Levenberg-Marquardt algorithms",
    )
    parser.add_argument(
        "--min_lambda",
        type=float,
        default=1e-6,
        help="The minimum value of lambda for the Levenberg/Levenberg-Marquardt algorithms",
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--read_nn", help="deserialize nn from directory before training"
    )
    parser.add_argument(
        "--write_nn",
        type=str,
        default=None,
        help="serialize nn to directory after training",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training" "status",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda",
        help="Device to train on: cpu / cuda (default: cuda)",
    )

    parser.add_argument(
        "--record",
        type=str,
        default=None,
        help="Enables storing results, specifies where",
    )

    return parser.parse_args()


def load_data(batch_size_train, batch_size_test, dataset):
    if dataset == "mnist":
        root_dir = "../../data/"

        trainset = fast_mnist.FastMNIST(root=root_dir, train=True, download=True)
        testset = fast_mnist.FastMNIST(root=root_dir, train=False, download=True)

    elif dataset == "cifar10":
        root_dir = "../../data/"
        trainset = fast_cifar10.FastCIFAR10(
            root=root_dir, resnet_permute=True, train=True, download=True
        )
        testset = fast_cifar10.FastCIFAR10(
            root=root_dir, resnet_permute=True, train=False, download=True
        )

    else:
        raise ValueError("Invalid dataset specified: {}".format(dataset))

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size_train, shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size_test, shuffle=False
    )

    return train_loader, test_loader


class MNISTNet(nn.Module):
    def __init__(self, hidden):
        super(MNISTNet, self).__init__()
        self.l1 = nn.Linear(784, hidden)
        self.l2 = nn.Linear(hidden, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.l1(x)
        x = torch.sigmoid(x)
        x = self.l2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch, loss_calc):
    model.to(device)
    model.train()
    correct = 0
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.dataset in ["mnist"]:
            one_hot = torch.zeros((target.size(0), 10), device=device)
            target = one_hot.scatter_(1, target.unsqueeze(1), 1)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_calc(output, target)
        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        if args.dataset == "cifar10":
            target = torch.unsqueeze(target, -1)
        correct += pred.eq(target.argmax(dim=1, keepdim=True)).sum().item()

        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
    accuracy = 100.0 * correct / len(train_loader.dataset)
    return accuracy, train_loss


def train_sso(args, model, device, train_loader, optimizer, epoch, loss_calc):
    model.to(device)
    model.train()
    train_loss = 0.0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):

        def closure():
            nonlocal data
            nonlocal target
            optimizer.zero_grad()
            output = model(data)
            loss = loss_calc(output, target)
            loss.backward()
            return loss

        loss = optimizer.step(closure)
        train_loss += loss.item()
        if math.isnan(train_loss):
            break

        with torch.no_grad():
            output = model(data)
            _, pred = torch.max(output, 1)
            correct += pred.eq(target).sum().item()

        if batch_idx % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )

    accuracy = 100.0 * correct / len(train_loader.dataset)
    print(
        "\nTrain set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)".format(
            train_loss,
            correct,
            len(train_loader.dataset),
            100.0 * correct / len(train_loader.dataset),
        )
    )

    return accuracy, train_loss


def test(args, model, device, test_loader, loss_calc):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_calc(output, target).item()
            _, pred = torch.max(output, 1)
            correct += pred.eq(target).sum().item()

    test_loss /= len(test_loader.dataset)
    print(
        "\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )
    accuracy = 100.0 * correct / len(test_loader.dataset)

    return accuracy, test_loss


def main(
    opt: str = "levenberg",
    dataset: str = "cifar10",
    batch_size_train: int = 50000,
    batch_size_test: int = 10000,
    momentum: float = 0.0,
    hidden: int = 15,
    max_newton: int = 10,
    newton_tol: float = 1e-5,
    max_cr: int = 10,
    cr_tol: float = 1e-5,
    learning_rate: float = 1.0,
    lambda0: float = 1.0,
    nu: float = 2**0.5,
    max_lambda: float = 1e3,
    min_lambda: float = 1e-3,
    num_epoch: int = 12,
    seed: int = 1,
    read_nn: Optional[str] = None,
    write_nn: bool = False,
    log_interval: int = 10,
    device: str = "cuda",
    record: Optional[Union[Path, str]] = None,
):
    args = SimpleNamespace(
        opt=opt,
        dataset=dataset,
        batch_size_train=batch_size_train,
        batch_size_test=batch_size_test,
        hidden=hidden,
        max_newton=max_newton,
        newton_tol=newton_tol,
        max_cr=max_cr,
        cr_tol=cr_tol,
        momentum=momentum,
        learning_rate=learning_rate,
        lambda0=lambda0,
        nu=nu,
        max_lambda=max_lambda,
        min_lambda=min_lambda,
        num_epoch=num_epoch,
        seed=seed,
        read_nn=read_nn,
        write_nn=write_nn,
        log_interval=log_interval,
        device=device,
        record=record,
    )
    fname = args_to_fname(vars(args), "json")
    if os.path.exists(fname):
        print("Expt has already been ran, skipping...")
        return

    torch.manual_seed(args.seed)
    device_name = args.device

    device = torch.device(device_name)

    train_loader, test_loader = load_data(
        args.batch_size_train, args.batch_size_test, args.dataset
    )

    if args.dataset in ["mnist"]:
        model = MNISTNet(args.hidden).to(device)
        loss_calc = F.nll_loss
    elif args.dataset in ["cifar10"]:
        model = models.resnet18(pretrained=False, num_classes=10).to(device)
        loss_calc = nn.CrossEntropyLoss()

    if args.read_nn:
        print("Reading: ", args.read_nn)
        model.load_state_dict(torch.load(args.read_nn))

    if args.opt == "sgd":
        optimizer = optim.SGD(
            model.parameters(), lr=args.learning_rate, momentum=args.momentum
        )
    elif args.opt == "levenberg":
        optimizer = mfcr_levenberg.LevenbergEveryStep(
            model.parameters(),
            lr=args.learning_rate,
            max_newton=args.max_newton,
            newton_tol=args.newton_tol,
            max_cr=args.max_cr,
            cr_tol=args.cr_tol,
            lambda0=args.lambda0,
            max_lambda=args.max_lambda,
            min_lambda=args.min_lambda,
            nu=args.nu,
            debug=False,
        )
    else:
        raise ValueError(
            "Invalid optimizer specified: {}, must be one of {}".format(
                args.opt, ("sgd", "levenberg")
            )
        )

    time = []
    train_loss_list = []
    test_loss_list = []

    train_acc_list = []
    test_accuracy = []

    first_order_optimizers = ("sgd",)
    second_order_optimizers = ("levenberg",)

    for epoch in range(1, args.num_epoch + 1):
        if args.opt in first_order_optimizers:
            t_start = perf_counter()
            train_acc, train_loss = train(
                args, model, device, train_loader, optimizer, epoch, loss_calc
            )
            t_stop = perf_counter()

        elif args.opt in second_order_optimizers:
            t_start = perf_counter()
            train_acc, train_loss = train_sso(
                args, model, device, train_loader, optimizer, epoch, loss_calc
            )
            t_stop = perf_counter()

        else:
            raise ValueError(f'Invalid optimizer "{args.opt}" requested!')

        t_elaps = t_stop - t_start
        time.append(t_elaps)
        train_acc_list.append(train_acc)
        train_loss_list.append(train_loss)
        test_acc, test_loss = test(args, model, device, test_loader, loss_calc)
        test_accuracy.append(test_acc)
        test_loss_list.append(test_loss)

    total_time = sum(time)
    print("The train loss list is: ", train_loss_list, "\n")
    print("The average test loss list is: ", test_loss_list, "\n")

    print("The train accuracy is: ", train_acc_list, "\n")
    print("The test accuracy is: ", test_accuracy, "\n")
    print("The time list is: ", time, "\n")
    print("The total training time is: ", total_time, "\n")

    if args.write_nn is not None:
        torch.save(model.state_dict(), args.write_nn)

    if args.record is not None:
        rslts = {}
        rslts["specs"] = vars(args)
        rslts["time"] = time
        rslts["train_loss_list"] = train_loss_list
        rslts["test_loss_list"] = test_loss_list
        rslts["train_accuracy_list"] = train_acc_list
        rslts["test_accuracy_list"] = test_accuracy

        fname = args_to_fname(vars(args), "json")

        with open(fname, "w", encoding="UTF-8") as outfile:
            json.dump(rslts, outfile)


if __name__ == "__main__":
    args = get_args()
    main(**vars(args))
