#!/usr/bin/env python3
"""
A testbench for running neural network tests with Second Order Optimizers
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
from copy import deepcopy
import os
import argparse
from pathlib import Path
from time import perf_counter
import json
import math
import hashlib
from types import SimpleNamespace
from typing import Optional, Union, Tuple
import warnings

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import transforms
from torchvision import models

from pytorch_sso import optim_hfcr_newton
from pytorch_sso import quasi_newton
from pytorch_sso import nonlinear_conjugate_gradient as nlcg
from pytorch_sso.line_search_spec import LineSearchSpec
from pytorch_sso.trust_region import BadTrustRegionSpec, TrustRegionSpec

try:
    import fast_mnist
    import fast_cifar10
except AssertionError:
    pass
import torchvision.datasets.mnist as mnist
import torchvision.datasets.cifar as cifar

HASH_LENGTH = 10


def args_to_fname(arg_dict_, ext):
    """
    Converts the argument dictionary into a filename with a hash
    """
    # Don't want the output directory influencing the hash
    # Old one did this as boolean, so let's add that back in
    outdir = deepcopy(arg_dict_["record"])
    arg_dict = deepcopy(arg_dict_)
    print("+" * 80)
    print("\tOutdir:", outdir)
    print("+" * 80)
    print("outdir:", outdir)
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
        print("outdir:", outdir)
        fname = os.path.join(outdir, fname)

    return fname


def get_args():
    """
    Get the arguments to the script for the dataset, optimizers, etc.
    """
    parser = argparse.ArgumentParser(description="PyTorch")
    # Batch Sizes
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

    # Dataset / implicitly model as well
    parser.add_argument(
        "--dataset", type=str, required=True, help="the dataset option: mnist / cifar10"
    )

    # Optimizer
    parser.add_argument(
        "--opt",
        type=str,
        required=True,
        help="the optimizer option: sgd / lbfgs / kn / kn2/ sr1 /sr1d / dfp / dfpi",
    )

    parser.add_argument(
        "--memory",
        type=int,
        default=None,
        help="Size of quasi-newton memory, leave blank for unlimited",
    )

    parser.add_argument(
        "--momentum", type=float, default=0.9, help="momentum for SGD algorithm"
    )

    # Network (if not Resnet)
    parser.add_argument("--hidden", type=int, default=15, help="size of hidden layer")

    # SOO Options
    parser.add_argument(
        "--max_newton", type=int, default=10, help="max number of newton iterations"
    )
    parser.add_argument(
        "--abs_newton_tol",
        type=float,
        default=1.0e-5,
        help="Absolute tolerance for Newton iteration convergence",
    )
    parser.add_argument(
        "--rel_newton_tol",
        type=float,
        default=1.0e-8,
        help="Relative tolerance for Newton iteration convergence",
    )
    parser.add_argument(
        "--max_cr", type=int, default=10, help="max number of conjugate residual"
    )
    parser.add_argument(
        "--cr_tol",
        type=float,
        default=1.0e-3,
        help="tolerance for conjugate residual iteration" "convergence",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.01,
        help="training update weight (default: 0.01)",
    )

    # Line Search Params
    parser.add_argument(
        "--sufficient_decrease",
        type=float,
        default=None,
        help="The Armijo rule coefficient",
    )
    parser.add_argument(
        "--curvature_condition",
        type=float,
        default=None,
        help="The curvature coeff for the Wolfe conditions",
    )
    parser.add_argument(
        "--extrapolation_factor",
        type=float,
        default=None,
        help="The factor of decrease for the line search extrapolation",
    )
    parser.add_argument(
        "--max_searches",
        type=int,
        default=None,
        help="The maximum number of line searches that can be attempted before accepting failure",
    )

    # Trust Region Params
    parser.add_argument(
        "--initial_radius",
        type=float,
        default=None,
        help="the initial radius of the trust region model",
    )
    parser.add_argument(
        "--max_radius",
        type=float,
        default=None,
        help="The maximum radius for the trust region model",
    )
    parser.add_argument(
        "--nabla0",
        type=float,
        default=None,
        help=(
            "The minimum acceptable step size for the trust region model. Must be >=0, but should "
            "be a very small value. Values lower than this will outright reject the step."
        ),
    )
    parser.add_argument(
        "--nabla1",
        type=float,
        default=None,
        help=(
            'The minimum value for the trust region model to be "good enough" and not prompt a '
            "decrease in trust region radius"
        ),
    )
    parser.add_argument(
        "--nabla2",
        type=float,
        default=None,
        help=(
            'The minimum value for the trust region model to be "better than good" and prompt an '
            "increase in trust region radius"
        ),
    )
    parser.add_argument(
        "--shrink_factor",
        type=float,
        default=None,
        help=(
            "The multiplicative factor by which the trust region will be reduced if needed. Must "
            "be in (0.0, 1.0)"
        ),
    )
    parser.add_argument(
        "--growth_factor",
        type=float,
        default=None,
        help=(
            "The multiplicative factor by which the trust region will be grown if needed. Must "
            "be >1.0"
        ),
    )
    parser.add_argument(
        "--max_subproblem_iter",
        type=int,
        default=None,
        help=(
            "The maximum number of iterations the trust region subproblem algorithm may take to "
            "to solve before accepting an imperfect answer"
        ),
    )

    # Training options
    parser.add_argument(
        "--num_epoch", type=int, default=12, help="max number of epochs"
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--read_nn", help="deserialize nn from directory before training"
    )
    parser.add_argument(
        "--write_nn",
        action="store_true",
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
        "--record",
        type=str,
        default=None,
        help="Enables storing results, specifies where",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda",
        help="Device to train on: cpu / cuda (default: cuda)",
    )

    return parser.parse_args()


def load_data(batch_size_train, batch_size_test, dataset, device):
    """
    Get the data based upon the relevant arguments
    """
    if dataset == "mnist":
        root_dir = "../../data/"
        if device == "cuda":
            trainset = fast_mnist.FastMNIST(root=root_dir, train=True, download=True)
            testset = fast_mnist.FastMNIST(root=root_dir, train=False, download=True)
        elif device == "cpu":
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ]
            )
            trainset = mnist.MNIST(
                root=root_dir, train=True, download=True, transform=transform
            )
            testset = mnist.MNIST(
                root=root_dir, train=False, download=True, transform=transform
            )
        else:
            raise Exception("Invalid device requested!")

        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size_train, shuffle=True
        )

        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size_test, shuffle=False
        )

    elif dataset == "cifar10":
        root_dir = "../../data/cifar10/"
        if device == "cuda":
            trainset = fast_cifar10.FastCIFAR10(
                root=root_dir, train=True, download=True, resnet_permute=True
            )
            testset = fast_cifar10.FastCIFAR10(
                root=root_dir, train=False, download=True, resnet_permute=True
            )
        elif device == "cpu":
            trainset = cifar.CIFAR10(
                root=root_dir, train=True, download=True, resnet_permute=True
            )
            testset = cifar.CIFAR10(
                root=root_dir, train=False, download=True, resnet_permute=True
            )
        else:
            raise Exception("Invalid device requested!")

        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size_train, shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size_train, shuffle=False
        )
    else:
        raise ValueError(f"Invalid dataset specified: {dataset}")

    return train_loader, test_loader


class MNISTNet(nn.Module):
    """
    A single hidden layer network for MNIST data
    """

    def __init__(self, hidden):
        super().__init__()
        self.layer1 = nn.Linear(784, hidden)
        self.layer2 = nn.Linear(hidden, 10)

    def forward(self, input_tensor):
        """Do the forward pass for the network (duh)"""
        input_tensor = input_tensor.view(-1, 784)
        input_tensor = self.layer1(input_tensor)
        input_tensor = torch.sigmoid(input_tensor)
        input_tensor = self.layer2(input_tensor)
        output = F.log_softmax(input_tensor, dim=1)

        return output


def get_model_and_loss(args, device):
    """
    Does what it says on the tin.
    """
    if args.dataset in ["mnist"]:
        model = MNISTNet(args.hidden).to(device)
        loss_calc = F.nll_loss
    elif args.dataset in ["cifar10"]:
        model = models.resnet18(pretrained=False, num_classes=10).to(device)
        # model = models.resnet18(pretrained=True)
        # model.fc = nn.Linear(model.fc.in_features, 10)
        model.to(device)
        loss_calc = nn.CrossEntropyLoss()

    if args.read_nn:
        print("Reading: ", args.read_nn)
        model.load_state_dict(torch.load(args.read_nn))

    return model, loss_calc


def get_line_search(args) -> Optional[LineSearchSpec]:
    """
    Get the line search spec from the arguments
    """
    lss = LineSearchSpec(
        extrapolation_factor=args.extrapolation_factor,
        sufficient_decrease=args.sufficient_decrease,
        curvature_constant=args.curvature_condition,
        max_searches=args.max_searches,
    )

    if None in (lss.max_searches, lss.sufficient_decrease):
        print("Using no line search")
        lss = None

    return lss


def get_trust_region(args) -> TrustRegionSpec:
    """
    Get the trust region spec from the arguments
    """
    try:
        # We specifically constrain the other params to narrow the space
        tregion = TrustRegionSpec(
            initial_radius=args.initial_radius,
            max_radius=args.max_radius,
            nabla0=args.nabla0,
            nabla1=args.nabla1,
            nabla2=args.nabla2,
            trust_region_subproblem_iter=args.max_subproblem_iter,
        )
    except BadTrustRegionSpec:
        tregion = None

    return tregion


def get_trust_or_search(args) -> Tuple[LineSearchSpec, TrustRegionSpec]:
    """Return the line search, trust region, or no spec"""
    lss = get_line_search(args)
    trs = get_trust_region(args)
    if lss is None:
        print("Not using Line Search!")
    if trs is None:
        print("Not using trust-region!")
    if lss is not None and trs is not None:
        warnings.warn(
            "Both Line search and trust region specified, defaulting to None!"
        )
        lss = None
        trs = None

    return (lss, trs)


def get_optimizer(args, model):
    """Get the optimizer, configured as desired."""
    lss, trs = get_trust_or_search(args)
    if args.opt == "sgd":
        optimizer = optim.SGD(
            model.parameters(), lr=args.learning_rate, momentum=args.momentum
        )

    elif args.opt == "kn":
        if trs is not None:
            raise ValueError("kn does not support trust region!")
        optimizer = optim_hfcr_newton.HFCR_Newton(
            model.parameters(),
            lr=args.learning_rate,
            max_newton=args.max_newton,
            abs_newton_tol=args.abs_newton_tol,
            rel_newton_tol=args.rel_newton_tol,
            max_cr=args.max_cr,
            cr_tol=args.cr_tol,
            line_search_spec=lss,
        )

    elif args.opt == "bfgs":
        if trs is None:
            optimizer = quasi_newton.BFGS(
                model.parameters(),
                lr=args.learning_rate,
                max_newton=args.max_newton,
                abs_newton_tol=args.abs_newton_tol,
                rel_newton_tol=args.rel_newton_tol,
                max_krylov=args.max_cr,
                krylov_tol=args.cr_tol,
                matrix_free_memory=args.memory,
                line_search=lss,
            )
        else:
            optimizer = quasi_newton.BFGSTrust(
                model.parameters(),
                lr=args.learning_rate,
                max_newton=args.max_newton,
                abs_newton_tol=args.abs_newton_tol,
                rel_newton_tol=args.rel_newton_tol,
                max_krylov=args.max_cr,
                krylov_tol=args.cr_tol,
                matrix_free_memory=args.memory,
                trust_region=trs,
            )

    elif args.opt == "bfgsi":
        if trs is None:
            optimizer = quasi_newton.BFGSInverse(
                model.parameters(),
                lr=args.learning_rate,
                max_newton=args.max_newton,
                abs_newton_tol=args.abs_newton_tol,
                rel_newton_tol=args.rel_newton_tol,
                matrix_free_memory=args.memory,
                line_search=lss,
            )
        else:
            # TODO
            pass

    elif args.opt == "sr1":
        if trs is None:
            optimizer = quasi_newton.SymmetricRankOne(
                model.parameters(),
                lr=args.learning_rate,
                max_newton=args.max_newton,
                abs_newton_tol=args.abs_newton_tol,
                rel_newton_tol=args.rel_newton_tol,
                max_krylov=args.max_cr,
                krylov_tol=args.cr_tol,
                matrix_free_memory=args.memory,
                line_search=lss,
            )
        else:
            optimizer = quasi_newton.SymmetricRankOneTrust(
                model.parameters(),
                lr=args.learning_rate,
                max_newton=args.max_newton,
                abs_newton_tol=args.abs_newton_tol,
                rel_newton_tol=args.rel_newton_tol,
                matrix_free_memory=args.memory,
                trust_region=trs,
            )

    elif args.opt == "sr1d":
        if trs is None:
            optimizer = quasi_newton.SymmetricRankOneInverse(
                model.parameters(),
                lr=args.learning_rate,
                max_newton=args.max_newton,
                abs_newton_tol=args.abs_newton_tol,
                rel_newton_tol=args.rel_newton_tol,
                matrix_free_memory=args.memory,
                line_search=lss,
            )
        else:
            optimizer = quasi_newton.SymmetricRankOneDualTrust(
                model.parameters(),
                lr=args.learning_rate,
                max_newton=args.max_newton,
                abs_newton_tol=args.abs_newton_tol,
                rel_newton_tol=args.rel_newton_tol,
                matrix_free_memory=args.memory,
                trust_region=trs,
            )

    elif args.opt == "dfp":
        if trs is None:
            optimizer = quasi_newton.DavidonFletcherPowell(
                model.parameters(),
                lr=args.learning_rate,
                max_newton=args.max_newton,
                abs_newton_tol=args.abs_newton_tol,
                rel_newton_tol=args.rel_newton_tol,
                max_krylov=args.max_cr,
                krylov_tol=args.cr_tol,
                matrix_free_memory=args.memory,
                line_search=lss,
            )
        else:
            optimizer = quasi_newton.DavidonFletcherPowellTrust(
                model.parameters(),
                lr=args.learning_rate,
                max_newton=args.max_newton,
                abs_newton_tol=args.abs_newton_tol,
                rel_newton_tol=args.rel_newton_tol,
                max_krylov=args.max_cr,
                krylov_tol=args.cr_tol,
                matrix_free_memory=args.memory,
                trust_region=trs,
            )

    elif args.opt == "dfpi":
        if trs is None:
            optimizer = quasi_newton.DavidonFletcherPowellInverse(
                model.parameters(),
                lr=args.learning_rate,
                max_newton=args.max_newton,
                abs_newton_tol=args.abs_newton_tol,
                rel_newton_tol=args.rel_newton_tol,
                matrix_free_memory=args.memory,
                line_search=lss,
            )
        else:
            # TODO
            pass

    elif args.opt == "broy":
        if trs is None:
            optimizer = quasi_newton.Broyden(
                model.parameters(),
                lr=args.learning_rate,
                max_newton=args.max_newton,
                abs_newton_tol=args.abs_newton_tol,
                rel_newton_tol=args.rel_newton_tol,
                max_krylov=args.max_cr,
                krylov_tol=args.cr_tol,
                matrix_free_memory=args.memory,
                line_search=lss,
            )
        else:
            # TODO
            pass
            optimizer = quasi_newton.BroydenTrust(
                model.parameters(),
                lr=args.learning_rate,
                max_newton=args.max_newton,
                abs_newton_tol=args.abs_newton_tol,
                rel_newton_tol=args.rel_newton_tol,
                max_krylov=args.max_cr,
                krylov_tol=args.cr_tol,
                matrix_free_memory=args.memory,
                trust_region=trs,
            )

    elif args.opt == "broyi":
        if trs is None:
            optimizer = quasi_newton.BrodyenInverse(
                model.parameters(),
                lr=args.learning_rate,
                max_newton=args.max_newton,
                abs_newton_tol=args.abs_newton_tol,
                rel_newton_tol=args.rel_newton_tol,
                matrix_free_memory=args.memory,
                line_search=lss,
            )
        else:
            # TODO
            pass

    elif args.opt == "fr":
        if trs is not None:
            warnings.warn(
                "NLCG methods don't support Trust Regions, defaulting to no line search!"
            )
        optimizer = nlcg.FletcherReeves(
            model.parameters(),
            lr=args.learning_rate,
            max_newton=args.max_newton,
            abs_newton_tol=args.abs_newton_tol,
            rel_newton_tol=args.rel_newton_tol,
            line_search_spec=lss,
        )

    elif args.opt == "pr":
        if trs is not None:
            warnings.warn(
                "NLCG methods don't support Trust Regions, defaulting to no line search!"
            )
        optimizer = nlcg.PolakRibiere(
            model.parameters(),
            lr=args.learning_rate,
            max_newton=args.max_newton,
            abs_newton_tol=args.abs_newton_tol,
            rel_newton_tol=args.rel_newton_tol,
            line_search_spec=lss,
        )

    elif args.opt == "hs":
        if trs is not None:
            warnings.warn(
                "NLCG methods don't support Trust Regions, defaulting to no line search!"
            )
        optimizer = nlcg.HestenesStiefel(
            model.parameters(),
            lr=args.learning_rate,
            max_newton=args.max_newton,
            abs_newton_tol=args.abs_newton_tol,
            rel_newton_tol=args.rel_newton_tol,
            line_search_spec=lss,
        )

    elif args.opt == "dy":
        if trs is not None:
            warnings.warn(
                "NLCG methods don't support Trust Regions, defaulting to no line search!"
            )
        optimizer = nlcg.DaiYuan(
            model.parameters(),
            lr=args.learning_rate,
            max_newton=args.max_newton,
            abs_newton_tol=args.abs_newton_tol,
            rel_newton_tol=args.rel_newton_tol,
            line_search_spec=lss,
        )

    else:
        valid_opts = (
            "sgd",
            "lbfgs",
            "kn",
            "sr1",
            "sr1d",
            "dfp",
            "dfpi",
            "bfgs",
            "bfgsi",
            "broy",
            "broyi",
            "fr",
            "pr",
            "hs",
            "dy",
        )
        raise ValueError(
            f"Invalid optimizer specified: {args.opt}, must be one of {valid_opts}"
        )

    print("Optimizer:", type(optimizer))
    return optimizer


def train(args, model, device, train_loader, optimizer, epoch, loss_calc):
    """Perform a training epoch using a first order optimizer"""
    model.to(device)
    model.train()
    correct = 0
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.dataset in ["mnist"]:
            # one_hot = torch.zeros((target.size(0), 10), device=device)
            # target = one_hot.scatter_(1, target.unsqueeze(1), 1)
            pass
        optimizer.zero_grad()
        output = model(data)
        loss = loss_calc(output, target)
        train_loss += loss.item()
        if math.isnan(train_loss):
            accuracy = float("nan")
            break

        _, pred = torch.max(output, 1)

        correct += pred.eq(target).sum().item()

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
    print(
        "\nTrain set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)".format(
            train_loss,
            correct,
            len(train_loader.dataset),
            100.0 * correct / len(train_loader.dataset),
        )
    )
    accuracy = 100.0 * correct / len(train_loader.dataset)

    return accuracy, train_loss


def train_sso(args, model, device, train_loader, optimizer, epoch, loss_calc):
    """Perform a training epoch using a Second Order Optimizer"""
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
            try:
                loss.backward()
            except RuntimeError:
                # we're in a scope that has disabled grad, "probably" on purpose
                pass
            return loss

        loss = optimizer.step(closure)
        train_loss += loss.item()
        if math.isnan(train_loss):
            break

        with torch.no_grad():
            output = model(data)
            _, pred = torch.max(output, 1)
            correct += pred.eq(target).sum().item()

        accuracy = 100.0 * correct / len(train_loader.dataset)

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
    """Evaluate the model using a test set."""
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
        "Test set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )
    accuracy = 100.0 * correct / len(test_loader.dataset)

    return accuracy, test_loss


def _main(
    opt: str,
    dataset: str,
    batch_size_train: int = 60000,
    batch_size_test: int = 1000,
    momentum: float = 0.9,
    hidden: int = 15,
    max_newton: int = 10,
    abs_newton_tol: float = 1e-5,
    rel_newton_tol: float = 1e-8,
    max_cr: int = 10,
    cr_tol: float = 1e-3,
    learning_rate: float = 0.01,
    sufficient_decrease: Optional[float] = None,
    curvature_condition: Optional[float] = None,
    extrapolation_factor: Optional[float] = None,
    max_searches: Optional[float] = None,
    initial_radius: Optional[float] = None,
    max_radius: Optional[float] = None,
    nabla0: Optional[float] = None,
    nabla1: Optional[float] = None,
    nabla2: Optional[float] = None,
    shrink_factor: Optional[float] = None,
    growth_factor: Optional[float] = None,
    max_subproblem_iter: Optional[int] = None,
    num_epoch: int = 12,
    seed: int = 1,
    read_nn: Optional[str] = None,
    write_nn: bool = True,
    log_interval: int = 10,
    device: str = "cuda",
    record: Optional[Union[Path, str]] = None,
    memory: Optional[int] = None,
):
    """Do the things."""
    args = SimpleNamespace(
        opt=opt,
        dataset=dataset,
        batch_size_train=batch_size_train,
        batch_size_test=batch_size_test,
        momentum=momentum,
        hidden=hidden,
        max_newton=max_newton,
        abs_newton_tol=abs_newton_tol,
        rel_newton_tol=rel_newton_tol,
        max_cr=max_cr,
        cr_tol=cr_tol,
        learning_rate=learning_rate,
        sufficient_decrease=sufficient_decrease,
        curvature_condition=curvature_condition,
        extrapolation_factor=extrapolation_factor,
        max_searches=max_searches,
        initial_radius=initial_radius,
        max_radius=max_radius,
        nabla0=nabla0,
        nabla1=nabla1,
        nabla2=nabla2,
        shrink_factor=shrink_factor,
        growth_factor=growth_factor,
        max_subproblem_iter=max_subproblem_iter,
        num_epoch=num_epoch,
        seed=seed,
        read_nn=read_nn,
        write_nn=write_nn,
        log_interval=log_interval,
        device=device,
        record=record,
        memory=memory,
    )
    if os.path.isfile(args_to_fname(vars(args), "json")):
        print("File already exists, not re-running experiment!")
        return

    torch.manual_seed(args.seed)
    device_name = args.device

    device = torch.device(device_name)

    model, loss_calc = get_model_and_loss(args, device)

    train_loader, test_loader = load_data(
        args.batch_size_train, args.batch_size_test, args.dataset, device_name
    )

    optimizer = get_optimizer(args, model)

    times = []
    train_loss_list = []
    test_loss_list = []

    train_accuracy = []
    test_accuracy = []

    first_order_optimizers = ("sgd",)
    second_order_optimizers = (
        "lbfgs",
        "kn",
        "kn2",
        "sr1",
        "sr1d",
        "bfgs",
        "bfgsi",
        "dfp",
        "dfpi",
        "broy",
        "broyi",
        "fr",
        "pr",
        "hs",
        "dy",
    )

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
        times.append(t_elaps)
        train_accuracy.append(train_acc)
        train_loss_list.append(train_loss)
        test_acc, test_loss = test(args, model, device, test_loader, loss_calc)
        test_accuracy.append(test_acc)
        test_loss_list.append(test_loss)
        if math.isnan(train_loss):
            print("=" * 80)
            print(f"NaN training loss, its borked at epoch {epoch}!")
            print("=" * 80)
            break

    total_time = sum(times)
    print("The train loss list is: ", train_loss_list)
    print("The average test loss list is: ", test_loss_list)
    print("The train accuracy is: ", train_accuracy)
    print("The test accuracy is: ", test_accuracy)
    print("The time list is: ", times)
    print("The total training time is: ", total_time)

    if args.write_nn:
        fname = args_to_fname(vars(args), "pkl")
        torch.save(model.state_dict(), fname)

    if args.record is not None:
        rslts = {}
        rslts["specs"] = vars(args)
        rslts["time"] = times
        rslts["train_loss_list"] = train_loss_list
        rslts["test_loss_list"] = test_loss_list
        rslts["test_accuracy_list"] = test_accuracy
        rslts["train_accuracy_list"] = train_accuracy

        fname = args_to_fname(vars(args), "json")

        with open(fname, "w", encoding="UTF-8") as outfile:
            json.dump(rslts, outfile)


def main(
    opt: str = "sgd",
    dataset: str = "cifar10",
    batch_size_train: int = 60000,
    batch_size_test: int = 1000,
    momentum: float = 0.9,
    hidden: int = 15,
    max_newton: int = 10,
    abs_newton_tol: float = 1e-5,
    rel_newton_tol: float = 1e-8,
    max_cr: int = 10,
    cr_tol: float = 1e-3,
    learning_rate: float = 0.01,
    sufficient_decrease: Optional[float] = None,
    curvature_condition: Optional[float] = None,
    extrapolation_factor: Optional[float] = None,
    max_searches: Optional[float] = None,
    initial_radius: Optional[float] = None,
    max_radius: Optional[float] = None,
    nabla0: Optional[float] = None,
    nabla1: Optional[float] = None,
    nabla2: Optional[float] = None,
    shrink_factor: Optional[float] = None,
    growth_factor: Optional[float] = None,
    max_subproblem_iter: Optional[int] = None,
    num_epoch: int = 12,
    seed: int = 1,
    read_nn: Optional[str] = None,
    write_nn: bool = True,
    log_interval: int = 10,
    device: str = "cuda",
    record: Optional[Union[Path, str]] = None,
    memory: Optional[int] = None,
):
    """Wrapper for internal main to catch a CUDA runtime error"""
    try:
        _main(
            opt,
            dataset,
            batch_size_train,
            batch_size_test,
            momentum,
            hidden,
            max_newton,
            abs_newton_tol,
            rel_newton_tol,
            max_cr,
            cr_tol,
            learning_rate,
            sufficient_decrease,
            curvature_condition,
            extrapolation_factor,
            max_searches,
            initial_radius,
            max_radius,
            nabla0,
            nabla1,
            nabla2,
            shrink_factor,
            growth_factor,
            max_subproblem_iter,
            num_epoch,
            seed,
            read_nn,
            write_nn,
            log_interval,
            device,
            record,
            memory,
        )
    except RuntimeError as runtime_error:
        if "CUDA out of memory" in runtime_error.args[0]:
            print("=" * 80)
            print("Need more vram!")
            arg_dict = dict(
                opt=opt,
                dataset=dataset,
                batch_size_train=batch_size_train,
                batch_size_test=batch_size_test,
                momentum=momentum,
                hidden=hidden,
                max_newton=max_newton,
                abs_newton_tol=abs_newton_tol,
                rel_newton_tol=rel_newton_tol,
                max_cr=max_cr,
                cr_tol=cr_tol,
                learning_rate=learning_rate,
                sufficient_decrease=sufficient_decrease,
                curvature_condtion=curvature_condition,
                extrapolation_factor=extrapolation_factor,
                max_searches=max_searches,
                initial_radius=initial_radius,
                max_radius=max_radius,
                nabla0=nabla0,
                nabla1=nabla1,
                nabla2=nabla2,
                shrink_factor=shrink_factor,
                growth_factor=growth_factor,
                max_subproblem_iter=max_subproblem_iter,
                num_epoch=num_epoch,
                seed=seed,
                read_nn=read_nn,
                write_nn=write_nn,
                log_interval=log_interval,
                device=device,
                record=record,
                memory=memory,
            )
            print(args_to_fname(arg_dict, ""))
            print("=" * 80)
        else:
            raise runtime_error


if __name__ == "__main__":
    args_ = get_args()
    main(
        opt=args_.opt,
        dataset=args_.dataset,
        batch_size_train=args_.batch_size_train,
        batch_size_test=args_.batch_size_test,
        momentum=args_.momentum,
        hidden=args_.hidden,
        max_newton=args_.max_newton,
        abs_newton_tol=args_.abs_newton_tol,
        rel_newton_tol=args_.rel_newton_tol,
        max_cr=args_.max_cr,
        cr_tol=args_.cr_tol,
        learning_rate=args_.learning_rate,
        sufficient_decrease=args_.sufficient_decrease,
        curvature_condition=args_.curvature_condition,
        extrapolation_factor=args_.extrapolation_factor,
        max_searches=args_.max_searches,
        initial_radius=args_.initial_radius,
        max_radius=args_.max_radius,
        nabla0=args_.nabla0,
        nabla1=args_.nabla1,
        nabla2=args_.nabla2,
        shrink_factor=args_.shrink_factor,
        growth_factor=args_.growth_factor,
        max_subproblem_iter=args_.max_subproblem_iter,
        num_epoch=args_.num_epoch,
        seed=args_.seed,
        read_nn=args_.read_nn,
        write_nn=args_.write_nn,
        log_interval=args_.log_interval,
        device=args_.device,
        record=args_.record,
        memory=args_.memory,
    )
