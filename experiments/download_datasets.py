#!/usr/bin/env python3
import shutil
import fast_mnist
import fast_cifar10

DATA_DIR = "../../data"


def main():
    shutil.rmtree(DATA_DIR, ignore_errors=True)

    mnist_train = fast_mnist.FastMNIST(root=DATA_DIR, train=True, download=True)
    mnist_test = fast_mnist.FastMNIST(root=DATA_DIR, train=False, download=True)

    cifar_train = fast_cifar10.FastCIFAR10(root=DATA_DIR, train=True, download=True)
    cifar_test = fast_cifar10.FastCIFAR10(root=DATA_DIR, train=False, download=True)


if __name__ == "__main__":
    main()
