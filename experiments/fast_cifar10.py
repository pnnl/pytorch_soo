"""
An attempt to get the loading to be faster, from here:
https://gist.github.com/y0ast/f69966e308e549f013a92dc66debeeb4
"""

from warnings import warn

# Needed to fix the 403 error, comes from here:
# https://github.com/pytorch/vision/issues/1938#issuecomment-789986996
from six.moves import urllib

opener = urllib.request.build_opener()
opener.addheaders = [("User-agent", "Mozilla/5.0")]
urllib.request.install_opener(opener)

import torch
from torchvision.datasets import CIFAR10

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    warn("No GPU present, defaulting to CPU")
    DEVICE = torch.device("cpu")
CIFAR10_MEAN = torch.tensor((0.4914, 0.4822, 0.4465)).to(DEVICE)
CIFAR10_STD_DEV = torch.tensor((0.2023, 0.1994, 0.2010)).to(DEVICE)


class FastCIFAR10(CIFAR10):
    def __init__(self, *args, resnet_permute=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = torch.from_numpy(self.data)
        self.targets = torch.tensor(self.targets)
        self.data, self.targets = self.data.to(DEVICE), self.targets.to(DEVICE)
        # Scale data to [0,1]
        # self.data = self.data.unsqueeze(1).float().div(255)
        self.data = self.data.float().div(255)

        # Normalize it with the usual CIFAR10 mean and std
        self.data = self.data.sub(CIFAR10_MEAN).div(CIFAR10_STD_DEV)
        # Permute it to match the expected ResNET50 input shape(s)
        if resnet_permute:
            self.data = self.data.permute(0, 3, 1, 2)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        return img, target
