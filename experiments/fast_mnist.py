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
from torchvision.datasets import MNIST

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    warn("No GPU present, defaulting to CPU")
    DEVICE = torch.device("cpu")


class FastMNIST(MNIST):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Scale data to [0,1]
        self.data = self.data.unsqueeze(1).float().div(255)

        # Normalize it with the usual MNIST mean and std
        self.data = self.data.sub_(0.1307).div_(0.3081)

        # Put both data and targets on GPU in advance
        self.data, self.targets = self.data.to(DEVICE), self.targets.to(DEVICE)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        return img, target
