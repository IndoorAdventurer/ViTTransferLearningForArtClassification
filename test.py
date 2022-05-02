from cProfile import label
from python_package.data_loading import RijksDataloaders

import torch
from torchvision import transforms

import matplotlib.pyplot as plt
import numpy as np

path = "/home/vincent/Documenten/BachelorsProject/"

dls = RijksDataloaders(
    ds_name = path + "GitHub_Repo/data_annotations/fullsize",
    hist_path = path + "Rijksdata/csv/subset_hist_data.csv",
    img_dir = path + "Rijksdata/jpg/",
    transforms = {"all": transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224)
    ])},
    target_transforms = {"train": None, "rest": transforms.Lambda(lambda y: torch.zeros(30, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))},
    batch_size = 4
)

itr = iter(dls.train)
for X, Y in itr:
    for x, y in zip(X, Y):
        plt.imshow(x.numpy().transpose(1,2,0))
        plt.title(dls.materials[y])
        plt.show()