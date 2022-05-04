from .rijksdataset import RijksDataset

import os
import torch
from torch.utils.data import DataLoader
from torch.nn.functional import one_hot
from torchvision.transforms import Lambda
import pandas as pd

class RijksDataloaders:
    """
    Encapsulates/groups dataloaders for the training, validation, and testing set
    """

    def __init__(self, ds_name: str, hist_path: str, img_dir: str,
                    transforms: dict, batch_size: int):
        """
        ## Constructor
        dsName is a path to the dataset such that f'{dsName}-train.csv', f'{dsName}-val.csv',
        and f'{dsName}-test.csv' exist.\n
        histPath is the path to a histogram stating for each material how often it occurs. This
        is used to extract a list of materials, such that a model learns to predict indeces into
        this list.\n
        img_dir is the directory containing all the images.\n
        transforms is a dictionary containing transforms to apply to each
        of the 3 datasets. If one key is defined, the same transform is applied to each dataset.
        If a key 'train' and 'rest' are defined, 'train' is applied for training, and 'rest' for
        testing and validating.\n
        """
        
        # Get a list of possible materials:
        self.materials = pd.read_csv(hist_path)["material"].to_list()

        # Stuff needed to make all datasets:
        info = {
            "ds_name": ds_name,
            "materials": self.materials,
            "img_dir": img_dir,
            "transforms": transforms,
            "batch_size": batch_size
        }

        # Creating the datasets and dataloaders
        self.train = RijksDataloaders.makeDataLoader("train", info)
        self.val   = RijksDataloaders.makeDataLoader("val",   info)
        self.test  = RijksDataloaders.makeDataLoader("test",  info)
    
    @staticmethod
    def makeDataLoader(which: str, info: dict) -> DataLoader:
        """Helper function to create each of the 3 datasets"""

        # Getting the right transform (from key if exists, else the first value in dict):
        key = "train" if which == "train" else "rest"
        t = info["transforms"]
        transform = t[key] if key in t else list(t.values())[0]
        
        # Creating the dataset and dataloader:
        ds = RijksDataset(f"{info['ds_name']}-{which}.csv",
                info['materials'], info['img_dir'], transform)
        return DataLoader(ds, batch_size=info["batch_size"], shuffle=True, num_workers=os.cpu_count())