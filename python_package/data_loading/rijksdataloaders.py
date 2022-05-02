from .rijksdataset import RijksDataset

from torch.utils.data import DataLoader
import pandas as pd

class RijksDataloaders:
    """
    Encapsulates/groups dataloaders the training, validation, and testing set
    """

    def __init__(self, ds_name: str, hist_path: str, img_dir: str,
                    transforms: dict, target_transforms: dict, batch_size: int):
        """
        ## Constructor
        dsName is a path to the dataset such that f'{dsName}-train.csv', f'{dsName}-val.csv',
        and f'{dsName}-test.csv' exist.\n
        histPath is the path to a histogram stating for each material how often it occurs. This
        is used to extract a list of materials, such that a model learns to predict indeces into
        this list.\n
        img_dir is the directory containing all the images.\n
        transforms and target_transforms are dictionaries containing transforms to apply to each
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
            "target_transforms": target_transforms,
            "batch_size": batch_size
        }

        # Creating the datasets and dataloaders
        self.train = RijksDataloaders.makeDataLoader("train", info)
        self.val   = RijksDataloaders.makeDataLoader("val",   info)
        self.test  = RijksDataloaders.makeDataLoader("test",  info)
    
    @staticmethod
    def makeDataLoader(which: str, info: dict) -> DataLoader:
        """Helper function to create each of the 3 datasets"""

        # Getting the right transforms (from key if exists, else the first value in dict):
        key = "train" if which == "train" else "rest"
        transform = info["transforms"][key] if key in info["transforms"] \
            else list(info["transforms"].values())[0]
        target_transform = info["target_transforms"][key] if key in info["target_transforms"] \
            else list(info["target_transforms"].values())[0]
        
        # Creating the dataset and dataloader:
        ds = RijksDataset(f"{info['ds_name']}-{which}.csv", info['materials'], info['img_dir'], transform, target_transform)
        return DataLoader(ds, batch_size=info["batch_size"], shuffle=True)