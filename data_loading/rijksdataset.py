from ctypes.wintypes import RGB
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
import pandas as pd
from os import path

class RijksDataset(Dataset):
    """A class that encapsulates the Rijksmuseum Challenge dataset."""
    
    def __init__(self, csv_file, materials, img_dir, transform=None, target_transform=None):
        """
        ## Default constructor

        :param csv_file:  A file containing [*.jpg, "material"] pairs for each element in the dataset
        :param materials: A list containing all the materials, such that a ML model can learn to
                    predict indeces into this list
        :param img_dir:   Directory containing all .jpg files mentioned in :csv_file:
        :param transform and target_transform: apply transforms to input and output resp.
        """
        self._df = RijksDataset._processTable(csv_file, materials)
        self._img_dir = img_dir
        self._transform = transform
        self._target_transform = target_transform

        # Better to check now than to find out while training:
        for jpg in self._df["jpg"]:
            if not path.exists(path.join(img_dir, jpg)):
                raise Exception(f"The file '{jpg}' was not found in {img_dir}.")
    
    def _processTable(csv_file, materials):
        """
        Returns pd.DataFrame containing [*.jpg, idx] pairs, such that materials[idx] == 'material'.
        """
        df = pd.read_csv(csv_file)

        try:
            df["idx"] = df["material"].map(lambda mat: materials.index(mat))
        except Exception as e:
            raise Exception("Can't map all material strings to indeces.") from e

        df.drop(columns="material", inplace=True)

        return df
    
    def __len__(self):
        """Returns number of samples in dataset"""
        return len(self._df)
    
    def __getitem__(self, idx):
        """Get x (image) and y (material index into materials list) at idx"""
        x = read_image(
            path = path.join(self._img_dir, self._df.loc[idx, "jpg"]),
            mode = ImageReadMode.RGB
        )
        if self._transform:
            x = self._transform(x)
        
        y = self._df.loc[idx, "idx"]
        if self._target_transform:
            y = self._target_transform(y)
        
        return x, y