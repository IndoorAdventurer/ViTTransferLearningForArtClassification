# Contains defaul values that most models will likely use

from torchvision import transforms
from ..data_loading.rijksdataloaders import RijksDataloaders

project_path = "/home/vincent/Documenten/BachelorsProject/"
dataset_fullsize = project_path + "GitHub_Repo/data_annotations/fullsize"
hist_path = project_path + "Rijksdata/csv/subset_hist_data.csv"
img_dir = project_path + "Rijksdata/jpg/"

# TODO create function that generates the transforms automatically:
transformdict = {"all": transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])}

batch_size = 32

dataloaders = RijksDataloaders(dataset_fullsize, hist_path, img_dir, transformdict, batch_size)

def freezeLayers(model, off_the_shelf: bool):
    """Prepare model for off the shelf learning if needed"""
    if off_the_shelf:
        for param in model.parameters():
            param.requires_grad = False