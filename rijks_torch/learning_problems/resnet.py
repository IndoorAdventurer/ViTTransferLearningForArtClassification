from .defaults import freezeLayers
from torch import nn
from torchvision import models

def get_resnet50_problem(off_the_shelf: bool, dl, pretrained: bool = True):
    """
    Returns the whole problem statement for training resnet50 on the Rijksdataset.
    In other words: a pre-trained model (with the head replaced), and the dataloaders.\n
    :off_the_shelf: says if it should freeze all but the new head for learning.\n
    :dataloaders: allows user to specify custom dataset.\n
    :pretrained: states if it should load a model pretrained om ImageNet.\n
    """
    model = models.resnet50(pretrained=pretrained)
    
    # Prepare for off the shelf learning if needed:
    freezeLayers(model, off_the_shelf)
    
    # Replace head with one that fits the task
    model.fc = nn.Linear(2048, len(dl.materials))

    return model, dl


def get_resnet50_drop_problem(off_the_shelf: bool, dl, pretrained: bool = True):
    """ Same but with a dropout layer. This version is used for fine tuning """
    model, dl = get_resnet50_problem(off_the_shelf, dl, pretrained)
    model.fc = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(2048, len(dl.materials))
    )
    return model, dl