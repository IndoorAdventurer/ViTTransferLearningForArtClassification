from .defaults import dataloaders, freezeLayers, RijksDataloaders
from torch import nn
from torchvision import models
from torch.nn import Module

def vgg19(off_the_shelf: bool, dl: RijksDataloaders = dataloaders) -> tuple[Module, RijksDataloaders]:
    """
    Returns the whole problem statement for training vgg19 on the Rijksdataset.
    In other words: a pre-trained model with the head replaced, and the dataloaders
    """
    model = models.vgg19(pretrained=True)
    
    # Prepare for off the shelf learning if needed:
    freezeLayers(model, off_the_shelf)
    
    # Replace head with one that fits the task
    model.classifier[6] = nn.Linear(4096, len(dl.materials))

    return model, dl