from .defaults import freezeLayers
from torch import nn
from torchvision import models

def get_convnext_t_problem(off_the_shelf: bool, dl, pretrained: bool = True):
    """
    Returns the whole problem statement for training convnext_tiny on the Rijksdataset.
    In other words: a pre-trained model (with the head replaced), and the dataloaders.\n
    :off_the_shelf: says if it should freeze all but the new head for learning.\n
    :dataloaders: allows user to specify custom dataset.\n
    :pretrained: states if it should load a model pretrained om ImageNet.\n
    """
    model = models.convnext_tiny(pretrained=pretrained)
    
    # Prepare for off the shelf learning if needed:
    freezeLayers(model, off_the_shelf)
    
    # Replace head with one that fits the task
    model.classifier[2] = nn.Linear(768, len(dl.materials))

    return model, dl