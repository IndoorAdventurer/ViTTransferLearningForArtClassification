from .defaults import freezeLayers
from torch import nn
import timm

def get_deit_b_16_problem(off_the_shelf: bool, dl, pretrained: bool = True):
    """
    Returns the whole problem statement for training deit_base_16 on the Rijksdataset.
    In other words: a pre-trained model (with the head replaced), and the dataloaders.\n
    :off_the_shelf: says if it should freeze all but the new head for learning.\n
    :dataloaders: allows user to specify custom dataset.\n
    :pretrained: states if it should load a model pretrained om ImageNet.\n
    """
    model = timm.create_model('deit_base_patch16_224', pretrained=pretrained)

    # Prepare for off the shelf learning if needed:
    freezeLayers(model, off_the_shelf)
    
    # Replace head with one that fits the task
    model.head = nn.Linear(768, len(dl.materials))

    return model, dl


def get_deit_b_16_drop_problem(off_the_shelf: bool, dl, pretrained: bool = True):
    """ Same but with a dropout layer. This version is used for fine tuning """
    model, dl = get_deit_b_16_problem(off_the_shelf, dl, pretrained)
    model.head = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(768, len(dl.materials))
    )
    return model, dl


def get_deit_t_16_problem(off_the_shelf: bool, dl, pretrained: bool = True):
    """
    Returns the whole problem statement for training deit_tiny_16 on the Rijksdataset.
    In other words: a pre-trained model (with the head replaced), and the dataloaders.\n
    :off_the_shelf: says if it should freeze all but the new head for learning.\n
    :dataloaders: allows user to specify custom dataset.\n
    :pretrained: states if it should load a model pretrained om ImageNet.\n
    """
    model = timm.create_model('deit_tiny_patch16_224', pretrained=pretrained)

    # Prepare for off the shelf learning if needed:
    freezeLayers(model, off_the_shelf)
    
    # Replace head with one that fits the task
    model.head = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(192, len(dl.materials))
    )

    return model, dl