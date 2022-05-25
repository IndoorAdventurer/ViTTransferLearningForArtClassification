from .defaults import freezeLayers
from torch import nn
import timm

def get_swin_b_problem(off_the_shelf: bool, dl, pretrained: bool = True):
    """
    Returns the whole problem statement for training swin_b on the Rijksdataset.
    In other words: a pre-trained model (with the head replaced), and the dataloaders.\n
    :off_the_shelf: says if it should freeze all but the new head for learning.\n
    :dataloaders: allows user to specify custom dataset.\n
    :pretrained: states if it should load a model pretrained om ImageNet.\n
    """
    model = timm.create_model('swin_base_patch4_window7_224', pretrained=pretrained)

    # Prepare for off the shelf learning if needed:
    freezeLayers(model, off_the_shelf)
    
    # Replace head with one that fits the task
    model.head = nn.Linear(1024, len(dl.materials))

    return model, dl


def get_swin_s_problem(off_the_shelf: bool, dl, pretrained: bool = True):
    """
    Returns the whole problem statement for training swin_s on the Rijksdataset.
    In other words: a pre-trained model (with the head replaced), and the dataloaders.\n
    :off_the_shelf: says if it should freeze all but the new head for learning.\n
    :dataloaders: allows user to specify custom dataset.\n
    :pretrained: states if it should load a model pretrained om ImageNet.\n
    """
    model = timm.create_model('swin_small_patch4_window7_224', pretrained=pretrained)

    # Prepare for off the shelf learning if needed:
    freezeLayers(model, off_the_shelf)
    
    # Replace head with one that fits the task
    model.head = nn.Linear(768, len(dl.materials))

    return model, dl


def get_swin_t_problem(off_the_shelf: bool, dl, pretrained: bool = True):
    """
    Returns the whole problem statement for training swin_t on the Rijksdataset.
    In other words: a pre-trained model (with the head replaced), and the dataloaders.\n
    :off_the_shelf: says if it should freeze all but the new head for learning.\n
    :dataloaders: allows user to specify custom dataset.\n
    :pretrained: states if it should load a model pretrained om ImageNet.\n
    """
    model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=pretrained)

    # Prepare for off the shelf learning if needed:
    freezeLayers(model, off_the_shelf)
    
    # Replace head with one that fits the task
    model.head = nn.Linear(768, len(dl.materials))

    return model, dl